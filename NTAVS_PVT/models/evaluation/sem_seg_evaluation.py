import itertools
import json
import logging
import numpy as np
import os
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional, Union
import pycocotools.mask as mask_util
import torch
import shutil
from PIL import Image
from itertools import zip_longest
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation.evaluator import DatasetEvaluator

_CV2_IMPORTED = True
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    _CV2_IMPORTED = False

# load img to np_array, transform to (0,1)
def load_image_into_numpy_array(
    filename: str,
    copy: bool = False,
    dtype: Optional[Union[np.dtype, str]] = None,
) -> np.ndarray:
    with PathManager.open(filename, "rb") as f:
        array = np.array(Image.open(f), copy=copy, dtype=dtype)
        array = array // 255  # * AVSS4 dataset, change object 255 to 1.
    return array


class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v

def mask_iou(
    pred,
    target,
    eps=1e-7,
):
    r"""
    param:
        pred: size [N x H x W]
        target: size [N x H x W]
    output:
        iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)
    pred = (pred > 0.5).int()
    num_pixels = pred.size(-1) * pred.size(-2)

    # one of N have no objects, then whole image is set 0, shall be calculated in inter and union
    no_obj_flag = target.sum(2).sum(1) == 0

    inter = (pred * target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1)

    inter[no_obj_flag] = inter_no_obj[no_obj_flag]
    union[no_obj_flag] = num_pixels

    iou = torch.sum(inter / (union + eps)) / N                                  
    return iou


def _eval_pr(y_pred, y, num, cuda_flag=True):
    if cuda_flag:
        prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
        thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    else:
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        # True predict / total predict, True predict / total gts
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)

    return prec, recall

def Eval_Fmeasure(pred, gt, pr_num=255):
    r"""
    param:
        pred: size [N x H x W]
        gt: size [N x H x W]
    output:
        iou: size [1] (size_average=True) or [N] (size_average=False)
    """

    N = pred.size(0)
    beta2 = 0.3

    avg_f, img_num = 0.0, 0
    score = torch.zeros(pr_num)

    for img_id in range(N):
        # examples with totally black GTs are out of consideration
        if torch.mean(gt[img_id]) == 0.0:
            # if gt[img_id].sum() == 0.0:
            continue

        # calculate precision and recall
        prec, recall = _eval_pr(pred[img_id], gt[img_id], pr_num)

        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score[f_score != f_score] = 0  # for Nan
        avg_f += f_score
        img_num += 1
        score = avg_f / img_num
        # print('score: ', score)

    # the max threshold value in pr_number
    return score.max().item()

class SemSegEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        sem_seg_loading_fn=load_image_into_numpy_array,
        num_classes=None,
        ignore_label=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            sem_seg_loading_fn: function to read sem seg file and load into numpy array.
                Default provided, but projects can customize.
            num_classes, ignore_label: deprecated argument (abondened)
        """
        self._logger = logging.getLogger(__name__)
        if num_classes is not None:
            self._logger.warn("SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata.")
        if ignore_label is not None:
            self._logger.warn("SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata.")
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self.ckp_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self.input_file_to_gt_file = {}
        self.input_file_to_gt_file = {
            tuple(dataset_record["file_names"]): dataset_record["sem_seg_file_names"] for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = meta.stuff_classes
        self.sem_seg_loading_fn = sem_seg_loading_fn
        self._num_classes = len(meta.stuff_classes)
        if num_classes is not None:
            assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

        # This is because cv2.erode did not work for int datatype. Only works for uint8.
        self._compute_boundary_iou = True
        if not _CV2_IMPORTED:
            self._compute_boundary_iou = False
            self._logger.warn(
                """Boundary IoU calculation requires OpenCV. B-IoU metrics are
                not going to be computed because OpenCV is not available to import."""
            )
        if self._num_classes >= np.iinfo(np.uint8).max:
            self._compute_boundary_iou = False
            self._logger.warn(
                f"""SemSegEvaluator(num_classes) is more than supported value for Boundary IoU calculation!
                B-IoU metrics are not going to be computed. Max allowed value (exclusive)
                for num_classes for calculating Boundary IoU is {np.iinfo(np.uint8).max}.
                The number of classes of dataset {self._dataset_name} is {self._num_classes}"""
            )
        
        # AverageMeter: get average value
        self.miou = AverageMeter("miou")
        self.f_score = AverageMeter("f_score")
        self.result_all = []

    def reset(self):
        self.miou = AverageMeter("miou")
        self.f_score = AverageMeter("f_score")
        self._predictions = []

    def process(self, inputs, outputs, if_train, dataset_name):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        num_video = -1
        for num_img, output in enumerate(outputs):
            output = output["sem_seg"]  # .argmax(dim=0)
            if num_img % 5 == 0:
                num_video += 1
                if num_img == 0:
                    gts = inputs[num_video]["sem_segs"].squeeze(dim=1).cuda()
                else:
                    gts = torch.cat((gts, inputs[num_video]["sem_segs"].squeeze(dim=1).cuda()), dim=0)
            if num_img == 0:
                preds = output.unsqueeze(dim=0)
            else:
                preds = torch.cat((preds, output.unsqueeze(dim=0)), dim=0)

        preds = F.softmax(preds, dim=1)

        file_name = inputs[num_video]["file_names"][0]
        path_img_dir = file_name.split('AVSBench_object')[1]
        path_img_dir = path_img_dir.split('.')[0]

        save_imgg_path = path_img_dir.split('/')[-1]

        with open('iter.txt', 'r') as f:
            iter = f.readlines()
            iter = iter[0]
            # print(iter)
            # f.close()
        save_root = './'+dataset_name+'_'+'result_npy'
        # print(iter)

        # print('path_img_dir', save_imgg_path)
        save_path = os.path.join(save_root, str(iter), save_imgg_path)
        # print('save_path', save_path)
        os.makedirs(save_path, exist_ok= True)
        pre_result = preds[:, 1] # [5, 224, 224]
        # print('pre_result', pre_result.shape)
        pred_path = os.path.join(save_path, 'pred.npy')
        np.save(pred_path, pre_result.cpu().numpy())

        # print(path_img_dir)
        miou = mask_iou(preds[:, 1], gts)
        fscore = Eval_Fmeasure(preds[:, 1], gts)

        self.result_all.append([path_img_dir, miou.item(), round(fscore,4)])
        # s4_best, 66299

        # add: add value to dictionary
        self.miou.add({"miou": miou})
        self.f_score.add({"f_score": fscore})

    def evaluate(self, if_train, dataset_name):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            # synchronzie all processes
            synchronize()

            # all_gather: list[data]: list of data gathered from each rank
            self._predictions = all_gather(self._predictions)
            # list(itertools.chain(*xxx)) : remove embedded list, e.g., [[1],[2]]-> [1, 2]
            self._predictions = list(itertools.chain(*self._predictions))
            miou_list = all_gather(self.miou.pop("miou"))
            miou = torch.tensor(miou_list).mean().item()
            f_score_list = all_gather(self.f_score.pop("f_score"))
            f_score = torch.tensor(f_score_list).mean().item()

            if not is_main_process():
                return

        res = {}
        # round: half adjust
        res["mIoU"] = round(miou, 4)  
        res["f_score"] = round(f_score, 4)


        with open('iter.txt', 'r') as f:
            iter = f.readlines()
            iter = iter[0]

        save_path_miou = './'+dataset_name+'_'+'save_best_miou.npy'

        top_k = 10

        best_m, best_f, best_iter, iter_array = 0, 0, 0, 0
        save_root = './'+dataset_name+'_'+'result_npy'

        if not os.path.exists(save_path_miou):
            save_array = np.zeros([1, 3])
            save_array[0, :] = np.array([res["mIoU"], res["f_score"], int(iter)])
            np.save(save_path_miou, save_array)

        else:
            save_array = np.load(save_path_miou)
            length = save_array.shape[0]
            if  length < top_k:
                new_array = np.zeros([length + 1, 3])

                new_array[:-1, :] = save_array
                new_array[-1, :] = np.array([res["mIoU"], res["f_score"], int(iter)])
                np.save(save_path_miou, new_array)
            else:
                pop_index = np.argmin(save_array, axis = 0)
                # print('pop_index', pop_index)
                pop_index_mIoU = pop_index[0]
                old_mIoU = save_array[pop_index_mIoU, 0]
                old_iter = save_array[pop_index_mIoU, 2]

                
                if if_train:
                    if res["mIoU"] > old_mIoU:
                        shutil.rmtree(os.path.join(save_root, str(int(old_iter))))
                        shutil.os.remove(os.path.join(self.ckp_dir, 
                                                      'model_'+'%07d'%int(old_iter)+'.pth'))

                        save_array[pop_index_mIoU, :] = np.array([res["mIoU"], res["f_score"], int(iter)])
                    else:
                        shutil.rmtree(os.path.join(save_root, str(int(iter))))
                        shutil.os.remove(os.path.join(self.ckp_dir, 
                                                      'model_'+'%07d'%int(iter)+'.pth'))
                        
                    best_index = np.argmax(save_array, axis = 0)
                    best_index_mIoU = best_index[0]
                    best_m, best_f, best_iter =\
                    save_array[best_index_mIoU, 0], save_array[best_index_mIoU, 1], save_array[best_index_mIoU, 2]
                    np.save(save_path_miou, save_array)

                iter_array = save_array[:, 2]


        
        os.makedirs(self._output_dir, exist_ok=True)


        # OrderedDict: Ordered Dict
        results = OrderedDict({"sem_seg": res,
                               'best_miou_iter_iou': best_m,
                               'best_miou_iter_fscore': best_f,
                               'best_iter': best_iter,

                               })
        self._logger.info(results)
        

        return results, iter_array
    
    def process_group(self, inputs, iter_array, dataset_name):

        num_video = -1
        for num_img, xx in enumerate(inputs):

            num_video += 1
            if num_img == 0:
                gts = inputs[num_video]["sem_segs"].squeeze(dim=1).cuda()
            else:
                gts = torch.cat((gts, inputs[num_video]["sem_segs"].squeeze(dim=1).cuda()), dim=0)

        file_name = inputs[num_video]["file_names"][0]
        path_img_dir = file_name.split('AVSBench_object')[1]
        path_img_dir = path_img_dir.split('.')[0]

        save_imgg_path = path_img_dir.split('/')[-1]


        top_k = 10
        judge = top_k // 2
        all_result_mIoU = 0
        all_result_fscore = 0

        for itt in iter_array:
            save_root = './'+dataset_name+'_'+'result_npy'
            save_path = os.path.join(save_root, str(int(itt)), save_imgg_path)
            pred_path = os.path.join(save_path, 'pred.npy')

            now_npy = np.load(pred_path)
            test_tensor = torch.Tensor(now_npy)
            test_tensor = test_tensor.cuda()

            all_result_mIoU += test_tensor
            all_result_fscore += test_tensor

        all_result_mIoU = all_result_mIoU / top_k 
        all_result_fscore = all_result_fscore / top_k


        miou = mask_iou(all_result_mIoU, gts)
        fscore = Eval_Fmeasure(all_result_fscore, gts)

        self.miou.add({"miou": miou})
        self.f_score.add({"f_score": fscore})

    def evaluate_group(self, dataset_name):

        if self._distributed:

            synchronize()

            self._predictions = all_gather(self._predictions)

            self._predictions = list(itertools.chain(*self._predictions))
            miou_list = all_gather(self.miou.pop("miou"))
            miou = torch.tensor(miou_list).mean().item()
            f_score_list = all_gather(self.f_score.pop("f_score"))
            f_score = torch.tensor(f_score_list).mean().item()

            if not is_main_process():
                return

        res = {}
        # round: half adjust
        res["mIoU"] = round(miou, 4)  
        res["f_score"] = round(f_score, 4)

        print('Vote', res["mIoU"], res["f_score"])
        # OrderedDict: Ordered Dict
        results = OrderedDict({
                               'Vote_best_miou_iter_iou': res["mIoU"],
                               'Vote_best_miou_iter_fscore': res["f_score"],
                               })
        self._logger.info(results)
        return results
        # return results, iter_array


    def process_recover(self, inputs, iter_):

        num_video = -1
        for num_img, xx in enumerate(inputs):

            num_video += 1
            if num_img == 0:
                gts = inputs[num_video]["sem_segs"].squeeze(dim=1).cuda()
            else:
                gts = torch.cat((gts, inputs[num_video]["sem_segs"].squeeze(dim=1).cuda()), dim=0)

        file_name = inputs[num_video]["file_names"][0]
        path_img_dir = file_name.split('AVSBench_object')[1]
        path_img_dir = path_img_dir.split('.')[0]

        save_imgg_path = path_img_dir.split('/')[-1]

        
        save_root = './result_npy'
        save_path = os.path.join(save_root, str(int(iter_)), save_imgg_path)
        pred_path = os.path.join(save_path, 'pred.npy')

        now_npy = np.load(pred_path)
        test_tensor = torch.Tensor(now_npy)
        test_tensor = test_tensor.cuda()

        miou = mask_iou(test_tensor, gts)
        fscore = Eval_Fmeasure(test_tensor, gts)

        self.miou.add({"miou": miou})
        self.f_score.add({"f_score": fscore})

    def evaluate_recover(self, iter_):

        if self._distributed:

            synchronize()

            self._predictions = all_gather(self._predictions)

            self._predictions = list(itertools.chain(*self._predictions))
            miou_list = all_gather(self.miou.pop("miou"))
            miou = torch.tensor(miou_list).mean().item()
            f_score_list = all_gather(self.f_score.pop("f_score"))
            f_score = torch.tensor(f_score_list).mean().item()

            if not is_main_process():
                return

        res = {}
        # round: half adjust
        res["mIoU"] = round(miou, 4)  
        res["f_score"] = round(f_score, 4)

        save_path_miou = './save_best_miou_recover.npy'
        top_k = 10

        print('iter:', iter_)

        if not os.path.exists(save_path_miou):
            save_array = np.zeros([1, 3])
            save_array[0, :] = np.array([res["mIoU"], res["f_score"], int(iter_)])
            np.save(save_path_miou, save_array)

        else:
            save_array = np.load(save_path_miou)
            length = save_array.shape[0]
            if  length < top_k:
                new_array = np.zeros([length + 1, 3])
                # print(new_array.shape, save_array.shape)
                new_array[:-1, :] = save_array
                new_array[-1, :] = np.array([res["mIoU"], res["f_score"], int(iter_)])
                np.save(save_path_miou, new_array)
            else:
                pop_index = np.argmin(save_array, axis = 0)

                pop_index_mIoU = pop_index[0]
                old_mIoU = save_array[pop_index_mIoU, 0]

                if res["mIoU"] > old_mIoU:
                    save_array[pop_index_mIoU, :] = np.array([res["mIoU"], res["f_score"], int(iter_)])
            
                np.save(save_path_miou, save_array)
        

    # ... 
    def encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results

        Args:
            sem_seg: predicted semantic segmentation of shape (H, W).
            input_file_name: the file name of the image.
        Returns:
            list[dict]: encoded prediction strings in COCO format.
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert label in self._contiguous_id_to_dataset_id, "Label {} is not in the metadata info for {}".format(
                    label, self._dataset_name
                )
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append({"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle})
        return json_list

    # get erode boundary
    def _mask_to_boundary(self, mask: np.ndarray, dilation_ratio=0.02):
        assert mask.ndim == 2, "mask_to_boundary expects a 2-dimensional image"
        h, w = mask.shape
        diag_len = np.sqrt(h**2 + w**2)
        dilation = max(1, int(round(dilation_ratio * diag_len)))
        kernel = np.ones((3, 3), dtype=np.uint8)

        padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        # cv2.erode: erode boundary, reduce noise
        eroded_mask_with_padding = cv2.erode(padded_mask, kernel, iterations=dilation)
        eroded_mask = eroded_mask_with_padding[1:-1, 1:-1]
        boundary = mask - eroded_mask
        return boundary
