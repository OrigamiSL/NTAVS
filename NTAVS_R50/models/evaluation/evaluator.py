import os
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
import numpy as np
import PIL.Image as image
import matplotlib.pyplot as plt
from models.evaluation.misc.visual import mean_iou, COLOR_MAP, COLOR_MAP_SS
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs, if_train, dataset_name):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs, if_train, dataset_name)

    def evaluate(self,if_train, dataset_name):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate(if_train, dataset_name)
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert k not in results, "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results
    
    def process_group(self, inputs, outputs, dataset_name):
        for evaluator in self._evaluators:
            evaluator.process_group(inputs, outputs, dataset_name)

    def evaluate_group(self, dataset_name):
        for evaluator in self._evaluators:
            evaluator.evaluate_group(dataset_name)

    def process_recover(self, inputs, iter_):
        for evaluator in self._evaluators:
            evaluator.process_recover(inputs, iter_)

    def evaluate_recover(self, iter_):
        for evaluator in self._evaluators:
            evaluator.evaluate_recover(iter_)

def inference_on_dataset(
    model,
    data_loader,
    evaluator: Union[
        DatasetEvaluator,
        List[DatasetEvaluator],
        None,
    ],
    output_dir=None,
    dataset_name=None, 
    if_train=True,
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger("NTAVS")
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    query_norm = torch.nn.LayerNorm(256, elementwise_affine = False)
    query_norm_1 = torch.nn.LayerNorm(144, elementwise_affine = False)

    # for iter_array in range(0, 200):
    #     evaluator.reset()
    #     for idx, inputs in enumerate(data_loader):
    #         evaluator.process_recover(inputs, iter_array*100+99)
    #     evaluator.evaluate_recover(iter_array*100+99)
    # exit(-1)
#     [[6.3850e-01 7.4420e-01 1.0599e+04]
#  [6.4280e-01 7.3420e-01 1.1299e+04]
#  [6.3960e-01 7.4080e-01 6.0990e+03]
#  [6.4490e-01 7.3860e-01 6.1990e+03]
#  [6.4010e-01 7.3750e-01 5.1990e+03]
#  [6.4290e-01 7.4180e-01 5.7990e+03]
#  [6.4490e-01 7.3860e-01 1.9999e+04]
#  [6.3910e-01 7.3730e-01 1.7299e+04]
#  [6.3950e-01 7.4450e-01 4.9990e+03]]

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        video_miou = {}
        vis = True

        all_hist = []

        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs, return_attn_map, query_map = model(inputs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs, if_train, dataset_name)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                current_time = time.time()
                n= 5
                log_time = time.time() if 'log_time' not in locals() else log_time
                if current_time - log_time >= n:
                    logger.info(
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}")
                    log_time = time.time()
                
            start_data_time = time.perf_counter()

            if vis:
                num_video = -1
                for num_img, output in enumerate(outputs):
                    output = output["sem_seg"].argmax(dim=0).to(evaluator._cpu_device)
                    pred = np.array(output, dtype=np.uint8)

                    if num_img % 5 == 0:
                        num_video += 1
                        file_names = inputs[num_video]["file_names"]
                        # print(file_names)
                        video_name = file_names[0].split("AVSBench_object")[1].split("/")[:-1]
                        # print(video_name)
                        video_name = "/".join(video_name)
                        video_miou[video_name] = []
                        gt_filenames = evaluator.input_file_to_gt_file[tuple(file_names)]

                    gt = evaluator.sem_seg_loading_fn(gt_filenames[num_img % 5], dtype=int)
                    gt[gt == evaluator._ignore_label] = evaluator._num_classes
                    per_img_iou = mean_iou(pred, gt, classes=evaluator._num_classes)
                    video_miou[video_name].append(per_img_iou)
                    pred_img_path = output_dir + "/vis/" + file_names[num_img % 5].split("AVSBench_object")[1]
                    dir_name = os.path.dirname(pred_img_path)
                    os.makedirs(dir_name, exist_ok=True)
                    label_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                    for i, rgb in zip(range(evaluator._num_classes), COLOR_MAP):
                        label_rgb[pred == i] = rgb
                    image.fromarray(label_rgb).save(pred_img_path)

                # file_names = inputs[0]["file_names"]
                # pred_img_path_query = output_dir + "/vis_attn/" + 'query_map'+ file_names[0].split("AVSBench_object")[1]
                # pred_img_path_query = pred_img_path_query.split(".")[0]+'.svg'
                # print(pred_img_path_query)
                # dir_name = os.path.dirname(pred_img_path_query)
                # os.makedirs(dir_name, exist_ok=True)
                
                # frame_number = query_map.shape[0] # 144, 5, 256
                # before_ = query_norm(query_map[:72, :, :])
                # after_ = query_map[72:, :, :]
                # query_map = torch.cat([before_, after_], dim = 0)

                # # query_map = query_norm_1(query_map.permute(1, 2, 0))
                # # query_map = query_map.permute(2, 0, 1)
                # # query_map = query_norm(query_map)

                # query_map = query_map[:, 0, :]
                
                # mid_feat = query_map.reshape(frame_number, -1)
                
                # sim_matrix = np.zeros([frame_number,frame_number])
                # sim_matrix_color = np.zeros([frame_number,frame_number, 4])
                # for i in range(frame_number):
                #     for j in range(frame_number):  
                #         cos_sim = F.cosine_similarity(mid_feat[i], mid_feat[j], dim = 0)
                #         cos_sim = cos_sim.item()

                #         # if cos_sim > 0.8:
                #         #     cos_sim -= 0.8
                #         # print(cos_sim)
                #         if i == j:
                #             # print(cos_sim)
                #             # sim_matrix[i, j] = 1
                #             sim_matrix[i, j] = 1
                #             pass
                #         else:
                #             sim_matrix[i, j] = cos_sim

                # # scaler = StandardScaler()
                # # scaler.fit(sim_matrix)
                # # sim_matrix = scaler.transform(sim_matrix)
                
                # for i in range(frame_number):
                #     for j in range(frame_number):
                #         color = plt.get_cmap('rainbow')(sim_matrix[i, j])
                #         sim_matrix_color[i, j, :] = color          

                # print(sim_matrix_color.shape)
                # plt.imshow((sim_matrix_color), cmap= 'rainbow')
                # plt.xticks([])
                # plt.yticks([])
            
                # cbar = plt.colorbar()
                # cbar.ax.tick_params(labelsize=20)

                # plt.savefig(pred_img_path_query, bbox_inches = 'tight', 
                #             pad_inches = 0.1, transparent = True, dpi = 1000, format = 'svg')
                # plt.close()
        

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results, iter_array = evaluator.evaluate(if_train, dataset_name)
    results_ = {}

    if not isinstance(iter_array, int):
        evaluator.reset()
        for idx, inputs in enumerate(data_loader):
            evaluator.process_group(inputs, iter_array, dataset_name)
        results_ = evaluator.evaluate_group(dataset_name)
    
    results = {**results, **results_}

    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle

    if results is None:
        results = {}
    return results


def inference_on_dataset_ss(
    model,
    data_loader,
    evaluator: Union[
        DatasetEvaluator,
        List[DatasetEvaluator],
        None,
    ],
    output_dir=None,
    dataset_name=None, 
    if_train=True
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger("NTAVS")
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())
        start_data_time = time.perf_counter()

        vis = True

        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs, return_attn_map, sim_hist = model(inputs)
            # print('input', inputs.shape, 'outputs', outputs.shape)

            # * outputs visualization
            if vis:
                num_video = -1
                for num_img, output in enumerate(outputs):
                    output = output["sem_seg"].argmax(dim=0).to(evaluator._cpu_device)
                    pred = np.array(output, dtype=np.uint8)
                    # print('pred', pred.shape) # (224, 224)

                    if num_img % len(outputs) == 0:
                        num_video += 1
                        file_names = inputs[num_video]["file_names"]
                        video_name = file_names[0].split("AVSBench_semantic")[1].split("/")[:-1]
                        video_name = "/".join(video_name)

                    pred_img_path = output_dir + "/vis/" + file_names[num_img % len(outputs)].split("AVSBench_semantic")[1]
                    dir_name = os.path.dirname(pred_img_path)
                    os.makedirs(dir_name, exist_ok=True)
                    label_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                    for i, rgb in zip(range(evaluator._num_classes), COLOR_MAP_SS):
                        # print('rgb', rgb.shape) # 3
                        label_rgb[pred == i] = rgb
                    image.fromarray(label_rgb).save(pred_img_path)

                # pred_img_path = output_dir + "/vis_attn/" + file_names[0].split("AVSBench_semantic")[1]
                # dir_name = os.path.dirname(pred_img_path)
                # os.makedirs(dir_name, exist_ok=True)
                # normed_mask = (return_attn_map)
                # # print(normed_mask)
                # plt.imshow(normed_mask, cmap= 'rainbow')
                # n_f = return_attn_map.shape[0]
                # plt.xticks([i for i in range(n_f)], [i+1 for i in range(n_f)], fontsize = 20)
                # plt.yticks([i for i in range(n_f)], [i+1 for i in range(n_f)], fontsize = 20)

                # for i in range(n_f):
                #     plt.hlines(i-0.5, -0.5, n_f-0.5, color="black", linestyles= 'dashed')#横线
                #     plt.vlines(i-0.5, -0.5, n_f-0.5, color="black", linestyles= 'dashed')#竖线
                # # plt.axis('off')
                # cbar = plt.colorbar()
                # cbar.ax.tick_params(labelsize=20)
                # # plt.axis('off')
                # plt.savefig(pred_img_path, bbox_inches = 'tight', pad_inches = 0.1, transparent = True)
                # plt.close()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs, if_train, dataset_name)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                current_time = time.time()
                n= 5
                log_time = time.time() if 'log_time' not in locals() else log_time
                if current_time - log_time >= n:
                    logger.info(
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}")
                    log_time = time.time()
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results, iter_array = evaluator.evaluate(if_train, dataset_name)
    results_ = {}
    
    with open('iter.txt', 'r') as f:
            iter = f.readlines()
            iter = iter[0]

    if if_train == False or (int(iter) - 99) % 500 == 0:
        if not isinstance(iter_array, int):
            evaluator.reset()
            for idx, inputs in enumerate(data_loader):
                evaluator.process_group(inputs, iter_array, dataset_name)
            results_ = evaluator.evaluate_group(dataset_name)
        
        results = {**results, **results_}
           
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
