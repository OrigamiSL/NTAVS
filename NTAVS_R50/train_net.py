"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    # SemSegEvaluator,
    verify_results,
    DatasetEvaluators,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger


from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)

# MaskFormer
from models import (
    BestCheckpointer,
    SemSegEvaluator,
    SemSegEvaluator_SS,
    AVSS4_SemanticDatasetMapper,
    AVSMS3_SemanticDatasetMapper,
    AVSS_SemanticDatasetMapper,
    add_maskformer2_config,
    add_audio_config,
    add_fuse_config,
    inference_on_dataset,
    inference_on_dataset_ss,
)

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    # classmethod: not needed to be instantiated, can be called directly
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            # output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            output_folder = os.path.join(cfg.OUTPUT_DIR)

        evaluator_list = []
        # MetadataCatalog is a global dictionary that provides access to
        # :class:`Metadata` of a given dataset.
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        # semantic segmentation
        if evaluator_type in ["sem_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        elif evaluator_type in ["sem_seg_ss"]:
            evaluator_list.append(
                SemSegEvaluator_SS(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        
        return DatasetEvaluators(evaluator_list)
    
    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "avss4_semantic":
            mapper = AVSS4_SemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper) 
        elif cfg.INPUT.DATASET_MAPPER_NAME == "avsms3_semantic":
            mapper = AVSMS3_SemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper) 
        elif cfg.INPUT.DATASET_MAPPER_NAME == "avss_semantic":
            mapper = AVSS_SemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper) 
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # Semantic segmentation dataset mapper
      
        if cfg.INPUT.DATASET_MAPPER_NAME == "avss4_semantic":
            mapper = AVSS4_SemanticDatasetMapper(cfg, is_train=False)
            return build_detection_test_loader(cfg, mapper=mapper, dataset_name=dataset_name)   
        
        elif cfg.INPUT.DATASET_MAPPER_NAME == "avsms3_semantic":
            mapper = AVSMS3_SemanticDatasetMapper(cfg, is_train=False)
            return build_detection_test_loader(cfg, mapper=mapper, dataset_name=dataset_name)
           
        elif cfg.INPUT.DATASET_MAPPER_NAME == "avss_semantic":
            mapper = AVSS_SemanticDatasetMapper(cfg, is_train=False)
            return build_detection_test_loader(cfg, mapper=mapper, dataset_name=dataset_name)         
        
    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        # use the WarmupPolyLR:

        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM # 0
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED # 0

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR # 0.0001
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY #0.05

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                # for backbone: lr * 0.1 = 0.00001;
                # for norm, embedding, elative_position_bias_table, absolute_pos_embed, weight_decay = 0
                # the others weight decay = 0.05
                hyperparams = copy.copy(defaults)
                # if "backbone" in module_name and ('audio_backbone' not in module_name):
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER # 0.1
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed

                params.append({"params": [value], **hyperparams})

        # not use it
        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE # 0.01
            # True
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED #True
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    # concat multiple iterators to one 
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        # use ADAMW
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        
        # not use it
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)

        return optimizer
    
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger("NTAVS")
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )
        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)

            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            
            if dataset_name[:5] == 'avss_':
                results_i = inference_on_dataset_ss(model, data_loader, evaluator, cfg.OUTPUT_DIR,
                                                    dataset_name, if_train=True)
            else:
                results_i = inference_on_dataset(model, data_loader, evaluator, cfg.OUTPUT_DIR,
                                                 dataset_name, if_train=True)
            results[dataset_name] = results_i

            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Testing results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

def setup(args):
    """
    Create configs and perform basic setups.
    """
    # create default setting
    cfg = get_cfg()

    # for poly lr schedule
    add_deeplab_config(cfg)
    add_audio_config(cfg)
    add_fuse_config(cfg)
    add_maskformer2_config(cfg)
    
    # load from config
    # 
    cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)

    # 
    cfg.freeze()

    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    # distributed train , 0
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="NTAVS")

    return cfg

# TXlqhV2cPIQ_5
def main(args):
    cfg = setup(args)

    # False, not used
    if args.eval_only:
        model = Trainer.build_model(cfg)
        # load weight
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg) 
    
    # Use Pretrained Model on S4 for avms3.
    # if cfg.INPUT.DATASET_MAPPER_NAME == "avsms3_semantic":
    #     model_pretrained_path = 'model_pretrained.pth'
    #     model_state_dict = torch.load(model_pretrained_path, map_location=torch.device("cpu"))
    #     trainer.model.load_state_dict(model_state_dict['model'])
    #     print("Load Pretrained Model on S4 for avms3.")
    #     del model_state_dict

    # hook: input a function to a class
    trainer.register_hooks(
        [BestCheckpointer(eval_period=cfg.TEST.EVAL_PERIOD, 
                          val_metric="sem_seg/mIoU", checkpointer=trainer.checkpointer, mode='max')])
    trainer.resume_or_load(resume=args.resume)
    # trainer.resume_or_load(resume=False)
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    # launch: used for multi-gpu or distributed training
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
