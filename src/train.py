import torch
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
import multiprocessing as mp

def main():
    register_coco_instances(
        "val2017_orig_train", {}, 
        "../datasets/instances_val2017.json",
        "../datasets/val2017"
    )

    register_coco_instances(
        "val2017_stylized_train", {},
        "../datasets/instances_val2017_stylized.json",
        "../datasets/coco_stylized"
    )

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.DEVICE = "mps" if torch.mps.is_available() else "cpu"

    cfg.DATASETS.TRAIN = ("val2017_orig_train", "val2017_stylized_train")

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR       = 0.0025
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000 
    cfg.SOLVER.MAX_ITER      = 20000 

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80

    cfg.OUTPUT_DIR = "./output/val2017_mixed"
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    
if __name__ == "__main__":
    mp.freeze_support()
    main()