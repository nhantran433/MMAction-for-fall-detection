from multiprocessing import freeze_support
from mmaction.apis import inference_recognizer, init_recognizer
from mmengine.runner import set_random_seed
from mmengine import Config
import os.path as osp
from mmengine.runner import Runner
import torch, mmengine

config = 'configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'
cfg = Config.fromfile(config)
# Setup a checkpoint file to load
checkpoint = 'checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'

# Modify dataset type and path
cfg.data_root = 'datasets/train/'
cfg.data_root_val = 'datasets/val/'
cfg.ann_file_train = 'datasets/train_label.txt'
cfg.ann_file_val = 'datasets/val_label.txt'


cfg.test_dataloader.dataset.ann_file = 'datasets/val_label.txt'
cfg.test_dataloader.dataset.data_prefix.video = 'datasets/val/'

cfg.train_dataloader.dataset.ann_file = 'datasets/train_label.txt'
cfg.train_dataloader.dataset.data_prefix.video = 'datasets/train/'

cfg.val_dataloader.dataset.ann_file = 'datasets/val_label.txt'
cfg.val_dataloader.dataset.data_prefix.video  = 'datasets/val/'


# Modify num classes of the model in cls_head
cfg.model.cls_head.num_classes = 2
# We can use the pre-trained TSN model
cfg.load_from = './checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './work_dir'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.train_dataloader.batch_size = cfg.train_dataloader.batch_size // 16
cfg.val_dataloader.batch_size = cfg.val_dataloader.batch_size // 16
cfg.optim_wrapper.optimizer.lr = cfg.optim_wrapper.optimizer.lr / 8 / 16
cfg.train_cfg.max_epochs = 10

cfg.train_dataloader.num_workers = 1
cfg.val_dataloader.num_workers = 1
cfg.test_dataloader.num_workers = 1

# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')


mmengine.mkdir_or_exist(osp.abspath(cfg.work_dir))

runner = Runner.from_cfg(cfg)

runner.train()