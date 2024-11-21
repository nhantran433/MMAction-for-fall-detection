import cv2, time, uuid
from operator import itemgetter
from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'work_dir/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'
checkpoint_file = "work_dir/best_acc_top1_epoch_2.pth"
model = init_recognizer(config_file, checkpoint_file, device='cpu')

frame_size = (640, 480)
fps = 20
video = "falling-video-input.mp4"
number_of_frames_per_check = 100 
label = 'label.txt'
labels = []
with open(label) as f:
    labels = f.readlines()

result = inference_recognizer(model, "input/fa6d11cd-5f8a-11ef-a95e-00155d96f873.mp4")
print(labels[int(result.pred_labels.get("item"))])
