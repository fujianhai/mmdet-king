from mmdet.apis import init_detector, inference_detector
import mmcv
import os

# Specify the path to model config and checkpoint file
config_file = '/home/willer/heywhale/UniverseNet/configs/universenet/underwater_ms.py'
checkpoint_file = '/home/willer/heywhale/UniverseNet/work_dirs/underwater/epoch_24.pth'
save_dir = './output/'
os.makedirs(save_dir, exist_ok=True)

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

img_dir = '/home/willer/heywhale/data/test-A-image/'
for name in os.listdir(img_dir):
    # test a single image and show the results
    img = os.path.join(img_dir,name)  # or img = mmcv.imread(img), which will only load it once
    print(img)
    result = inference_detector(model, img)
    # or save the visualization results to image files
    model.show_result(img, result, out_file=os.path.join(save_dir,name))
