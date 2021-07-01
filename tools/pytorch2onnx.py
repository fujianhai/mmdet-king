import argparse
import os.path as osp
import warnings

import numpy as np
import onnx
from onnxsim import simplify
import torch
from mmcv import DictAction
import mmcv
from functools import partial
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint

config_path = '/home/willer/tianchi/mmd_solution/UniverseNet/configs/universenet/universenet50_gfl_fp16_4x4_mstrain_480_960_1x_coco.py'
checkpoint_path = '/home/willer/tianchi/mmd_solution/UniverseNet/weights/universenet50_gfl_fp16_4x4_mstrain_480_960_2x_coco_20200729_epoch_24-c9308e66.pth'
output_file = './universenet50_gfl.onnx'

cfg = mmcv.Config.fromfile(config_path)
model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
output_names = ['boxes','labels']
dummy_input = torch.autograd.Variable(torch.randn(1, 3, 1333, 800))

(C, H, W) = (3, 1333, 800)
one_meta = {
    'img_shape': (H, W, C),
    'ori_shape': (H, W, C),
    'pad_shape': (H, W, C),
    'filename': '<demo>.png',
    'scale_factor': 1.0,
    'flip': False
}
model.forward = partial(model.forward, img_metas=[[one_meta]], return_loss=False)
torch.onnx.export(model,[dummy_input],output_file,input_names=['input'],output_names=output_names,export_params=True,keep_initializers_as_inputs=True,verbose=True,opset_version=12)
onnx_model = onnx.load(output_file)
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, output_file)
print('finished exporting onnx ')
