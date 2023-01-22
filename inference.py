import os
import time
#os.system('pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html')
#os.system("git clone https://github.com/microsoft/unilm.git")

import sys
sys.path.append("unilm")

# import cv2

from unilm.dit.object_detection.ditod import add_vit_config

import torch

# from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
# from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

# import gradio as gr

from PIL import Image
import numpy as np
import io
import base64

def setup_model(weights_filepath, cfg_filepath):
    # Step 1: instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(cfg_filepath)

    # Step 2: add model weights URL to config
    cfg.MODEL.WEIGHTS = weights_filepath  
    # "https://layoutlm.blob.core.windows.net/dit/dit-fts/publaynet_dit-b_cascade.pth"

    # Step 3: set device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 4: define model
    predictor = DefaultPredictor(cfg)
    return (cfg, predictor)


def extract_bound_boxes(img, model, cfg):
    now = time.time()
    DiT_classes = ["text","title","list","table","figure"]
    int_to_classes = dict(enumerate(DiT_classes))
    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    if cfg.DATASETS.TEST[0]=='icdar2019_test':
        md.set(thing_classes=["table"])
    else:
        md.set(thing_classes=["text","title","list","table","figure"])
    
    output = model(img)["instances"]
    output_instance = output.to("cpu")
    fields = output_instance.get_fields()
    bound_boxes_classes = list(zip(
                            fields['pred_boxes'].tensor.tolist(),
                            [int_to_classes.get(c) for c in fields['pred_classes'].tolist()]
                               ))
    print(time.time()-now)
    return (output.image_size, bound_boxes_classes) 

def boundbox_predict(img_b64_str, model, cfg):
    imgdata = base64.b64decode(img_b64_str)
    img = Image.open(io.BytesIO(imgdata))
    image = np.asarray(img)
    img_size_tpl, bound_boxes = extract_bound_boxes(image, model, cfg)
    # im = Image.fromarray(result_image)
    # im.save("output.jpg")
    return (img_size_tpl, bound_boxes)

'''
examples =[['publaynet_example.jpeg']]

# Open the image form working directory
base64_file = "./img_b64"
output_filename = "./img_output.jpg"
import base64
with open(base64_file) as inpf:
    img_b64_str = inpf.read()
    #imgdata = base64.b64decode(img_b64_str)

cfg_filepath = "cascade_dit_base.yml"
weights_filepath = "./publaynet_dit-b_cascade.pth"
cfg, model = setup_model(weights_filepath, cfg_filepath)
boundbox_predict(img_b64_str, model, cfg)
'''
