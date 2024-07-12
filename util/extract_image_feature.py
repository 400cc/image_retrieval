import argparse
import os
import copy
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, load_image_from_memory, load_image_from_url, predict

import supervision as sv

# segment anything
base_path = os.path.join(os.path.dirname(__file__), "segment-anything")


# 경로를 sys.path에 추가
if base_path not in sys.path:
    sys.path.append(base_path)

from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

# embedding / vector DB
import clip
import psycopg2
from psycopg2.extras import execute_batch


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model  

ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sam_checkpoint = 'sam_vit_h_4b8939.pth'
sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

model, preprocess = clip.load('ViT-L/14', device=DEVICE)

# grounding DINO로 box detection
def detect(image, text_prompt, model, image_source, box_threshold = 0.3, text_threshold = 0.25):
  boxes, logits, phrases = predict(
      model=model, 
      image=image, 
      caption=text_prompt,
      box_threshold=box_threshold,
      text_threshold=text_threshold,
      device=DEVICE
  )
  
  annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
  annotated_frame = annotated_frame[...,::-1] # BGR to RGB 

  return boxes, annotated_frame


# 얻은 박스를 프롬프트로 활용하여 SAM 적용
def segment(image, sam_model, boxes):
  sam_model.set_image(image)
  H, W, _ = image.shape
  boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

  transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(DEVICE), image.shape[:2])
  masks, _, _ = sam_model.predict_torch(
      point_coords = None,
      point_labels = None,
      boxes = transformed_boxes,
      multimask_output = False,
      )
  return masks.cpu()


# 상품 크기에 맞추어 이미지 crop
def apply_mask_to_image(image, mask):
    # 마스크된 부분만을 원본 이미지에 적용합니다.
    masked_image = image.copy()
    masked_image[mask == 0] = [255, 255, 255]

    masked_indices = np.where(mask != 0)

    # 마스크된 부분의 최소 및 최대 y, x 좌표를 찾습니다.
    min_y, min_x = np.min(masked_indices, axis=1)
    max_y, max_x = np.max(masked_indices, axis=1)

    # 마스크된 부분만을 포함하는 새로운 이미지 생성
    masked_region_only = masked_image[min_y:max_y+1, min_x:max_x+1, :]

    return masked_region_only

# grounded SAM pipeline
def SAM(prompt, image, image_source):
    detected_boxes, annotated_frame = detect(image, prompt, image_source=image_source, model=groundingdino_model)
    if detected_boxes.size(0) == 0:
        return image_source, None
    segmented_frame_masks = segment(image_source, sam_predictor, boxes=detected_boxes)
    masked_region_only = apply_mask_to_image(image_source, segmented_frame_masks[0][0])
    return masked_region_only, annotated_frame

def process_image_and_feature_by_local(image_path, category):
    image_source, image = load_image(image_path)
    sam_image, _ = SAM(category, image, image_source)
    sam_image_result = Image.fromarray(sam_image)
    with torch.no_grad():
        preprocessed_image = preprocess(sam_image_result).unsqueeze(0).to(DEVICE)
        image_features = model.encode_image(preprocessed_image)
        image_feature = image_features[0]
        
    return sam_image_result, image_feature

def process_image_and_feature(image_path, category):
    image_source, image = load_image_from_url(image_path)
    sam_image, _ = SAM(category, image, image_source)
    sam_image_result = Image.fromarray(sam_image)
    with torch.no_grad():
        preprocessed_image = preprocess(sam_image_result).unsqueeze(0).to(DEVICE)
        image_features = model.encode_image(preprocessed_image)
        image_feature = image_features[0].cpu().numpy().tolist()
        
    return image_feature

def process_image_and_feature_by_app(image_path, category):
    image_source, image = load_image_from_memory(image_path)
    sam_image, _ = SAM(category, image, image_source)
    sam_image_result = Image.fromarray(sam_image)
    with torch.no_grad():
        preprocessed_image = preprocess(sam_image_result).unsqueeze(0).to(DEVICE)
        image_features = model.encode_image(preprocessed_image)
        image_feature = image_features[0]
        
    return sam_image_result, image_feature
