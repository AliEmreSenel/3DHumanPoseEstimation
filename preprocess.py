#!/usr/bin/env python3
"""
Standalone CLI for pose & depth estimation on local subfolders with tqdm progress bars.
Usage:
    python process_all.py <input_base> <output_base>
"""
import os
import sys
import json
import logging
from pathlib import Path

import torch
import cv2
from PIL import Image
from ultralytics import YOLO
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
from tqdm import tqdm

def gather_subfolders(base_dir):
    return sorted([p for p in Path(base_dir).iterdir() if p.is_dir()])


def load_models(pose_model_path):
    print("Loading models...")
    # suppress ultralytics logs
    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    pose_model = YOLO(pose_model_path, verbose=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
    depth_model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)
    depth_model.eval()
    print(f"Using device: {device}")
    return pose_model, depth_processor, depth_model, device


def process_subfolder(input_dir, output_dir, pose_model, depth_processor, depth_model, device):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skeleton = [
        (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),
        (5,7),(7,9),(6,8),(8,10),(5,6),(5,11),
        (6,12),(11,12),(11,13),(13,15),(12,14),(14,16),
    ]

    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    for file in tqdm(images, desc=f"Processing {input_dir.name}", unit="img"):
        base = Path(file).stem
        out_img = output_dir / f"{base}_depth.png"
        out_json = output_dir / f"{base}.json"

        if out_img.exists() and out_json.exists():
            continue

        img_path = input_dir / file
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Pose estimation
        results = pose_model(img, device=str(device))
        keypoints = []
        for res in results:
            for person in res.keypoints.xy:
                pts = [{'x':int(x),'y':int(y)} for x,y in person]
                keypoints.append(pts)

        # Depth estimation
        pil_img = Image.open(img_path).convert('RGB')
        inputs = depth_processor(images=pil_img, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = depth_model(**inputs)
        post = depth_processor.post_process_depth_estimation(
            outputs, target_sizes=[(pil_img.height, pil_img.width)]
        )
        depth = post[0]['predicted_depth']
        dmin, dmax = depth.min(), depth.max()
        norm = ((depth - dmin)/(dmax-dmin)*255).detach().cpu().numpy().astype('uint8')
        depth_vis = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

        metadata = {
            'image_size':[img.shape[1],img.shape[0]],
            'depth_size':[norm.shape[1],norm.shape[0]],
            'skeleton':skeleton,
            'keypoints':keypoints,
            'depth_min':float(dmin),
            'depth_max':float(dmax),
        }

        cv2.imwrite(str(out_img), depth_vis)
        with open(out_json, 'w') as f:
            json.dump(metadata, f, indent=2)

    # finish marker
    marker = output_dir / 'finished.txt'
    marker.write_text('complete')


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_base> <output_base>")
        sys.exit(1)

    input_base = Path(sys.argv[1])
    output_base = Path(sys.argv[2])
    subfolders = gather_subfolders(input_base)

    # adjust this path or accept as argument
    pose_model_path = "yolo11x-pose"

    pose_model, depth_processor, depth_model, device = load_models(pose_model_path)

    for subfolder in tqdm(subfolders, desc="Overall folders", unit="folder"):
        rel = subfolder.relative_to(input_base)
        out_dir = output_base / rel
        process_subfolder(subfolder, out_dir, pose_model, depth_processor, depth_model, device)

    print("All done.")

if __name__=='__main__':
    main()
