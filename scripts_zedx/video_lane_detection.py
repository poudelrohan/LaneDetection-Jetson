# python scripts_zedx/video_lane_detection.py

import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import scipy.special
import torchvision.transforms as transforms

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
from data.constant import tusimple_row_anchor

def main():
    # --- Config and paths ---
    sys.argv = ["video_lane_detection.py", "configs/tusimple.py", "--test_model", "weights/tusimple_18.pth"]
    args_cfg, cfg = merge_config()
    dist_print("Loading model...")

    input_video_path = "scripts_zedx/input/videoplayback.mp4"
    output_video_path = "scripts_zedx/zedx_outputs/lane_output_from_video.mp4"
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # --- Lane detection settings ---
    cls_num_per_lane = 56
    row_anchor = tusimple_row_anchor
    img_w, img_h = 1280, 720

    col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    # --- Load model ---
    net = parsingNet(pretrained=False, backbone=cfg.backbone,
                     cls_dim=(cfg.griding_num + 1, cls_num_per_lane, 4),
                     use_aux=False).cuda()
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    net.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}, strict=False)
    net.eval()

    # --- Transforms ---
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    # --- Open video ---
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {input_video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (img_w, img_h))

    print("Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for display and model
        frame = cv2.resize(frame, (img_w, img_h))

        # Preprocess
        img = Image.fromarray(frame[..., ::-1])  # BGR to RGB
        img_tensor = img_transforms(img).unsqueeze(0).cuda()

        # Forward pass
        with torch.no_grad():
            out = net(img_tensor)

        out_j = out[0].cpu().numpy()
        out_j = out_j[:, ::-1, :]  # flip width
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == cfg.griding_num] = 0
        out_j = loc

        vis = frame.copy()
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        p = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                             int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1)
                        cv2.circle(vis, p, 5, (0, 255, 0), -1)

        video_writer.write(vis)

    cap.release()
    video_writer.release()
    print(f"Done! Output saved to: {output_video_path}")

if __name__ == "__main__":
    main()
