#!/usr/bin/env python3
#python scripts_zedx/run_lane_detection.py configs/tusimple.py --test_model weights/tusimple_18.pth

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import scipy.special
import torchvision.transforms as transforms
import argparse
import pyzed.sl as sl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
from data.constant import culane_row_anchor, tusimple_row_anchor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config file path')
    parser.add_argument('--test_model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--num_frames', type=int, default=150, help='Number of frames to process')
    parser.add_argument('--output_dir', type=str, default='./scripts_zedx/zedx_outputs', help='Output folder')
    parser.add_argument('--save_video', action='store_true', help='Save output video')
    args = parser.parse_args()
    args.save_video = True  # Force enable video saving

    args_cfg, cfg = merge_config()
    dist_print('start testing...')

    # Set dataset-specific settings
    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
        row_anchor = culane_row_anchor
        img_w, img_h = 1640, 590
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
        row_anchor = tusimple_row_anchor
        img_w, img_h = 1280, 720
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset} not supported")

    col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    # Load model
    net = parsingNet(pretrained=False, backbone=cfg.backbone,
                     cls_dim=(cfg.griding_num + 1, cls_num_per_lane, 4),
                     use_aux=False).cuda()
    state_dict = torch.load(args.test_model, map_location='cpu')['model']
    compatible_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize ZED X
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1200
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NONE

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open Error: {zed.get_last_error()}")
        exit(1)

    runtime_params = sl.RuntimeParameters()
    left_image = sl.Mat()

    # Video writer setup
    video_writer = None
    if args.save_video:
        video_path = os.path.join(args.output_dir, "lane_output.mp4")
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))

    print(f"Starting lane detection on ZED X frames...")

    for frame_id in range(args.num_frames):
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            frame = left_image.get_data()

            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Resize to 1280x720 for consistent input/output resolution
            frame_resized = cv2.resize(frame, (1280, 720))

            # Convert to PIL RGB for model transforms
            img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # Prepare for model input
            img_tensor = img_transforms(img_pil).unsqueeze(0).cuda()

            with torch.no_grad():
                out = net(img_tensor)

            out_j = out[0].cpu().numpy()
            out_j = out_j[:, ::-1, :]
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(cfg.griding_num) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == cfg.griding_num] = 0
            out_j = loc

            # Draw lanes on resized image (1280x720)
            vis = frame_resized.copy()
            for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            p_x = int(out_j[k, i] * col_sample_w * img_w / 800) - 1
                            p_y = int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1
                            if 0 <= p_x < 1280 and 0 <= p_y < 720:
                                cv2.circle(vis, (p_x, p_y), 5, (0, 255, 0), -1)

            if video_writer:
                video_writer.write(vis)

            print(f"Processed frame {frame_id+1}")

        else:
            print(f"Warning: Failed to grab frame {frame_id+1}")

    if video_writer:
        video_writer.release()
    zed.close()
    print(f"Done! Video saved to: {video_path}")

if __name__ == "__main__":
    main()
