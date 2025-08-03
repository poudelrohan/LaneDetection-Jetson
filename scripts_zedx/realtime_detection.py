#!/usr/bin/env python3

#python scripts_zedx/realtime_detection.py

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import scipy.special
import torchvision.transforms as transforms
import pyzed.sl as sl

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
from data.constant import tusimple_row_anchor

def main():
    # --- Load model config ---
    sys.argv = ["realtime_detection.py", "configs/tusimple.py", "--test_model", "weights/tusimple_18.pth"]
    args_cfg, cfg = merge_config()
    dist_print("Loading model...")

    cls_num_per_lane = 56
    row_anchor = tusimple_row_anchor
    img_w, img_h = 1280, 720  # Display/processing frame size

    col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    # --- Load the pretrained lane detection model ---
    net = parsingNet(pretrained=False, backbone=cfg.backbone,
                     cls_dim=(cfg.griding_num + 1, cls_num_per_lane, 4),
                     use_aux=False).cuda()
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    # --- Initialize ZED X camera ---
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

    print("Running real-time lane detection. Press 'q' to quit.")

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            frame = left_image.get_data()

            # Convert BGRA to BGR if necessary
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Resize for display and consistent detection
            frame = cv2.resize(frame, (img_w, img_h))

            # Convert BGR to RGB and apply transforms
            img = Image.fromarray(frame[..., ::-1])
            img_tensor = img_transforms(img).unsqueeze(0).cuda()

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

            vis = frame.copy()
            for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            p = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                                 int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1)
                            cv2.circle(vis, p, 5, (0, 255, 0), -1)

            # Display
            cv2.imshow("Lane Detection", vis)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break

        else:
            print("Failed to grab frame.")

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
