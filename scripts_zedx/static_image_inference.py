# python -m scripts_zedx.static_image_inference configs/tusimple.py --test_model weights/tusimple_18.pth
import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import scipy.special
import numpy as np
import torchvision.transforms as transforms
from data.constant import culane_row_anchor, tusimple_row_anchor
from PIL import Image

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args, cfg = merge_config()

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
        row_anchor = culane_row_anchor
        img_w, img_h = 1640, 590
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
        row_anchor = tusimple_row_anchor
        img_w, img_h = 1280, 720
    else:
        raise NotImplementedError

    net = parsingNet(pretrained=False, backbone=cfg.backbone,
                     cls_dim=(cfg.griding_num + 1, cls_num_per_lane, 4),
                     use_aux=False).cuda()

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {k[7:] if 'module.' in k else k: v for k, v in state_dict.items()}
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    input_folder = './scripts_zedx/input'
    output_folder = './scripts_zedx/output'

    os.makedirs(output_folder, exist_ok=True)

    img_names = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]
    img_names.sort()

    col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    for name in img_names:
        img_path = os.path.join(input_folder, name)
        img = Image.open(img_path).convert('RGB')
        img_tensor = img_transforms(img)
        img_tensor = img_tensor.unsqueeze(0).cuda()

        with torch.no_grad():
            out = net(img_tensor)

        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == cfg.griding_num] = 0
        out_j = loc

        vis = cv2.imread(img_path)
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                               int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1)
                        cv2.circle(vis, ppp, 5, (0, 255, 0), -1)

        save_path = os.path.join(output_folder, name)
        cv2.imwrite(save_path, vis)

    print(f"Done. Processed {len(img_names)} images. Saved to '{output_folder}'")
