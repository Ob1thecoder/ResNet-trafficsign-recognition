# run_eval.py
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader

from testloader import GTSRB_Test_Loader
from evaluation import evaluate
from model import build_resnet18

if __name__ == '__main__':
    torch.manual_seed(118)

    # --- Paths ---
    TEST_IMG_DIR = 'GTSRB/Final_Test/Images'       
    TEST_GT_CSV  = 'evaluation/GTSRB_Test_GT.csv'
    CKPT_PATH    = 'checkpoints/resnet18_best.pt'

    # --- Data ---
    testset = GTSRB_Test_Loader(
        TEST_PATH=TEST_IMG_DIR,
        TEST_GT_PATH=TEST_GT_CSV,
        img_size=64,
        use_roi=True,          # ROI crop often helps
        imagenet_norm=True     # match pretrained normalization
    )
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # --- Model ---
    model = build_resnet18(num_classes=43, pretrained=False)
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    # support both plain state_dict and wrapped dict
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=False)

    # --- Eval ---
    acc = evaluate(model, testloader, device=None, expects_logits=True)
    print('testing finished, accuracy: {:.3f}'.format(acc))
