# run_eval.py
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
import os
from testloader import GTSRB_Test_Loader
from evaluation import evaluate
from model import build_resnet18

if __name__ == '__main__':
    torch.manual_seed(118)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- Paths ---
    TEST_IMG_DIR = os.path.join(script_dir, 'GTSRB/Final_Test/Images')  
    TEST_GT_CSV  = os.path.join(script_dir, 'evaluation/GTSRB_Test_GT.csv')
    CKPT_PATH    = os.path.join(script_dir, 'checkpoints/resnet18_best.pt')

    # --- Data ---
    testset = GTSRB_Test_Loader(
        TEST_PATH=TEST_IMG_DIR,
        TEST_GT_PATH=TEST_GT_CSV
    )
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # --- Model ---
    model = build_resnet18(num_classes=43, pretrained=False)
    
    # Load checkpoint
    try:
        ckpt = torch.load(CKPT_PATH, map_location='cpu')
        # Support both plain state_dict and wrapped dict
        state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        print(f"Successfully loaded checkpoint from {CKPT_PATH}")
    except FileNotFoundError:
        print(f"Warning: Checkpoint file {CKPT_PATH} not found. Using randomly initialized model.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Using randomly initialized model.")

    # Set model to evaluation mode
    model.eval()
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")

    # --- Eval ---
    with torch.no_grad():
        acc = evaluate(model, testloader)
    
    print('Testing finished, accuracy: {:.3f}'.format(acc))