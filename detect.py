import os
import torch
from gorilla.config import Config
from utils import *
import argparse
import pickle
from tqdm import tqdm


GUIDE_W = 0.5
DEVICE=torch.device('cuda')


# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Test a model")
    parser.add_argument("--config", help="test config file path")
    parser.add_argument("--checkpoint", help="path to checkpoint model")
    parser.add_argument("--test_data", help="path to test_data")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu
    model = build_model(cfg).to(DEVICE)
    
    if args.checkpoint != None:
        print("Loading checkpoint....")
        _, exten = os.path.splitext(args.checkpoint)
        if exten == '.t7':
            model.load_state_dict(torch.load(args.checkpoint))
        elif exten == '.pth':
            check = torch.load(args.checkpoint)
            model.load_state_dict(check['model_state_dict'])
    else:
        raise ValueError("Must specify a checkpoint path!")
    
    with open(args.test_data, 'rb') as f:
        shape_data = pickle.load(f)
    
    print("Testing")
    model.eval()
    with torch.no_grad():
        for shape in tqdm(shape_data):
            xyz = torch.from_numpy(shape['full_shape']['coordinate']).unsqueeze(0).float().cuda()
            shape['result'] = {text: [*(model.detect_and_sample(xyz, text, 2000, guide_w=GUIDE_W))] for text in shape['affordance']}
    
    with open(f'{cfg.log_dir}/result.pkl', 'wb') as f:
        pickle.dump(shape_data, f)