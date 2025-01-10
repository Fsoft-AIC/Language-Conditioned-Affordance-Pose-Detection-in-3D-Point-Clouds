from utils.eval import affordance_eval, pose_eval
import argparse
import pickle
 
 
AFFORDANCE_LIST = ['grasp to pour', 'grasp to stab', 'stab', 'pourable', 'lift', 'wrap_grasp', 'listen', 'contain', 'displaY', 'grasp to cut', 'cut', 'wear', 'openable', 'grasp']
 

def parse_args():
    parser = argparse.ArgumentParser(description="Test a model")
    parser.add_argument("--result", help="result file")
    args = parser.parse_args()
    return args
 
 
if __name__ == "__main__":
    args = parse_args()
    with open(args.result, 'rb') as f:
        result = pickle.load(f)
    mIoU, Acc, mAcc = affordance_eval(AFFORDANCE_LIST, result)
    print(f'mIoU: {mIoU}, Acc: {Acc}, mAcc: {mAcc}')
   
    mESM, mCR = pose_eval(result)
    print(f'mESM: {mESM}, mCR: {mCR}')
 
 