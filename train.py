import os
from os.path import join as opj
from gorilla.config import Config
from utils import *
import argparse
import torch


# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", help="train config file path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
        
    logger = IOStream(opj(cfg.log_dir, 'run.log'))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu
    num_gpu = len(cfg.training_cfg.gpu.split(','))      # number of GPUs to use
    logger.cprint('Use %d GPUs: %s' % (num_gpu, cfg.training_cfg.gpu))
    if cfg.get('seed') != None:     # set random seed
        set_random_seed(cfg.seed)
        logger.cprint('Set seed to %d' % cfg.seed)
    model = build_model(cfg).cuda()     # build the model from configuration

    print("Training from scratch!")

    dataset_dict = build_dataset(cfg)       # build the dataset
    loader_dict = build_loader(cfg, dataset_dict)       # build the loader
    optim_dict = build_optimizer(cfg, model)        # build the optimizer
    
    # construct the training process
    training = dict(
        model=model,
        dataset_dict=dataset_dict,
        loader_dict=loader_dict,
        optim_dict=optim_dict,
        logger=logger
    )

    task_trainer = Trainer(cfg, training)
    task_trainer.run()
