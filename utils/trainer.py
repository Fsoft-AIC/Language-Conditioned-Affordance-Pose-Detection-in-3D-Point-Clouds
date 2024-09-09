import torch
from tqdm import tqdm
from os.path import join as opj
from utils import *


DEVICE = torch.device('cuda')


class Trainer(object):
    def __init__(self, cfg, running):
        super().__init__()
        self.cfg = cfg
        self.logger = running['logger']
        self.model = running["model"]
        self.dataset_dict = running["dataset_dict"]
        self.loader_dict = running["loader_dict"]
        self.train_loader = self.loader_dict.get("train_loader", None)
        self.optimizer_dict = running["optim_dict"]
        self.optimizer = self.optimizer_dict.get("optimizer", None)
        self.scheduler = self.optimizer_dict.get("scheduler", None)
        self.epoch = 0
        self.bn_momentum = self.cfg.training_cfg.get('bn_momentum', None)
              
    def train(self):
        self.model.train()
        self.logger.cprint("Epoch(%d) begin training........" % self.epoch)
        pbar = tqdm(self.train_loader)
        for _, _, xyz, text, affordance_label, rotation, translation in pbar:
            self.optimizer.zero_grad()
            xyz = xyz.float()
            rotation = rotation.float()
            translation = translation.float()
            affordance_label = affordance_label.squeeze().long()

            g = torch.cat((rotation, translation), dim=1)
            xyz = xyz.to(DEVICE)
            affordance_label = affordance_label.to(DEVICE)
            g = g.to(DEVICE)
            
            affordance_loss, pose_loss = self.model(xyz, text, affordance_label, g)
            loss = affordance_loss + pose_loss
            loss.backward()
            
            affordance_l = affordance_loss.item()
            pose_l = pose_loss.item()
            pbar.set_description(f'Affordance loss: {affordance_l:.5f}, Pose loss: {pose_l:.5f}')
            self.optimizer.step()
            
        if self.scheduler != None:
            self.scheduler.step()   
        if self.bn_momentum != None:
            self.model.apply(lambda x: self.bn_momentum(x, self.epoch))
        
        outstr = f"\nEpoch {self.epoch}, Last Affordance loss: {affordance_l:.5f}, Last Pose loss: {pose_l:.5f}"
        self.logger.cprint(outstr)
        print('Saving checkpoint')
        torch.save(self.model.state_dict(), opj(self.cfg.log_dir, 'current_model.t7'))
        self.epoch += 1

    def val(self):
       raise NotImplementedError
       
    def test(self):
        raise NotImplementedError

    def run(self):
        EPOCH = self.cfg.training_cfg.epoch
        workflow = self.cfg.training_cfg.workflow
        
        while self.epoch < EPOCH:
            for key, running_epoch in workflow.items():
                epoch_runner = getattr(self, key)
                for _ in range(running_epoch):
                    epoch_runner()
