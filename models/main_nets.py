import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .components import TextEncoder, PointNetPlusPlus, PoseNet


text_encoder = TextEncoder(device=torch.device('cuda'))


def linear_diffusion_schedule(betas, T):
    """_summary_
    Linear cheduling for sampling in training.
    """
    beta_t = (betas[1] - betas[0]) * torch.arange(0, T + 1, dtype=torch.float32) / T + betas[0]
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
    
    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DetectionDiffusion(nn.Module):
    def __init__(self, betas, n_T, device, background_text, drop_prob=0.1):
        """_summary_

        Args:
            drop_prob: probability to drop the conditions
        """
        super(DetectionDiffusion, self).__init__()
        self.posenet = PoseNet()
        self.pointnetplusplus = PointNetPlusPlus()
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Register_buffer allows accessing dictionary, e.g. can access self.sqrtab later
        for k, v in linear_diffusion_schedule(betas, n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.background_text = background_text
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, xyz, text, affordance_label, g):
        """_summary_
        This method is used in training, so samples _ts and noise randomly.
        """
        B = xyz.shape[0]    # xyz's size [B, 3, 2048]
        point_features, c = self.pointnetplusplus(xyz) # point_features' size [B, 512, 2048], c'size [B, 1024]
        with torch.no_grad():
            foreground_text_features = text_encoder(text)   # size [B, 512]
            background_text_features = text_encoder([self.background_text] * B)
        text_features = torch.cat((background_text_features.unsqueeze(1), \
            foreground_text_features.unsqueeze(1)), dim=1)  # size [B, 2, 512]
        
        affordance_prediction = self.logit_scale * torch.einsum('bij,bjk->bik', text_features, point_features) \
            / (torch.einsum('bij,bjk->bik', torch.norm(text_features, dim=2, keepdim=True), \
                torch.norm(point_features, dim=1, keepdim=True)))   # size [B, 2, 2048]
        
        affordance_prediction = F.log_softmax(affordance_prediction, dim=1)
        affordance_loss = F.nll_loss(affordance_prediction, affordance_label)
        
        _ts = torch.randint(1, self.n_T + 1, (B,)).to(self.device)
        noise = torch.randn_like(g)  # eps ~ N(0, 1), g size [B, 7]
        g_t = (
            self.sqrtab[_ts - 1, None] * g
            + self.sqrtmab[_ts - 1, None] * noise
        )  # This is the g_t, which is sqrt(alphabar) g_0 + sqrt(1-alphabar) * eps

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros(B, 1) + 1 - self.drop_prob).to(self.device)
        
        # Loss for poseing is MSE between added noise, and our predicted noise
        pose_loss = self.loss_mse(noise, self.posenet(g_t, c, foreground_text_features, context_mask, _ts / self.n_T))
        return affordance_loss, pose_loss
    
    def detect_and_sample(self, xyz, text, n_sample, guide_w):
        """_summary_
        Detect affordance for one point cloud and sample [n_sample] poses that support the 'text' affordance task,
        following the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'.
        """
        g_i = torch.randn(n_sample, (7)).to(self.device) # start by sampling from Gaussian noise
        point_features, c = self.pointnetplusplus(xyz) # point_features size [1, 512, 2048], c size [1, 1024]
        foreground_text_features = text_encoder(text)   # size [1, 512]
        background_text_features = text_encoder([self.background_text] * 1)
        text_features = torch.cat((background_text_features.unsqueeze(1), \
            foreground_text_features.unsqueeze(1)), dim=1)  # size [B, 2, 512]
        
        affordance_prediction = self.logit_scale * torch.einsum('bij,bjk->bik', text_features, point_features) \
            / (torch.einsum('bij,bjk->bik', torch.norm(text_features, dim=2, keepdim=True), \
                torch.norm(point_features, dim=1, keepdim=True)))   # size [1, 2, 2048]
        
        affordance_prediction = F.log_softmax(affordance_prediction, dim=1) # .cpu().numpy()
        c_i = c.repeat(n_sample, 1)
        t_i = foreground_text_features.repeat(n_sample, 1)
        context_mask = torch.ones((n_sample, 1)).float().to(self.device)
        
        # Double the batch
        c_i = c_i.repeat(2, 1)
        t_i = t_i.repeat(2, 1)
        context_mask = context_mask.repeat(2, 1)
        context_mask[n_sample:] = 0.    # make second half of the back context-free
        
        for i in range(self.n_T, 0, -1):
            _t_is = torch.tensor([i / self.n_T]).repeat(n_sample).repeat(2).to(self.device)
            g_i = g_i.repeat(2, 1)
            
            z = torch.randn(n_sample, (7)) if i > 1 else torch.zeros((n_sample, 7))
            z = z.to(self.device)
            eps = self.posenet(g_i, c_i, t_i, context_mask, _t_is)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            
            g_i = g_i[:n_sample]
            g_i = self.oneover_sqrta[i] * (g_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
        return np.argmax(affordance_prediction.cpu().numpy(), axis=1), g_i.cpu().numpy()