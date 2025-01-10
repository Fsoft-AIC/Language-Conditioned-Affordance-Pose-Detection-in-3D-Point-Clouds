import os
import torch
from os.path import join as opj
from utils import PN2_BNMomentum

exp_name = 'detectiondiffusion'
seed = 1
log_dir = opj("./log/", exp_name)
try:
    os.makedirs(log_dir)
except:
    print('Logging Dir is already existed!')

# scheduler = dict(
#     type='lr_lambda',
#     lr_lambda=PN2_Scheduler(init_lr=0.001, step=20,
#                             decay_rate=0.5, min_lr=1e-5)
# )

scheduler = None

optimizer = dict(
    type='adam',
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-5,
)

model = dict(
    type='detectiondiffusion',
    device=torch.device('cuda'),
    background_text='none',
    betas=[1e-4, 0.02],
    n_T=1000,
    drop_prob=0.1,
    weights_init='default_init',
)

training_cfg = dict(
    model=model,
    batch_size=32,
    epoch=200,
    gpu='0',
    workflow=dict(
        train=1,
    ),
    bn_momentum=PN2_BNMomentum(origin_m=0.1, m_decay=0.5, step=20),
)

data = dict(
    data_path="../full_shape_release.pkl",
)