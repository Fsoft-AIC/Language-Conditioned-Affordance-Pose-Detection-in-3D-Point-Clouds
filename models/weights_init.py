import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.state_dict().get('bias') != None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.state_dict().get('bias') != None:
            torch.nn.init.constant_(m.bias.data, 0.0)