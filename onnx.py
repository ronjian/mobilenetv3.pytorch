#%%
import torch
from mobilenetv3 import mobilenetv3_large, mobilenetv3_small


#%%
net_large = mobilenetv3_large()
# net_small = mobilenetv3_small()

net_large.load_state_dict(torch.load('pretrained/mobilenetv3-large-b4e262ea.pth'))
# net_small.load_state_dict(torch.load('pretrained/mobilenetv3-small-547c1152.pth'))

# %%
net_large.eval()

# %%
x = torch.rand(1, 3, 384, 384)

# %%
torch_res = torch.onnx._export(net_large
                            , x
                            , 'pretrained/mobilenetv3-large-b4e262ea.onnx'
                            , export_params=True
                            , opset_version = 11)

# %%
from torchvision import models

# %%
m2 = models.mobilenet_v2(pretrained=True)

# %%
m2.eval()

# %%
torch_res = torch.onnx._export(m2
                            , x
                            , 'pretrained/mobilenetv2.onnx'
                            , export_params=True
                            , opset_version = 11)

# %%
