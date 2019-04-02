from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
from torch.autograd import gradcheck
from student_code import custom_conv2d, CustomConv2d

# set up param here
# you can try some random numbers here as long as they are valid convolutions
num_imgs = 2
in_channels = 3
out_channels = 64
kernel_size = 7
stride = 2
padding = 3
input_height = 12
input_width = 12
atol = 1e-05

# let us see what we have
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: " + str(device))

# set up the variables
# double precision is needed for numerical gradients!
# this will also turn off the cudnn backend (remove randomness)
input_feats = torch.randn(num_imgs,
                          in_channels,
                          input_height,
                          input_width,
                          requires_grad=True,
                          device=device).double()
weight = torch.randn(out_channels,
                     in_channels,
                     kernel_size,
                     kernel_size,
                     requires_grad=True,
                     device=device).double()
bias = torch.randn(out_channels,
                   requires_grad=True,
                   device=device).double()

# forward
print('Check Fprop ...')
ref_conv2d = torch.nn.functional.conv2d
ref_output = ref_conv2d(input_feats, weight, bias, stride, padding)
custom_output = custom_conv2d(input_feats, weight, bias, stride, padding)
err = (custom_output - ref_output).abs().max()
if err < atol:
  print("Fprop testing passed")
else:
  print("Fprop testing failed")

# backward
print('Check Bprop ...')
inputs = (input_feats, weight, bias, stride, padding)
test = gradcheck(custom_conv2d, inputs, eps=1e-4, atol=atol)
if test:
  print("Bprop testing passed")
else:
  print("Bprop testing failed")

print('Check nn.module wrapper ...')

# instantiate custom conv2d module and test the wrapper
conv2d_module = CustomConv2d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding).to(device)
output = conv2d_module(input_feats.float())
print('All passed! End of testing.')
