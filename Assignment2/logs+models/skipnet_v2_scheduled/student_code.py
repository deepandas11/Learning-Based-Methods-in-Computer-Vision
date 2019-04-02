from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.functional import fold, unfold
from torchvision.utils import make_grid
import math
from utils import resize_image
from skip_net_modified import SkipNetModified


#################################################################################
# You will need to fill in the missing code in this file
#################################################################################


#################################################################################
# Part I: Understanding Convolutions
#################################################################################
class CustomConv2DFunction(Function):

  @staticmethod
  def forward(ctx, input_feats, weight, bias, stride=1, padding=0):
    """
    Forward propagation of convolution operation.
    We only consider square filters with equal stride/padding in width and height!
    Args:
      input_feats: input feature map of size N * C_i * H * W
      weight: filter weight of size C_o * C_i * K * K
      bias: (optional) filter bias of size C_o
      stride: (int, optional) stride for the convolution. Default: 1
      padding: (int, optional) Zero-padding added to both sides of the input. Default: 0
    Outputs:
      output: responses of the convolution  w*x+b
    """
    # sanity check
    assert weight.size(2) == weight.size(3)
    assert input_feats.size(1) == weight.size(1)
    assert isinstance(stride, int) and (stride > 0)
    assert isinstance(padding, int) and (padding >= 0)

    # save the conv params
    kernel_size = weight.size(2)
    ctx.stride = stride
    ctx.padding = padding
    ctx.input_height = input_feats.size(2)
    ctx.input_width = input_feats.size(3)

    # make sure this is a valid convolution
    assert kernel_size <= (input_feats.size(2) + 2 * padding)
    assert kernel_size <= (input_feats.size(3) + 2 * padding)

    #################################################################################
    # Fill in the code here
    #################################################################################

    # Calculating the output height and width
    new_h = (input_feats.size(-2) + 2 * padding - kernel_size) // stride + 1
    new_w = (input_feats.size(-1) + 2 * padding - kernel_size) // stride + 1

    # Unfolding the inputs, weights, and output
    unfold_input = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
    input_unfolded = unfold_input(input_feats)

    weights_unfolded = weight.view(weight.size(0), -1)
    output_unfolded = weights_unfolded.matmul(input_unfolded)

    # Adding bias
    op_unf = output_unfolded.transpose(0, 1)
    for i in range(op_unf.size(0)):
      op_unf[i] += bias[i]
    output_unfolded = op_unf.transpose(0, 1)

    fold_output = nn.Fold(output_size=(new_h, new_w), kernel_size=(1,1))
    output = fold_output(output_unfolded)

    # save for backward (you need to save the unfolded tensor into ctx)
    # ctx.save_for_backward(your_vars, weight, bias)
    ctx.save_for_backward(input_unfolded, weight, bias)

    return output


  @staticmethod
  def backward(ctx, grad_output):
    """
    Backward propagation of convolution operation

    Args:
      grad_output: gradients of the outputs

    Outputs:
      grad_input: gradients of the input features
      grad_weight: gradients of the convolution weight
      grad_bias: gradients of the bias term

    """
    # unpack tensors and initialize the grads
    input_unfolded, weight, bias = ctx.saved_tensors
    grad_input = grad_weight = grad_bias = None

    # recover the conv params
    kernel_size = weight.size(2)
    stride = ctx.stride
    padding = ctx.padding
    input_height = ctx.input_height
    input_width = ctx.input_width

    #################################################################################
    # Fill in the code here
    #################################################################################
    # compute the gradients w.r.t. input and params

    unfold_grad_op = nn.Unfold(kernel_size=(1, 1))
    weight_unfolded = weight.view(weight.size(0), -1).transpose(0, 1)
    grad_op_unfolded = unfold_grad_op(grad_output)
    dx_unfolded = weight_unfolded.matmul(grad_op_unfolded)

    fold_dx = nn.Fold(output_size=(input_height, input_width), kernel_size=(kernel_size, kernel_size), stride=stride, padding=padding)
    dx = fold_dx(dx_unfolded)

    dw_unfolded = grad_op_unfolded.matmul(input_unfolded.transpose(1, 2))
    dw = dw_unfolded.sum(dim=0)
    dw_final = dw.view(weight.size())

    if bias is not None and ctx.needs_input_grad[2]:
      #compute the gradients w.r.t. bias (if any)
      grad_bias = grad_output.sum((0, 2, 3))

    return dx, dw_final, grad_bias, None, None


custom_conv2d = CustomConv2DFunction.apply

class CustomConv2d(Module):
  """
  The same interface as torch.nn.Conv2D
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
         padding=0, dilation=1, groups=1, bias=True):
    super(CustomConv2d, self).__init__()
    assert isinstance(kernel_size, int), "We only support squared filters"
    assert isinstance(stride, int), "We only support equal stride"
    assert isinstance(padding, int), "We only support equal padding"
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    # not used (for compatibility)
    self.dilation = dilation
    self.groups = groups

    # register weight and bias as parameters
    self.weight = nn.Parameter(torch.Tensor(
      out_channels, in_channels, kernel_size, kernel_size))
    if bias:
      self.bias = nn.Parameter(torch.Tensor(out_channels))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self):
  	# initialization using Kaiming uniform
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
      fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
      bound = 1 / math.sqrt(fan_in)
      nn.init.uniform_(self.bias, -bound, bound)

  def forward(self, input):
    # call our custom conv2d op
    return custom_conv2d(input, self.weight, self.bias, self.stride, self.padding)

  def extra_repr(self):
    s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
       ', stride={stride}, padding={padding}')
    if self.bias is None:
      s += ', bias=False'
    return s.format(**self.__dict__)

#################################################################################
# Part II: Design and train a network
#################################################################################
class SimpleNet(nn.Module):
  # a simple CNN for image classifcation
  def __init__(self, conv_op=nn.Conv2d, num_classes=100):
    super(SimpleNet, self).__init__()
    # you can start from here and create a better model
    self.features = nn.Sequential(
      # conv1 block: 3x conv 3x3
      conv_op(3, 64, kernel_size=7, stride=2, padding=3),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      # conv2 block: simple bottleneck
      conv_op(64, 64, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      conv_op(64, 256, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      # conv3 block: simple bottleneck
      conv_op(256, 64, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      conv_op(64, 256, kernel_size=1, stride=1, padding=0),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      # conv4 block: conv 3x3
      conv_op(256, 512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
    )
    # global avg pooling + FC
    self.avgpool =  nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, num_classes)

  def forward(self, x):
    # you can implement adversarial training here
    # if self.training:
    #   # generate adversarial sample based on x
    x = self.features(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


class SimplerNet(nn.Module):
  # a simple CNN for image classifcation
  def __init__(self, conv_op=nn.Conv2d, num_classes=100):
    super(SimplerNet, self).__init__()
    # you can start from here and create a better model

    # introconv: 1/4 spatial map, channels: 3->64
    self.introconv = nn.Sequential(
      # conv1 block: 3x conv 3x3
      conv_op(3, 64, kernel_size=7, stride=2, padding=3),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )

    # bottleneck 1 layer, 1/2 spatial, channels: 64->256
    self.bottleneck1 = nn.Sequential(
      # conv2 block: simple bottleneck
      conv_op(64, 64, kernel_size=1, stride=1, padding=0),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      conv_op(64, 256, kernel_size=1, stride=1, padding=0),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )

    # bottleneck 2 layer, 1/2 spatial, channels: 256->256
    self.bottleneck2 = nn.Sequential(
      # conv3 block: simple bottleneck
      conv_op(256, 64, kernel_size=1, stride=1, padding=0),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      conv_op(64, 256, kernel_size=1, stride=1, padding=0),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      # max pooling 1/2
      #nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )

    # Does bottlenecking twice, and maxpool is applied only once
    # bottleneck 2x layer, 1/2 spatial, channels: 256->256
    self.bottleneck2x = nn.Sequential(
      # simple bottleneck 1
      conv_op(256, 64, kernel_size=1, stride=1, padding=0),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      conv_op(64, 256, kernel_size=1, stride=1, padding=0),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),

      # simple bottleneck 2
      conv_op(256, 64, kernel_size=1, stride=1, padding=0),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      conv_op(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      conv_op(64, 256, kernel_size=1, stride=1, padding=0),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),

      # max pooling 1/2
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )

    # finalconv layer:
    self.finalconv = nn.Sequential(
      # conv4 block: conv 3x3
      conv_op(256, 512, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(inplace=True),
    )

    # downsampling block for skip connections:
    self.downconv_256 = nn.Sequential(
      # conv4 block: conv 3x3
      conv_op(256, 256, kernel_size=1, stride=2, bias=False),
    )

    # downsampling block for skip connections:
    self.maxpool_alt = nn.Sequential(
      # conv4 block: conv 3x3
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )

    # global avg pooling + FC
    self.avgpool =  nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, num_classes)

  def forward(self, x):
    # you can implement adversarial training here
    # if self.training:
    #   # generate adversarial sample based on x
    x = self.introconv(x)
    x = self.bottleneck1(x)

    preserve_1 = x

    x = self.bottleneck2x(x)

    preserve_1 = self.downconv_256(preserve_1)
    x = preserve_1 + x
    preserve_2 = x

    x = self.bottleneck2x(x)

    preserve_2 = self.downconv_256(preserve_2)
    x = preserve_2 + x
    
    x = self.finalconv(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


import torch.nn as nn
import torch.nn.functional as f


class SkipNet(nn.Module):
    # a simple CNN for image classifcation
    def __init__(self, conv_op=nn.Conv2d, num_classes=100):
        super(SkipNet, self).__init__()

        # introconv: 1/4 spatial map, channels: 3->64
        self.introconv = nn.Sequential(
            # conv1 block: 3x conv 3x3
            conv_op(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv_op(64, 64, kernel_size=3, stride=1, padding=1),
        )

        # bottleneck 1 layer, retains spatial, channels: 64->128
        self.bottleneck1 = nn.Sequential(
            conv_op(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_op(64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            # max pooling 1/2
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # identity connection for bottleneck 1
        self.id_bn1 = nn.Sequential(
            conv_op(64, 128, kernel_size=1, stride=1, bias=False),
        )

        # bottleneck 2 layer, retains spatial, channels: 128->256
        self.bottleneck2 = nn.Sequential(
            # conv3 block: simple bottleneck
            conv_op(128, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_op(64, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # identity connection for bottleneck 2
        self.id_bn2 = nn.Sequential(
            conv_op(128, 256, kernel_size=1, stride=1, bias=False),
        )

        # bottleneck 3 layer, 1/2 spatial, channels: 256->512
        self.bottleneck3 = nn.Sequential(
            conv_op(256, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv_op(64, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # identity connection for bottleneck 3
        self.id_bn3 = nn.Sequential(
            conv_op(256, 512, kernel_size=1, stride=2, bias=False),
        )
        # global avg pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, x):
        # 3x128x128 -> 64x32x32
        x = self.introconv(x)
        # 64x32x32 -> 128x32x32
        preserve_1 = x
        x = self.bottleneck1(x)
        preserve_1 = self.id_bn1(preserve_1)
        x = x + preserve_1
        x = nn.functional.relu(x, inplace=True)
        # 128x32x32 -> 256x16x16
        preserve_2 = x
        x = self.bottleneck2(x)
        preserve_2 = self.id_bn2(preserve_2)
        x = x + preserve_2
        x = nn.functional.relu(x, inplace=True)
        # 256x16x16 -> 512x8x8
        preserve_3 = x
        x = self.bottleneck3(x)
        preserve_3 = self.id_bn3(preserve_3)
        x = x + preserve_3
        x = nn.functional.relu(x, inplace=True)
        # 512x8x8 -> 512x8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


# change this to your model!
#default_model = SimpleNet
default_model = SkipNetModified
#default_model = SkipNet

#################################################################################
# Part III: Adversarial samples and Attention
#################################################################################
class PGDAttack(object):
  def __init__(self, loss_fn, num_steps=10, step_size=0.01, epsilon=0.1):
    """
    Attack a network by Project Gradient Descent. The attacker performs
    k steps of gradient descent of step size a, while always staying
    within the range of epsilon from the input image.

    Args:
      loss_fn: loss function used for the attack
      num_steps: (int) number of steps for PGD
      step_size: (float) step size of PGD
      epsilon: (float) the range of acceptable samples
               for our normalization, 0.1 ~ 6 pixel levels
    """
    self.loss_fn = loss_fn
    self.num_steps = num_steps
    self.step_size = step_size
    self.epsilon = epsilon

  def perturb(self, model, input):
    """
    Given input image X (torch tensor), return an adversarial sample
    (torch tensor) using PGD of the least confident label.

    See https://openreview.net/pdf?id=rJzIBfZAb

    Args:
      model: (nn.module) network to attack
      input: (torch tensor) input image of size N * C * H * W

    Outputs:
      output: (torch tensor) an adversarial sample of the given network
    """
    # clone the input tensor and disable the gradients
    output = input.clone()
    input.requires_grad = True

    # loop over the number of steps
    for _ in range(self.num_steps):
      #pass
      #################################################################################
      # Fill in the code here
      pred_scores=model(input)
      #print ("PRED SCORES SIZE",pred_scores.shape)
      values,indices=torch.min(pred_scores,0)
      #print (indices.shape)
      pred_scores=pred_scores.gather(1,indices.view(-1,1)).squeeze()
      #print ("AFTER GATHER",pred_scores)
      pred_scores.backward(torch.cuda.FloatTensor(pred_scores.size()).fill_(0))
      loss_grad=input.grad
      print ("LOSS SHAPE",loss_grad.shape)
      sign_grad=loss_grad.sign()
      adv_image=input+self.epsilon*sign_grad
      adv_image = torch.clamp(adv_image, 0, 1)
      output=adv_image
      #################################################################################

    return output

default_attack = PGDAttack


class GradAttention(object):
  def __init__(self, loss_fn):
    """
    Visualize a network's decision using gradients

    Args:
      loss_fn: loss function used for the attack
    """
    self.loss_fn = loss_fn

  def explain(self, model, input):
    """
    Given input image X (torch tensor), return a saliency map
    (torch tensor) by computing the max of abs values of the gradients
    given by the predicted label

    See https://arxiv.org/pdf/1312.6034.pdf

    Args:
      model: (nn.module) network to attack
      input: (torch tensor) input image of size N * C * H * W

    Outputs:
      output: (torch tensor) a saliency map of size N * 1 * H * W
    """
    # make sure input receive grads
    input.requires_grad = True
    if input.grad is not None:
      input.grad.zero_()

    #################################################################################
    # Fill in the code here
    pred_scores=model(input)
    values,indices=torch.max(pred_scores,0)
    pred_scores = pred_scores.gather(1, indices.view(-1, 1)).squeeze()
    pred_scores.backward(torch.cuda.FloatTensor(pred_scores.size()).fill_(1))
    sal=input.grad
    sal=sal.abs()
    sal,_=torch.max(sal,dim=1)
    sal=torch.reshape(sal,(sal.shape[0],1,sal.shape[1],sal.shape[2]))
    output=sal
    #################################################################################

    return output

default_attention = GradAttention

def vis_grad_attention(input, vis_alpha=2.0, n_rows=10, vis_output=None):
  """
  Given input image X (torch tensor) and a saliency map
  (torch tensor), compose the visualziations

  Args:
    input: (torch tensor) input image of size N * C * H * W
    output: (torch tensor) input map of size N * 1 * H * W

  Outputs:
    output: (torch tensor) visualizations of size 3 * HH * WW
  """
  # concat all images into a big picture
  input_imgs = make_grid(input.cpu(), nrow=n_rows, normalize=True)
  if vis_output is not None:
    output_maps = make_grid(vis_output.cpu(), nrow=n_rows, normalize=True)

    # somewhat awkward in PyTorch
    # add attention to R channel
    mask = torch.zeros_like(output_maps[0, :, :]) + 0.5
    mask = (output_maps[0, :, :] > vis_alpha * output_maps[0,:,:].mean())
    mask = mask.float()
    input_imgs[0,:,:] = torch.max(input_imgs[0,:,:], mask)
  output = input_imgs
  return output

default_visfunction = vis_grad_attention
