from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# code modified from pytorch imagenet example

# python imports
import argparse
import os
import time
import math

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

# for visualization
from tensorboardX import SummaryWriter

# our code
import custom_transforms as transforms
from custom_dataloader import MiniPlacesLoader
from utils import AverageMeter

# all your code goes into student_code.py
# part I and II
from student_code import CustomConv2d, default_model
# part III
from student_code import default_attack, default_attention, default_visfunction

# the arg parser
parser = argparse.ArgumentParser(description='PyTorch Image Classification')
parser.add_argument('data_folder', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup-epochs', default=0, type=int,
                    help='number of epochs for warmup')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-a', '--attack', dest='attack', action='store_true',
                    help='Attack with adersarial samples on validation set')
parser.add_argument('-v', '--vis', dest='vis', action='store_true',
                    help='Visualize the attention map')
parser.add_argument('--use-custom-conv', action='store_true',
                    help='Use custom convolution')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU ID to use.')

# tensorboard writer
writer = SummaryWriter('../logs')

# main function for training and testing
def main(args):
  # parse args
  best_acc1 = 0.0

  if args.gpu >= 0:
    print("Use GPU: {}".format(args.gpu))
  else:
    print('You are using CPU for computing!',
          'Yet we assume you are using a GPU.',
          'You will NOT be able to switch between CPU and GPU training!')

  # set up the model + loss
  if args.use_custom_conv:
    print("Using custom convolutions in the network")
    model = default_model(conv_op=CustomConv2d, num_classes=100)
  else:
    model = default_model(num_classes=100)
  model_arch = "simplenet"
  criterion = nn.CrossEntropyLoss()
  # put everthing to gpu
  if args.gpu >= 0:
    model = model.cuda(args.gpu)
    criterion = criterion.cuda(args.gpu)

  # setup the optimizer
  optimizer = torch.optim.SGD(model.parameters(), args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay)

  # resume from a checkpoint?
  if args.resume:
    if os.path.isfile(args.resume):
      print("=> loading checkpoint '{}'".format(args.resume))
      checkpoint = torch.load(args.resume)
      args.start_epoch = checkpoint['epoch']
      best_acc1 = checkpoint['best_acc1']
      model.load_state_dict(checkpoint['state_dict'])
      if args.gpu < 0:
        model = model.cpu()
      else:
        model = model.cuda(args.gpu)
      # only load the optimizer if necessary
      if (not args.evaluate) and (not args.attack):
        optimizer.load_state_dict(checkpoint['optimizer'])
      print("=> loaded checkpoint '{}' (epoch {}, acc1 {})"
          .format(args.resume, checkpoint['epoch'], best_acc1))
    else:
      print("=> no checkpoint found at '{}'".format(args.resume))

  # set up transforms for data augmentation
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
  train_transforms = []
  train_transforms.append(transforms.Scale(160))
  train_transforms.append(transforms.RandomHorizontalFlip())
  train_transforms.append(transforms.RandomColor(0.15))
  train_transforms.append(transforms.RandomRotate(15))
  train_transforms.append(transforms.RandomSizedCrop(128))
  train_transforms.append(transforms.ToTensor())
  train_transforms.append(normalize)
  train_transforms = transforms.Compose(train_transforms)
  # val transofrms
  val_transforms=[]
  val_transforms.append(transforms.Scale(160, interpolations=None))
  val_transforms.append(transforms.ToTensor())
  val_transforms.append(normalize)
  val_transforms = transforms.Compose(val_transforms)
  if (not args.evaluate) and (not args.attack):
    print("Training time data augmentations:")
    print(train_transforms)

  # setup dataset and dataloader
  train_dataset = MiniPlacesLoader(args.data_folder,
                  split='train', transforms=train_transforms)
  val_dataset = MiniPlacesLoader(args.data_folder,
                  split='val', transforms=val_transforms)

  train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)
  val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=100, shuffle=False,
    num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)

  # testing only
  if (args.evaluate==args.attack) and args.evaluate:
    print("Cann't set evaluate and attack to True at the same time!")
    return

  # set up visualizer
  if args.vis:
    visualizer = default_attention(criterion)
  else:
    visualizer = None

  # evaluation
  if args.resume and args.evaluate:
    print("Testing the model ...")
    cudnn.deterministic = True
    validate(val_loader, model, -1, args, visualizer=visualizer)
    return

  # attack
  if args.resume and args.attack:
    print("Generating adversarial samples for the model ..")
    cudnn.deterministic = True
    validate(val_loader, model, -1, args,
             attacker=default_attack(criterion),
             visualizer=visualizer)
    return

  # enable cudnn benchmark
  cudnn.enabled = True
  cudnn.benchmark = True

  # warmup the training
  if (args.start_epoch == 0) and (args.warmup_epochs > 0):
    print("Warmup the training ...")
    for epoch in range(0, args.warmup_epochs):
      train(train_loader, model, criterion, optimizer, epoch, "warmup", args)

  # start the training
  print("Training the model ...")
  for epoch in range(args.start_epoch, args.epochs):
    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch, "train", args)

    # evaluate on validation set
    acc1 = validate(val_loader, model, epoch, args)

    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)
    save_checkpoint({
      'epoch': epoch + 1,
      'model_arch': model_arch,
      'state_dict': model.state_dict(),
      'best_acc1': best_acc1,
      'optimizer' : optimizer.state_dict(),
    }, is_best)


def save_checkpoint(state, is_best,
                    file_folder="../models/", filename='checkpoint.pth.tar'):
  """save checkpoint"""
  if not os.path.exists(file_folder):
    os.mkdir(file_folder)
  torch.save(state, os.path.join(file_folder, filename))
  if is_best:
    # skip the optimization state
    state.pop('optimizer', None)
    torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def train(train_loader, model, criterion, optimizer, epoch, stage, args):
  """Training the model"""
  assert stage in ["train", "warmup"]
  # adjust the learning rate
  num_iters = len(train_loader)
  lr = 0.0

  # set up meters
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to train mode
  model.train()

  end = time.time()
  for i, (input, target) in enumerate(train_loader):
    # adjust the learning rate
    if stage == "warmup":
      # warmup: linear scaling
      lr = (epoch * num_iters + i) / float(
        args.warmup_epochs * num_iters) * args.lr
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = 0.0
    else:
      # cosine learning rate decay
      lr = 0.5 * args.lr * (1 + math.cos(
        (epoch * num_iters + i) / float(args.epochs * num_iters) * math.pi))
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = args.weight_decay

    # measure data loading time
    data_time.update(time.time() - end)
    if args.gpu >= 0:
      input = input.cuda(args.gpu, non_blocking=True)
      target = target.cuda(args.gpu, non_blocking=True)

    # compute output
    output = model(input)
    loss = criterion(output, target)

    # measure accuracy and record loss
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    losses.update(loss.item(), input.size(0))
    top1.update(acc1[0], input.size(0))
    top5.update(acc5[0], input.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    # printing
    if i % args.print_freq == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
         epoch + 1, i, len(train_loader), batch_time=batch_time,
         data_time=data_time, loss=losses, top1=top1, top5=top5))
      # log loss / lr
      if stage == "train":
        writer.add_scalar('data/training_loss',
          losses.val, epoch * num_iters + i)
        writer.add_scalar('data/learning_rate',
          lr, epoch * num_iters + i)


  # print the learning rate
  print("[Stage {:s}]: Epoch {:d} finished with lr={:f}".format(
            stage, epoch + 1, lr))
  # log top-1/5 acc
  writer.add_scalars('data/top1_accuracy',
    {"train" : top1.avg}, epoch + 1)
  writer.add_scalars('data/top5_accuracy',
    {"train" : top5.avg}, epoch + 1)


def validate(val_loader, model, epoch, args, attacker=None, visualizer=None):
  """Test the model on the validation set"""
  batch_time = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to evaluate mode (autograd will still track the graph!)
  model.eval()

  # disable/enable gradients
  grad_flag = (attacker is not None) or (visualizer is not None)
  with torch.set_grad_enabled(grad_flag):
    end = time.time()
    # loop over validation set
    for i, (input, target) in enumerate(val_loader):
      if args.gpu >= 0:
        input = input.cuda(args.gpu, non_blocking=False)
        target = target.cuda(args.gpu, non_blocking=False)

      # generate adversarial samples
      if attacker is not None:
        # generate adversarial samples
        adv_input = attacker.perturb(model, input)
        # forward the model
        output = model(adv_input)
      else:
        # forward the model
        output = model(input)

      # test time augmentation (minor performance boost)
      if args.evaluate:
        flipped_input = torch.flip(input, (3,))
        flipped_output = model(flipped_input)
        output = 0.5 * (output + flipped_output)

      # measure accuracy and record loss
      acc1, acc5 = accuracy(output, target, topk=(1, 5))
      top1.update(acc1[0], input.size(0))
      top5.update(acc5[0], input.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      # printing
      if i % args.print_freq == 0:
        print('Test: [{0}/{1}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
           i, len(val_loader), batch_time=batch_time,
           top1=top1, top5=top5))

        # visualize the results
        if args.vis and args.evaluate:
          vis_output = visualizer.explain(model, input)
          vis_output = default_visfunction(input, vis_output=vis_output)
          writer.add_image("Image/Image_Atten", vis_output, i)
        if args.vis and args.attack:
          vis_input = default_visfunction(input)
          vis_adv_input = default_visfunction(adv_input)
          writer.add_image("Image/Org_Image", vis_input, i)
          writer.add_image("Image/Adv_Image", vis_adv_input, i)

  print('******Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
            top1=top1, top5=top5))

  if (not args.evaluate) and (not args.attack):
    # log top-1/5 acc
    writer.add_scalars('data/top1_accuracy',
      {"val" : top1.avg}, epoch + 1)
    writer.add_scalars('data/top5_accuracy',
      {"val" : top5.avg}, epoch + 1)

  return top1.avg

def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
