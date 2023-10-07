import torch.nn.functional as F
import torch
import logging
import torch.nn as nn

from typing import List

import torch
from torch import Tensor, einsum

# from .utils import simplex,one_hot



__all__ = ['sigmoid_dice_loss','softmax_dice_loss','GeneralizedDiceLoss','FocalLoss']

cross_entropy = F.cross_entropy


def FocalLoss(output, target, alpha=0.25, gamma=2.0):
    target[target == 4] = 3 # label [4] -> [3]
    # target = expand_target(target, n_class=output.size()[1]) # [N,H,W,D] -> [N,4,H,W,D]
    if output.dim() > 2:
        output = output.view(output.size(0), output.size(1), -1)  # N,C,H,W,D => N,C,H*W*D
        output = output.transpose(1, 2)  # N,C,H*W*D => N,H*W*D,C
        output = output.contiguous().view(-1, output.size(2))  # N,H*W*D,C => N*H*W*D,C
    if target.dim() == 5:
        target = target.contiguous().view(target.size(0), target.size(1), -1)
        target = target.transpose(1, 2)
        target = target.contiguous().view(-1, target.size(2))
    if target.dim() == 4:
        target = target.view(-1) # N*H*W*D
    # compute the negative likelyhood
    logpt = -F.cross_entropy(output, target)
    pt = torch.exp(logpt)
    # compute the loss
    loss = -((1 - pt) ** gamma) * logpt
    # return loss.sum()
    return loss.mean()

def dice(output, target,eps =1e-5): # soft dice loss
    target = target.float()
    # num = 2*(output*target).sum() + eps
    num = 2*(output*target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den

def sigmoid_dice_loss(output, target,alpha=1e-5):
    # output: [-1,3,H,W,T]   #这里默认给了 输出通道是3
    # target: [-1,H,W,T] noted that it includes 0,1,2,4 here
    loss1 = dice(output[:,0,...],(target==1).float(),eps=alpha)
    loss2 = dice(output[:,1,...],(target==2).float(),eps=alpha)
    loss3 = dice(output[:,2,...],(target==4).float(),eps=alpha)
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(1-loss1.data, 1-loss2.data, 1-loss3.data))
    return loss1+loss2+loss3

def seg3_sigmoid_dice_loss(output, target,alpha=1e-5):
    # output: [-1,3,H,W,T]   #这里默认给了 输出通道是3
    # target: [-1,H,W,T] noted that it includes 0,1,2,4 here
    loss1 = dice(output[:,0,...],((target==1)|(target==2)|(target==4)).float(),eps=alpha)
    loss2 = dice(output[:,1,...],((target==1)|(target==4)).float(),eps=alpha)
    loss3 = dice(output[:,2,...],(target==4).float(),eps=alpha)
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(1-loss1.data, 1-loss2.data, 1-loss3.data))
    #WT = 1+2+4, TC = 1+4, WT= 4    使用此损失函数必须修改predict
    return loss1+loss2+loss3



def seg3_softmax_dice_loss(output, target,eps=1e-5): #
    # output : [bsize,c,H,W,D]
    # target : [bsize,H,W,D]
    loss1 = dice(output[:,1,...],((target==1)|(target==2)|(target==4)).float())
    loss2 = dice(output[:,2,...],((target==1)|(target==4)).float())
    loss3 = dice(output[:,3,...],(target==4).float())
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(1-loss1.data, 1-loss2.data, 1-loss3.data))

    return loss1+loss2+loss3

def softmax_dice_loss(output, target,eps=1e-5): #
    # output : [bsize,c,H,W,D]
    # target : [bsize,H,W,D]
    loss1 = dice(output[:,1,...],(target==1).float())
    loss2 = dice(output[:,2,...],(target==2).float())
    loss3 = dice(output[:,3,...],(target==4).float())
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(1-loss1.data, 1-loss2.data, 1-loss3.data))
    # WT = 1+2+4, TC = 1+4, WT= 4  使用此损失函数，必须修改predict
    return loss1+loss2+loss3


# Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
def GeneralizedDiceLoss(output,target,eps=1e-5,weight_type='square'): # Generalized dice loss
    """
        Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
    """

    # target = target.float()

    if target.dim() == 4:   #target size[240,240,155],but have 4 labels [0,1,2,4]
        target[target == 4] = 3 # label [4] -> [3]
        target = expand_target(target, n_class=output.size()[1]) # [N,H,W,D] -> [N,4，H,W,D]

    output = flatten(output)[1:,...] # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:,...] # [class, N*H*W*D]
    #这里应该是忽略了标签0，即背景
    target_sum = target.sum(-1) # sub_class_voxels [3,1] -> 3个voxels target_sum [3]
    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :',weight_type)

    # print(class_weights)
    intersect = (output * target).sum(-1)
    intersect_sum = (intersect * class_weights).sum()
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    loss1 = 2*intersect[0] / (denominator[0] + eps)   #dice
    loss2 = 2*intersect[1] / (denominator[1] + eps)
    loss3 = 2*intersect[2] / (denominator[2] + eps)
    logging.info('GDL: 1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(loss1.data, loss2.data, loss3.data))
    # 这里其实是输出第一行
    return 1 - 2. * intersect_sum / denominator_sum

def DomainDiceLoss_fromGDL(output,target,eps=1e-5,weight_type='square'): # Generalized dice loss
    # actually dice for ET,TC,WT

    if target.dim() == 4:   #target size[240,240,155],but have 4 labels [0,1,2,4]
        target[target == 4] = 3 # label [4] -> [3]
        target = expand_target(target, n_class=output.size()[1]) # [N,H,W,D] -> [N,4，H,W,D]

    output = flatten(output)[1:,...] # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:,...] # [class, N*H*W*D]

    intersect = torch.zeros(3)
    denominator = torch.zeros(3)
    intersect_T = (output * target).sum(-1)  # sub_class_voxels [3,N*D*H*W] -> [3] 3个voxels
    intersect[0] = intersect_T[2]
    intersect[1] = intersect_T[2] + intersect_T[0]
    intersect[2] = intersect_T[2] + intersect_T[0] + intersect_T[1]

    denominator_T = (output + target).sum(-1)
    denominator[0] = denominator_T[2] #label 3
    denominator[1] = denominator_T[2] + denominator_T[0] #label 1+3
    denominator[2] = denominator_T[2] + denominator_T[0] + denominator_T[1] #label 1+2+3

    loss_ET = 2*intersect[0] / (denominator[0] + eps)
    loss_TC = 2*intersect[1] / (denominator[1] + eps)
    loss_WT = 2*intersect[2] / (denominator[2] + eps)
    logging.info('GDL: E:{:.4f} | T:{:.4f} | W:{:.4f}'.format(loss_ET.data, loss_TC.data, loss_WT.data))
    return ((1-loss_ET)+(1-loss_TC)+(1-loss_WT))/3

def Domain_NoWeighted_GeneralizedDiceLoss(output,target,eps=1e-5,weight_type='square'): # Generalized dice loss

    if target.dim() == 4:   #target size[240,240,155],but have 4 labels [0,1,2,4]
        target[target == 4] = 3 # label [4] -> [3]
        target = expand_target(target, n_class=output.size()[1]) # [N,H,W,D] -> [N,4，H,W,D]

    output = flatten(output)[1:,...] # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:,...] # [class, N*H*W*D]
    #这里应该是忽略了标签0，即背景
    target_sum = target.sum(-1) # sub_class_voxels [3,1] -> 3个voxels
    if weight_type == 'square':
        class_weights = 1.
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :',weight_type)

    # print(class_weights)
    intersect = torch.zeros(3)
    denominator = torch.zeros(3)
    intersect_T = (output * target).sum(-1)   #sub_class_voxels [3,1] -> 3个voxels
    intersect[0] = intersect_T[2]
    intersect[1] = intersect_T[2]+intersect_T[0]
    intersect[2] = intersect_T[2]+intersect_T[0]+intersect_T[1]
    intersect_sum = (intersect * class_weights).sum()

    denominator_T = (output + target).sum(-1)
    denominator[0] = denominator_T[2]
    denominator[1] = denominator_T[2]+denominator_T[0]
    denominator[2] = denominator_T[2]+denominator_T[0]+denominator_T[1]
    denominator_sum = (denominator * class_weights).sum() + eps

    # 这里其实是输出第一行
    loss_ET = 2*intersect[0] / (denominator[0] + eps)
    loss_TC = 2*intersect[1] / (denominator[1] + eps)
    loss_WT = 2*intersect[2] / (denominator[2] + eps)
    logging.info('GDL: E:{:.4f} | T:{:.4f} | W:{:.4f}'.format(loss_ET.data, loss_TC.data, loss_WT.data))
    return 1 - 2. * intersect_sum / denominator_sum


def Domain_Weighted_GeneralizedDiceLoss(output, target, eps=1e-5, weight_type='square'):  # Generalized dice loss

    if target.dim() == 4:  # target size[240,240,155],but have 4 labels [0,1,2,4]
        target[target == 4] = 3  # label [4] -> [3]
        target = expand_target(target, n_class=output.size()[1])  # [N,H,W,D] -> [N,4，H,W,D]

    output = flatten(output)[1:, ...]  # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:, ...]  # [class, N*H*W*D]
    # 这里应该是忽略了标签0，即背景

    intersect = torch.zeros(3)
    # intersect = []
    denominator = torch.zeros(3)
    # denominator = []
    intersect_T = (output * target).sum(-1)  # sub_class_voxels [3,N*H*W*D] -> [3] 3个voxels
    intersect[0] = intersect_T[2]
    intersect[1] = intersect_T[2] + intersect_T[0]
    intersect[2] = intersect_T[2] + intersect_T[0] + intersect_T[1]
    # intersect.append()
    print('intersect',intersect)
    denominator_T = (output + target).sum(-1)
    denominator[0] = denominator_T[2]
    denominator[1] = denominator_T[2] + denominator_T[0]
    denominator[2] = denominator_T[2] + denominator_T[0] + denominator_T[1]
    print('denominator',denominator)
    target_sum = denominator.sum(-1)  # sub_class_voxels [3,1] -> 3个voxels
    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :', weight_type)

    intersect_sum = (intersect * class_weights).sum()
    denominator_sum = (denominator * class_weights).sum() + eps

    # 这里其实是输出第一行
    loss_ET = 2 * intersect[0] / (denominator[0] + eps)
    loss_TC = 2 * intersect[1] / (denominator[1] + eps)
    loss_WT = 2 * intersect[2] / (denominator[2] + eps)
    logging.info('DWG: E:{:.4f} | T:{:.4f} | W:{:.4f}'.format(loss_ET.data, loss_TC.data, loss_WT.data))
    return 1 - 2. * intersect_sum / denominator_sum

def expand_target(x, n_class,mode='softmax'):
    """
        Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :return: 5D output image (NxCxDxHxW)
        """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        xx[:,1,:,:,:] = (x == 1)
        xx[:,2,:,:,:] = (x == 2)
        xx[:,3,:,:,:] = (x == 3)
    if mode.lower() == 'sigmoid':
        xx[:,0,:,:,:] = (x == 1)
        xx[:,1,:,:,:] = (x == 2)
        xx[:,2,:,:,:] = (x == 3)
    return xx.to(x.device)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)

# 以下出自boundary loss
class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = target[:, self.idc, ...].type(torch.float32)

        loss = - einsum("bcwh,bcwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class GeneralizedDice():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss = divided.mean()

        return loss


class DiceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - (2 * intersection + 1e-10) / (union + 1e-10)

        loss = divided.mean()

        return loss


class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return loss


