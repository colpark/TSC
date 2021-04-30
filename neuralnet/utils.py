from typing import Optional
import glob
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt as distance

def choose_model(model_addr, n=1):
    addrs = glob.glob(model_addr + '/*')
    scores = []
    for ad in addrs:
        score = float(ad.split('_')[-2][4:])
        scores.append(score)
    order = np.array(scores).argsort()[::-1]
    return addrs[order[n]]

def DICE_new(input, target, epsilon=1e-6):

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    batch_size = input.size(0)
    input = input.contiguous().view(batch_size, -1).float()
    target = target.contiguous().view(batch_size, -1).float()

    intersect = (input * target).sum(-1)
    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input + target).sum(-1)
    dice = 2 * (intersect / denominator.clamp(min=epsilon))
    assert dice.size(0) ==  batch_size
    return dice.mean()

def JACCARD_new(input, target, epsilon=1e-6):

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    batch_size = input.size(0)
    input = input.contiguous().view(batch_size, -1).float()
    target = target.contiguous().view(batch_size, -1).float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input).sum(-1) + (target).sum(-1)
    jaccard = (intersect / (denominator-intersect).clamp(min=epsilon))
    assert jaccard.size(0) == batch_size
    return jaccard.mean()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def multimodal_dropout(img, both_prob = 0.5):
    """
    Randomly assign 0 to either T1 or flair. For (1-both_prob) prob, either T1 or FLAIR will be dropped to 0
    :param img: torch tensor of batch x 2 x 256 x 256
           both_prob: prob of having both modality (T1 + FLAIR)
    """
    both = random.random() < both_prob
    if both:
        dropped_img = img
    else:
        flair = random.random() < 0.5
        if flair:
            img[:,1,:,:] = 0.0
        else:
            img[:,0,:,:] = 0.0
        dropped_img = img

    return dropped_img


def znormalize(brain):
    brain = (brain - brain.mean()) / brain.std()
    print(brain.std())
    return brain

def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
    r"""Converts an integer label 2D tensor to a one-hot 3D tensor.

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, H, W)`,
                                where N is batch siz. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.

    Returns:
        torch.Tensor: the labels in one hot tensor.

    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> tgm.losses.one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not len(labels.shape) == 3:
        raise ValueError("Invalid depth shape, we expect BxHxW. Got: {}"
                         .format(labels.shape))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    batch_size, height, width = labels.shape
    one_hot = torch.zeros(batch_size, num_classes, height, width,
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))



def compute_dtm(img_gt, out_shape):
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM)
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """

    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(1, out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                fg_dtm[b][c] = posdis

    return fg_dtm

def hd_loss(seg_soft, gt, seg_dtm, gt_dtm):
    """
    compute huasdorff distance loss for binary segmentation
    input: seg_soft: softmax results,  shape=(b,2,x,y,z)
           gt: ground truth, shape=(b,x,y,z)
           seg_dtm: segmentation distance transform map; shape=(b,2,x,y,z)
           gt_dtm: ground truth distance transform map; shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """

    delta_s = (seg_soft[:,1,...] - gt.float()) ** 2
    s_dtm = seg_dtm[:,1,...] ** 2
    g_dtm = gt_dtm[:,1,...] ** 2
    dtm = s_dtm + g_dtm
    multipled = torch.einsum('bxy, bxy->bxy', delta_s, dtm)
    hd_loss = multipled.mean()

    return hd_loss

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        predict = F.softmax(predict, dim=1)
        target = one_hot(target, num_classes=predict.shape[1],
                                 device=predict.device, dtype=predict.dtype)
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        # predict = F.softmax(predict, dim=1)

        for i in range(1, target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]

#


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def display_row(imgs, user_cut, axis_idx, title, save_dir):
    f = plt.figure(figsize=(15, 4))
    for i, img in enumerate(imgs):
        if user_cut is None:
            cut = int(img.shape[axis_idx] / 2)
        else:
            cut = user_cut
        f.add_subplot(1, len(imgs), i + 1)
        plt.imshow(img.take([cut], axis=axis_idx).squeeze(), cmap="gray")
        # plt.colorbar()
        plt.gca().set_axis_off()
        if title is not None:
            plt.title(title[i])
    plt.savefig(save_dir)
    # plt.show()


def save_many(*images, save_dir, axis='z', Title=None, if_numpy=False, cut=None):
    """
    Input:
        images: a number of numpy image addresses, optimal below 5
        if_numpy: True if numpy array. Nibabel import is omitted when True.
        Title: title for the plotting. If none, pass.
        cut: explicitly say which slice to look
        axis: which axis to see from ('x', 'y', 'z')
    """
    axis_dict = {'x': 0, 'y': 1, 'z': 2}
    imgs = []
    if if_numpy == True:
        imgs = images
    else:
        for img in images:
            nib_img = nib.load(img).get_fdata()
            print(img.split('/')[-1], nib_img.shape)
            imgs.append(nib_img)

    if axis == 'all':
        for axis in axis_dict.keys():
            display_row(imgs, cut, axis_dict[axis], Title, save_dir)
    elif axis in axis_dict.keys():
        display_row(imgs, cut, axis_dict[axis], Title, save_dir)

