import os
import shutil
import cv2
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        filepath = '/'.join(filename.split('/')[0:-1])
        shutil.copyfile(filename, os.path.join(filepath, 'model_best.pth.tar'))


def adjust_learning_rate(opt, optimizer, epoch, iter=0, total_iters=0, iters_per_epoch=0, warmup_iters=0,
                             action='multi_step'):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    T = epoch * iters_per_epoch + iter
    if warmup_iters > 0 and T < warmup_iters:
        lr = opt.lr * 1.0 * T / warmup_iters
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 1.0 * T / warmup_iters
    if action == 'multi_step':
        lr = opt.lr * (opt.gamma_step ** (epoch // opt.lr_step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif action == 'poly':
        T = T - warmup_iters
        lr = opt.lr * (0.9 * pow((1 - 1.0 * T / total_iters), 0.9) + 0.1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        raise Exception("Cannot recognize action of ", action)

    return lr


def get_video_spatial_feature(featmap_H, featmap_W):
    spatial_batch_val = np.zeros((8, featmap_H, featmap_W), dtype=np.float32)
    for h in range(featmap_H):
        for w in range(featmap_W):
            xmin = w / featmap_W * 2 - 1
            xmax = (w + 1) / featmap_W * 2 - 1
            xctr = (xmin + xmax) / 2
            ymin = h / featmap_H * 2 - 1
            ymax = (h + 1) / featmap_H * 2 - 1
            yctr = (ymin + ymax) / 2
            spatial_batch_val[:, h, w] = [xmin, ymin, xmax, ymax, xctr, yctr, 1 / featmap_W, 1 / featmap_H]
    return spatial_batch_val


def resize(im, input_h, input_w):
    new_im = cv2.resize(im, (input_w, input_h))
    return new_im


SMOOTH = 1e-6
def calculate_IoU(pred, gt):
    IArea = (pred & (gt == 1.0)).astype(float).sum()
    OArea = (pred | (gt == 1.0)).astype(float).sum()
    IoU = (IArea + SMOOTH) / (OArea + SMOOTH)
    return IoU, IArea, OArea


def report_evaluation_result(dataloader, model, spatials):
    MeanIoU, IArea, OArea, Overlap = [], [], [], []
    idx = 0
    bar = tqdm(dataloader)
    for data in bar:
        size, video, txt, txt_mask, mask_large = data
        idx += 1
        video, txt_mask, txt = video.cuda(), txt_mask.cuda(), txt.cuda()
        size, mask_large = size.numpy(), mask_large.numpy()

        with torch.no_grad():
            predictions = model(video, txt, txt_mask, spatials)

            res6 = torch.sigmoid(predictions[-1]) * 255.0
            res6 = res6.detach().cpu().numpy()
            pred = [resize((res6[i] > np.amax(res6[i]) * 0.5).astype(np.uint8), size[i][0], size[i][1]) for i in range(res6.shape[0])]
            gt = [resize((mask_large[i]).astype(np.uint8), size[i][0], size[i][1]) for i in range(res6.shape[0])]

            for i in range(len(pred)):
                iou, iarea, oarea = calculate_IoU(pred[i], gt[i])
                MeanIoU.append(iou)
                IArea.append(iarea)
                OArea.append(oarea)
                Overlap.append(iou)

        bar.set_description(f"{idx}, mean_iou:{np.mean(np.array(MeanIoU)):.3f}, overall_iou:{np.array(IArea).sum() / np.array(OArea).sum():.3f}")
    prec5, prec6, prec7, prec8, prec9 = np.zeros((len(Overlap), 1)), np.zeros((len(Overlap), 1)), np.zeros((len(Overlap), 1)), \
                                        np.zeros((len(Overlap), 1)), np.zeros((len(Overlap), 1))
    for i in range(len(Overlap)):
        if Overlap[i] >= 0.5:
            prec5[i] = 1
        if Overlap[i] >= 0.6:
            prec6[i] = 1
        if Overlap[i] >= 0.7:
            prec7[i] = 1
        if Overlap[i] >= 0.8:
            prec8[i] = 1
        if Overlap[i] >= 0.9:
            prec9[i] = 1

    mAP_thres_list = list(range(50, 95+1, 5))
    mAP = []
    for i in range(len(mAP_thres_list)):
        tmp = np.zeros((len(Overlap), 1))
        for j in range(len(Overlap)):
            if Overlap[j] >= mAP_thres_list[i] / 100.0:
                tmp[j] = 1
        mAP.append(tmp.sum() / tmp.shape[0])

    return np.mean(np.array(MeanIoU)), np.array(IArea).sum() / np.array(OArea).sum(), \
           prec5.sum() / prec5.shape[0], prec6.sum() / prec6.shape[0], prec7.sum() / prec7.shape[0], \
           prec8.sum() / prec8.shape[0], prec9.sum() / prec9.shape[0], np.mean(np.array(mAP))
