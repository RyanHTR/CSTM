import os
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch import optim, nn

from models.cstm_model import CSTMModel
from datasets.a2d_dataset import VideoTextDataset
from utils.tool import get_video_spatial_feature
from utils.tool import report_evaluation_result


def prepare_environment(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(opt.seed)
    print(f'environment prepared done: {opt}')


def prepare_model(opt):
    model = CSTMModel(opt)
    print(model)
    if ',' in opt.gpu_id:  # multiple gpus
        model = nn.DataParallel(model).cuda()
    else:                # single gpu
        model = model.cuda()

    if opt.resume is not None:
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint['epoch']
        print("Resume from {} with epoch {}".format(opt.resume, opt.start_epoch))
        save_dict = checkpoint['state_dict']
        model.load_state_dict(save_dict)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=opt.lr, weight_decay=opt.weight_decay)

    param_list = list(filter(lambda p: p.requires_grad, model.parameters()))
    print("{} trainable parameters".format(len(param_list)))

    return model, optimizer


def test(opt, savedir, dataloader, model, spatials):
    resume = os.path.join(opt.project_root, 'checkpoint', savedir, opt.checkpoint)
    print("Resume from ", resume)
    if os.path.isfile(resume):
        checkpoint = torch.load(resume)
        opt.start_epoch = checkpoint['epoch']

        save_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        trash_vars = [k for k in save_dict.keys() if k not in model_dict.keys()]
        if len(trash_vars) > 0:
            print(f'trashed vars from resume dict: {trash_vars}')
        resume_dict = {k: v for k, v in save_dict.items() if k in model_dict}
        model_dict.update(resume_dict)
        model.load_state_dict(model_dict)
    else:
        print("=> no checkpoint found at '{}'".format(resume))

    model.eval()
    mean_iou, overall_iou, precision5, precision6, precision7, precision8, precision9, precision_mAP = \
        report_evaluation_result(dataloader, model, spatials)

    print(f'Test split results:\n'
          f'Precision@0.5 {precision5:.3f}, Precision@0.6 {precision6:.3f}, '
          f'Precision@0.7 {precision7:.3f}, Precision@0.8 {precision8:.3f}, Precision@0.9 {precision9:.3f},\n'
          f'mAP Precision @0.5:0.05:0.95 {precision_mAP:.3f},\n'
          f'Overall IoU {overall_iou:.3f}, Mean IoU {mean_iou:.3f}')


if __name__ == '__main__':
    from opts import get_opts
    opt = get_opts()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    prepare_environment(opt)
    model, optimizer = prepare_model(opt)

    # prepare spatial and spatial_3d
    spatials = []
    for i in range(5):
        feat_size = opt.resize // 32 * (2 ** i)
        cur_spatial = get_video_spatial_feature(feat_size, feat_size)
        spatials.append(torch.from_numpy(cur_spatial).unsqueeze(0).repeat(opt.batch_size, 1, 1, 1).cuda())
    
    save_dir = '{}'.format(opt.model_root)
    dataloader = torch.utils.data.DataLoader(VideoTextDataset(opt, 'test'),
                                                batch_size=opt.batch_size, shuffle=False, pin_memory=True,
                                                num_workers=8, drop_last=True)
    test(opt, save_dir, dataloader, model, spatials)
