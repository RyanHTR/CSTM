import argparse

parser = argparse.ArgumentParser(description='Experiment on Actor and Action Video Segmentation')
# Project Structure
parser.add_argument('--project_root', type=str, default='.')
parser.add_argument('--savedir_root', type=str, default='.')
parser.add_argument('--model_root', type=str, default='')
parser.add_argument('--checkpoint', type=str, default='checkpoint.pth.tar')
parser.add_argument('--resume', type=str, default=None)

# Dataset Specific
parser.add_argument('--dataset', type=str, choices=['A2D', 'JHMDB', 'YTVOS'])
parser.add_argument('--modality', type=str, default='rgb', choices=['rgb', 'flow'])
parser.add_argument('--data_margin', type=int, default=1)
parser.add_argument('--resize', type=int, default=320)
parser.add_argument('--single_im', action='store_true', default=False)

# Model Specific
parser.add_argument('--arch', type=str, default='acga')
parser.add_argument('--sentence_length', type=int, default=20)
parser.add_argument('--dim_semantic', type=int, default=300)
parser.add_argument('--dim_spatial', type=int, default=8)
parser.add_argument('--up_mask', action='store_true', default=False)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--use_dp', action='store_true', default=False)
parser.add_argument('--skip', type=str, choices=[None, 'single', 'mean', 'all'], default='single')

# Training Specific
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--lr_action', type=str, default='multi_step')
parser.add_argument('--diff_lr', action='store_true', default=False)
parser.add_argument('--nepoch', type=int, default=9)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr_step', type=int, default=8)
parser.add_argument('--warmup_epoch', type=int, default=2)
parser.add_argument('--gamma_step', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=2019)
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--loss', type=str, choices=['bce', 'focal', 'dice'], default='bce')
parser.add_argument('--maxpool_size', type=int, default=8)

# Misc
parser.add_argument('--testing', action='store_true', default=False)
parser.add_argument('--pos_weight', type=float, default=1.5)
parser.add_argument('--save_freq', type=int, default=1)
parser.add_argument('--save_epoch', type=int, default=1)
parser.add_argument('--preprocess_clip', action='store_true', default=False)


def get_opts():
    opt = parser.parse_args()
    return opt
