"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
# 获取当前文件的路径
import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将当前文件夹路径添加到Python路径中
sys.path.append(current_dir)
import argparse
import torch
import torch.nn as nn
from core.solver import Solver
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
from core.model import build_model
from os.path import join as ospj
from core.checkpoint import CheckpointIO
from torch.backends import cudnn
import torchvision.utils as vutils
from core.get_dataset import get_loader
from tqdm import tqdm
import glob
device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser()
parser.add_argument("--gpus", type=int, default=0)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--hmm", type=bool, default=False)
parser.add_argument("--att", type=bool, default=False)
parser.add_argument("--JPEG", type=bool, default=False)
parser.add_argument("--feat", type=bool, default=False)
parser.add_argument("--id", type=bool, default=False)
parser.add_argument("--model", type=str, default="advgan")
parser.add_argument("--mask", type=str, default="None")
parser.add_argument("--KPI", type=str, default=False)
args = parser.parse_args()
# model arguments
parser.add_argument('--img_size', type=int, default=256,
                    help='Image resolution')
parser.add_argument('--num_domains', type=int, default=2,
                    help='Number of domains')
parser.add_argument('--latent_dim', type=int, default=16,
                    help='Latent vector dimension')
parser.add_argument('--hidden_dim', type=int, default=512,
                    help='Hidden dimension of mapping network')
parser.add_argument('--style_dim', type=int, default=64,
                    help='Style code dimension')

# weight for objective functions
parser.add_argument('--lambda_reg', type=float, default=1,
                    help='Weight for R1 regularization')
parser.add_argument('--lambda_cyc', type=float, default=1,
                    help='Weight for cyclic consistency loss')
parser.add_argument('--lambda_sty', type=float, default=1,
                    help='Weight for style reconstruction loss')
parser.add_argument('--lambda_ds', type=float, default=1,
                    help='Weight for diversity sensitive loss')
parser.add_argument('--ds_iter', type=int, default=100000,
                    help='Number of iterations to optimize diversity sensitive loss')
parser.add_argument('--w_hpf', type=float, default=1,
                    help='weight for high-pass filtering')

# training arguments
parser.add_argument('--randcrop_prob', type=float, default=0.5,
                    help='Probabilty of using random-resized cropping')
parser.add_argument('--total_iters', type=int, default=100000,
                    help='Number of total iterations')
parser.add_argument('--resume_iter', type=int, default=100000,
                    help='Iterations to resume training/testing')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch size for training')
parser.add_argument('--val_batch_size', type=int, default=32,
                    help='Batch size for validation')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate for D, E and G')
parser.add_argument('--f_lr', type=float, default=1e-6,
                    help='Learning rate for F')
parser.add_argument('--beta1', type=float, default=0.0,
                    help='Decay rate for 1st moment of Adam')
parser.add_argument('--beta2', type=float, default=0.99,
                    help='Decay rate for 2nd moment of Adam')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay for optimizer')
parser.add_argument('--num_outs_per_domain', type=int, default=10,
                    help='Number of generated images per domain during sampling')

# misc
parser.add_argument('--mode', default="sample",type=str,
                    choices=['train', 'sample', 'eval', 'align'],
                    help='This argument is used in solver')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers used in DataLoader')
parser.add_argument('--seed', type=int, default=777,
                    help='Seed for random number generator')

# directory for training
parser.add_argument('--train_img_dir', type=str, default='data/celeba_hq/train',
                    help='Directory containing training images')
parser.add_argument('--val_img_dir', type=str, default='data/celeba_hq/val',
                    help='Directory containing validation images')
parser.add_argument('--sample_dir', type=str, default='expr/samples',
                    help='Directory for saving generated images')
parser.add_argument('--checkpoint_dir', type=str, default='starganv2/expr/checkpoints/celeba_hq',
                    help='Directory for saving network checkpoints')

# directory for calculating metrics
parser.add_argument('--eval_dir', type=str, default='expr/eval',
                    help='Directory for saving metrics, i.e., FID and LPIPS')

# directory for testing
parser.add_argument('--result_dir', type=str, default='expr/results',
                    help='Directory for saving generated images and videos')
parser.add_argument('--src_dir', type=str, default='assets/representative/celeba_hq/src',
                    help='Directory containing input source images')
parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref',
                    help='Directory containing input reference images')
parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
                    help='input directory when aligning faces')
parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
                    help='output directory when aligning faces')

# face alignment
parser.add_argument('--wing_path', type=str, default='starganv2/expr/checkpoints/wing.ckpt')
parser.add_argument('--lm_path', type=str, default='starganv2/expr/checkpoints/celeba_lm_mean.npz')

# step size
parser.add_argument('--print_every', type=int, default=10)
parser.add_argument('--sample_every', type=int, default=5000)
parser.add_argument('--save_every', type=int, default=10000)
parser.add_argument('--eval_every', type=int, default=50000)


# def main(args):
#     cudnn.benchmark = True
#     torch.manual_seed(args.seed)
#     solver = Solver(args)
    # dataloader = get_loader(imagefile,listfile,args.img_size)
    # solver.mysample(dataloader)

args = parser.parse_args()
cudnn.benchmark = True
# torch.manual_seed(args.seed)
solver = Solver(args)
def starganv2_Model():
    return solver.load_net()
def Processref_starganv2(net,ref_path,ref):
    return solver.get_myref(net,ref_path,ref)
def starganv2_Fake(img,ref,net):
    return solver.mysample2(img,ref,net)

def starganv2(img):
    args = parser.parse_args()
    cudnn.benchmark = True
    # torch.manual_seed(args.seed)
    solver = Solver(args)
    return solver.mysample(img)


def starganv22(img,ref):
    args = parser.parse_args()
    cudnn.benchmark = True
    # torch.manual_seed(args.seed)
    solver = Solver(args)
    return solver.mysample2(img,ref)

# args = parser.parse_args()
# main(args)
# starganv2(1)

