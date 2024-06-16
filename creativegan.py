import argparse
import re

from utils import zdataset, show, labwidget, renormalize
from rewrite import ganrewrite, rewriteapp
import torch, copy, os, json, shutil
from torchvision.utils import save_image
from torchvision import transforms
import utils.stylegan2, utils.proggan
from utils.stylegan2 import load_seq_stylegan

import numpy as np
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm, trange
import cv2

from utils import unet, anomaly

from pytorch_msssim import ssim, ms_ssim

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True, help='Name of experiment')
parser.add_argument('--model_path', type=str, required=True, help='Path to Stylegan model')
parser.add_argument('--model_size', type=int, default=512, help='GAN model output size')
parser.add_argument('--truncation', type=float, default=0.5, help="Value for truncation trick in Stylegan")

parser.add_argument('--seg_model_path', type=str, required=True, help="Path to segmentation model")
parser.add_argument('--seg_total_class', type=int, default=7, help="Total class/channel in segmentation model")
parser.add_argument('--seg_channels', type=_parse_num_range, required=True, help="List of segmentation channel that will be considered for rewriting")

parser.add_argument('--data_path', type=str, required=True, help='Path to dataset, the folder should directly contain the images')
parser.add_argument('--k', type=int, default=50, help='topk value for anomaly detection')
parser.add_argument('--anomaly_threshold', type=int, default=3.5, help='Threshold for novelty segmentation')

parser.add_argument('--copy_id', type=int, required=True, help='Seed id for target copy')
parser.add_argument('--paste_id', type=int, required=True, help='Seed id for target paste')
parser.add_argument('--context_ids', type=_parse_num_range, help='List of context ids', required=True)
parser.add_argument('--layernum', type=int, required=True, help='layer to be edited')
parser.add_argument('--rank', type=int, default=30, help='rank used in rewriting')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate in rewriting')
parser.add_argument('--niter', type=float, default=2000, help='number of iterations in rewriting')

parser.add_argument('--n_outputs', type=int, default=9, help='Number of outputs to display')

parser.add_argument('--ssim', action='store_true', help="calculate ssim of modified model")
parser.add_argument('--novelty_score', action='store_true', help="calculate average novelty score of modified model")


args = parser.parse_args()

model_path = args.model_path
model_size = args.model_size
truncation = args.truncation

name=args.name

seg_model_path = args.seg_model_path

data_path = args.data_path
k=args.k
anomaly_threshold = args.anomaly_threshold
n_outputs = args.n_outputs

# Copy id for frame example shown in paper
# 907, 728, 348, 960

# Copy id for handle example shown in paper
# 580, 811, 576

copy_id=args.copy_id
paste_id=args.paste_id
key_ids=args.context_ids

seg_class = args.seg_total_class
channels=args.seg_channels
# 0 - frame
# 1 - saddle
# 2 - wheel
# 3 - handle
# eg. [0, 3] - only frame or handle will be used for rewriting

layer=args.layernum
rank=args.rank
lr=args.lr
niter=args.niter

use_copy_as_paste_mask = False
dilate_mask= True
dilate_kernel_size=(16,16)

def dilate(mask,kernel_size=(8,8)):
    kernel = np.ones(kernel_size, np.uint8)
    mask = cv2.dilate(mask, kernel)
    return mask

def segment(seg_model, images, ch=3, size=(224,224), threshold=0.5):
    trans = transforms.Compose([
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                              ])
    images_tensor = torch.empty((len(images), ch, size[0], size[1]))
    for i in range(len(images)):
        images_tensor[i] = trans(images[i])
    
    # Move tensors to CPU
    seg_model = seg_model.to('cpu')
    images_tensor = images_tensor.to('cpu')

    seg_masks = seg_model(images_tensor).sigmoid().detach().cpu()
    seg_masks = torch.where(seg_masks > threshold, torch.ones(seg_masks.size()), torch.zeros(seg_masks.size()))
    return seg_masks

def masks_to_colorimg(masks):
    colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228),(56, 34, 132), (160, 194, 56), (155, 89, 182)])

    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)



def find_best_seg_match(mask, seg_mask, channels=None):
    scores = []
    if channels is None:
        channels = list(range(seg_mask.shape[0]))
    for i in range(seg_mask.shape[0]):
        if i not in channels:
            scores.append(-1)
            continue
        
        iou_score = jaccard_score(mask.reshape(-1), seg_mask[i].reshape(-1))
        scores.append(iou_score)
    best_ch = np.argmax(scores)
    return best_ch

def render_mask(tup, gw, size=512):
    imgnum, mask = tup
    area = (renormalize.from_url(mask, target='pt', size=(size,size))[0] > 0.25)
    return gw.render_image(imgnum, mask=area)

def show_masks(masks, gw):
    n = len(masks)
    if n == 1:
        masks = masks[0]
    if type(masks) is tuple:
        plt.imshow(render_mask(masks, gw))
        return

    fig, axes = plt.subplots(1, n, figsize=(n*3, 3))
    for i in range(n):
        axes[i].imshow(render_mask(masks[i], gw))

if __name__ == '__main__':
    # Choices: ganname = 'stylegan' or ganname = 'proggan'
    ganname = 'stylegan'

    modelname = name

    layernum = layer

    # Number of images to sample when gathering statistics.
    size = 10000

    # Make a directory for caching some data.
    layerscheme = 'default'
    expdir = 'results/pgw/%s/%s/%s/layer%d' % (ganname, modelname, layerscheme, layernum)
    os.makedirs(expdir, exist_ok=True)

    # Load (and download) a pretrained GAN
    if ganname == 'stylegan':
        model = load_seq_stylegan(model_path, path=True, size=model_size, mconv='seq', truncation=truncation)
        Rewriter = ganrewrite.SeqStyleGanRewriter
    elif ganname == 'proggan':
        model = utils.proggan.load_pretrained(modelname)
        Rewriter = ganrewrite.ProgressiveGanRewriter

    # Move model to CPU
    model = model.to('cpu')
        
    # Create a Rewriter object - this implements our method.
    zds = zdataset.z_dataset_for_model(model, size=size)
    gw = Rewriter(
        model, zds, layernum, cachedir=expdir,
        low_rank_insert=True, low_rank_gradient=False,
        use_linear_insert=False,  # If you set this to True, increase learning rate.e
        key_method='zca')

    # Display a user interface to allow model rewriting.
    savedir = f'masks/{ganname}/{modelname}'
    interface = rewriteapp.GanRewriteApp(gw, size=256, mask_dir=savedir, num
