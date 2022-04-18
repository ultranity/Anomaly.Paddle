import os
import random
import argparse
import numpy as np
import datetime
from PIL import Image
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib
import matplotlib.pyplot as plt

import paddle

import datasets.mvtec as mvtec
from model import PaDiMPlus
from utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('picture_path', type=str)
    parser.add_argument('--save_path', type=str, default='./output')
    parser.add_argument('--model_path', type=str, default=None, help="specify model path if needed")
    parser.add_argument("--category", type=str, default='leather', help="category name for MvTec AD dataset")
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument("--arch", type=str, default='resnet18', help="backbone model arch, one of [resnet18, resnet50, wide_resnet50_2]")
    parser.add_argument("--k", type=int, default=100, help="feature used")
    parser.add_argument("--method", type=str, default='sample', help="projection method, one of [sample,ortho]")
    parser.add_argument("--save_pic", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--norm", type=str2bool, default=True)

    args, _ =  parser.parse_known_args()
    return args



def main():
    args = parse_args()
    args.save_path += f"/{args.method}_{args.arch}_{args.k}"
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)

    class_name = args.category
    assert class_name in mvtec.CLASS_NAMES
    print("Testing model for {}".format(class_name))
    # build model
    args.model_path = args.model_path or args.save_path + '/{}.pdparams'.format(class_name)
    model = PaDiMPlus(arch=args.arch, pretrained=False, fout=args.k, method=args.method)
    model.eval()
    state = paddle.load(args.model_path)
    model.model.set_dict(state["params"])
    model.projection = state["projection"]
    model.mean = state["mean"]
    model.inv_covariance = state["inv_covariance"]
    model.eval()
    
    # build data
    transform_x = mvtec.MVTecDataset.get_transform(cropsize=args.crop_size)[0]
    x = Image.open(args.picture_path).convert('RGB')
    x = transform_x(x).unsqueeze(0)
    predict(args, model, x)

def predict(args, model, x):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + "Starting eval model...")
    # extract test set features
    # model prediction
    out = model(x)
    out = model.project(out)
    score_map = model.generate_scores_map(out, x.shape[-2:])
    #score_map = np.concatenate(score_map, 0)
    
    # Normalization
    if args.norm:
        max_score = score_map.max()
        min_score = score_map.min()
        score_map = (score_map - min_score) / (max_score - min_score)
    save_name = os.path.join(args.save_path, args.category)
    dir_name = os.path.dirname(save_name)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    plot_fig(x.numpy(), score_map, args.threshold, save_name, args.category, args.save_pic)

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + "Predict :  Picture {}".format(
        args.picture_path) + " done!")
    if args.save_pic: print("Result saved at {}/{}_predict.png".format(save_name, args.category))



def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x

def plot_fig(test_img, scores, threshold, save_dir, class_name, save_pic=True):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(1):
        img = test_img[i]
        img = denormalization(img)
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax = ax_img[1].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[1].imshow(img, cmap='gray', interpolation='none')
        ax_img[1].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[1].title.set_text('Predicted heat map')
        ax_img[2].imshow(mask, cmap='gray')
        ax_img[2].title.set_text('Predicted mask')
        ax_img[3].imshow(vis_img)
        ax_img[3].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)
        if save_pic:
            fig_img.savefig(os.path.join(save_dir, '{}_predict'.format(class_name)), dpi=100)
        else:
            plt.show()
        plt.close()


if __name__ == '__main__':
    main()
