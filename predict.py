import os
import random
import argparse
import numpy as np
import datetime
from PIL import Image

import paddle

import datasets.mvtec as mvtec
from model import PaDiMPlus
from utils import plot_fig, str2bool


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
    plot_fig(x.numpy(), score_map, None, args.threshold, save_name, args.category, args.save_pic, 'predict')

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + "Predict :  Picture {}".format(
        args.picture_path) + " done!")
    if args.save_pic: print("Result saved at {}/{}_predict.png".format(save_name, args.category))

if __name__ == '__main__':
    main()
