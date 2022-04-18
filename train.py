import os
import time
import random
import argparse
import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader

import datasets.mvtec as mvtec
from model import PaDiMPlus
from utils import str2bool
from eval import eval

#CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
#               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
#               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

textures = ['carpet', 'grid', 'leather', 'tile', 'wood']
objects = ['bottle','cable', 'capsule','hazelnut', 'metal_nut',
            'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
CLASS_NAMES = textures+objects
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='D:/dataset/mvtec_anomaly_detection')
    parser.add_argument('--save_path', type=str, default='./output')
    parser.add_argument("--category", type=str , default='tile', help="category name for MvTec AD dataset")
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--arch", type=str, default='resnet18', help="backbone model arch, one of [resnet18, resnet50, wide_resnet50_2]")
    parser.add_argument("--k", type=int, default=100, help="feature used")
    parser.add_argument("--method", type=str, default='sample',choices=['sample','h_sample', 'ortho', 'svd_ortho', 'gaussian'], help="projection method, one of [sample, ortho, svd_ortho, gaussian]")
    parser.add_argument("--save_model", type=str2bool, default=True)
    parser.add_argument("--save_pic", type=str2bool, default=True)
    parser.add_argument("--inc",  action='store_true', help="use incremental cov & mean")
    parser.add_argument("--eval", action='store_true')
    parser.add_argument('--eval_PRO', action='store_true')
    parser.add_argument('--einsum', action='store_true')
    parser.add_argument('--cpu', action='store_true', help="use cpu device")
    parser.add_argument("--save_model_subfolder", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=521)
    
    args, _ =  parser.parse_known_args()
    return args


def main():

    args = parse_args()
    if args.save_model_subfolder: args.save_path += f"/{args.method}_{args.arch}_{args.k}"
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    if args.cpu: paddle.device.set_device("cpu")
    # build model
    model = PaDiMPlus(arch=args.arch, pretrained=True, fout=args.k, method=args.method)
    model.init_projection()
    model.eval()
    #print(model.projection)
    result = []
    if args.category == 'all':
        class_names = mvtec.CLASS_NAMES 
    elif args.category == 'textures':
        class_names = mvtec.textures 
    elif args.category == 'objects':
        class_names = mvtec.objects 
    else:
        class_names = [args.category]
    csv_columns = ['category','Image_AUROC','Pixel_AUROC', 'PRO_score']
    csv_name = os.path.join(args.save_path, '{}_seed{}.csv'.format(args.category, args.seed))
    for i,class_name in enumerate(class_names):
        print("Training model {}/{} for {}".format(i+1, len(class_names), class_name))
        # build datasets
        train_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=True, cropsize=args.crop_size)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
        train(args, model, train_dataloader, class_name)
        if args.eval:
            test_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=False, cropsize=args.crop_size)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
            result.append([class_name, *eval(args, model, test_dataloader, class_name)])
            if args.category == 'all':
                pd.DataFrame(result, columns=csv_columns).set_index('category').to_csv(csv_name)
    if args.eval:
        result = pd.DataFrame(result, columns=csv_columns).set_index('category')
        if not args.eval_PRO: del result['PRO_score']
        print("Evaluation result saved at{}:".format(csv_name))
        print(result)
        result.to_csv(csv_name)
        if args.category == 'all':
            print("=========Mean Performance========")
            print(result.mean(numeric_only=True))

def train(args, model, train_dataloader, class_name):
    epoch_begin = time.time()
    #paddle.device.set_device("gpu")
    # extract train set features
    
    if args.inc:
        c = model.k #args.k
        h = w = args.crop_size//4
        X_cov = paddle.zeros([h, w, c, c])  # covariance
        X_mean = paddle.zeros([c, h, w])  # mean
        N = 0 # sample num
        for (x,_) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
            # model prediction
            out = model(x)
            out = model.project(out)
            X_mean += out.sum(0)
            X_cov += paddle.einsum('bchw, bdhw -> hwcd', out, out)
            N += x.shape[0]
        del out, x
        model.compute_inv_incremental(X_mean, X_cov, N)
    else:
        outs = []
        for (x,_) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
            # model prediction
            out = model(x)
            out = model.project(out)
            outs.append(out)
        del out, x
        outs = paddle.concat(outs, 0)
        #paddle.device.set_device("cpu")
        if args.einsum:
            model.compute_distribution_einsum(outs)
        else:
            model.compute_distribution(outs)
    
    t = time.time() - epoch_begin
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
          "Train ends, total {:.2f}s".format(t))
    if args.save_model:
        print("Saving model...")
        save_name = os.path.join(args.save_path, '{}.pdparams'.format(class_name))
        dir_name = os.path.dirname(save_name)
        os.makedirs(dir_name, exist_ok=True)
        state_dict = {
            "params":model.model.state_dict(),
            "mean":model.mean,
            "inv_covariance":model.inv_covariance,
            "projection":model.projection,
        }
        paddle.save(state_dict, save_name)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + "Save model in {}".format(str(save_name)))

if __name__ == '__main__':
    main()