import os
import random
import argparse
import datetime
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

import paddle
from paddle.io import DataLoader

import datasets.mvtec as mvtec
from model import get_model
from utils import compute_pro_score, compute_roc_score, plot_fig, str2bool

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
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument("--category", type=str , default='tile', help="category name for MvTec AD dataset")
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument("--arch", type=str, default='resnet18', help="backbone model arch, one of [resnet18, resnet50, wide_resnet50_2]")
    parser.add_argument("--k", type=int, default=100, help="feature used")
    parser.add_argument("--method", type=str, default='sample', help="projection method, one of ['sample','h_sample', 'ortho', 'svd_ortho', 'gaussian']")
    parser.add_argument("--save_pic", type=str2bool, default=True)
    parser.add_argument('--eval_PRO', action='store_true')
    parser.add_argument('--non_partial_AUC', action='store_true')
    parser.add_argument('--eval_threthold_step', type=int, default=500, help="threthold_step when computing PRO Score and non_partial_AUC")
    parser.add_argument("--seed", type=int, default=521)
    
    args, _ =  parser.parse_known_args()
    return args

@paddle.no_grad()
def main():

    args = parse_args()
    args.save_path += f"/{args.method}_{args.arch}_{args.k}"
    #if args.method =='coreset': args.test_batch_size=1
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)

    result = []
    assert args.category in mvtec.CLASS_NAMES + ['all', 'textures', 'objects']
    class_names = mvtec.CLASS_NAMES if args.category == 'all' else [args.category]
    csv_columns = ['category','Image_AUROC','Pixel_AUROC', 'PRO_score']
    csv_name = os.path.join(args.save_path, '{}_seed{}.csv'.format(args.category, args.seed))
    for i,class_name in enumerate(class_names):
        print("Testing model {}/{} for {}".format(i+1, len(class_names), class_name))
        
        # build model
        model_path = args.model_path or args.save_path + '/{}.pdparams'.format(class_name)
        model = get_model(args.method)(arch=args.arch, pretrained=False, k=args.k, method= args.method)
        state = paddle.load(model_path)
        model.model.set_dict(state["params"])
        model.load(state["stats"])
        model.eval()
        #model.compute_inv(state["stats"])
        
        # build datasets
        test_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=False, resize=args.resize, cropsize=args.crop_size)
        test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers)
        result.append([class_name, *eval(args, model, test_dataloader, class_name)])
        if args.category in ['all', 'textures', 'objects']:
            pd.DataFrame(result, columns=csv_columns).set_index('category').to_csv(csv_name)
    result = pd.DataFrame(result, columns=csv_columns).set_index('category')
    if not args.eval_PRO: result = result.drop(columns="PRO_score")
    if args.category in ['all', 'textures', 'objects']:
        result.loc['mean'] = result.mean(numeric_only=True)
    print(result)
    print("Evaluation result saved at{}:".format(csv_name))
    result.to_csv(csv_name)

@paddle.no_grad()
def eval(args, model, test_dataloader, class_name):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + "Starting eval model...")

    gt_list = []
    gt_mask_list = []
    test_imgs = []
    score_map = []
    #paddle.device.set_device("gpu")
    # extract test set features
    for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):

        test_imgs.extend(x.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        gt_mask_list.extend(mask.cpu().detach().numpy())
        # model prediction
        out = model(x)
        out = model.project(out)
        score_map.append(model.generate_scores_map(out, x.shape[-2:]))
    del out
    score_map, image_score = list(zip(*score_map))
    score_map = np.concatenate(score_map, 0)
    image_score = np.concatenate(image_score, 0)
    
    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    score_map = (score_map - min_score) / (max_score - min_score)
    print(f"max_score:{max_score} min_score:{min_score}")
    # calculate image-level ROC AUC score
    gt_list = np.asarray(gt_list)
    #fpr, tpr, _ = roc_curve(gt_list, image_score)
    img_auroc = compute_roc_score(gt_list, image_score, args.eval_threthold_step, args.non_partial_AUC)
    # get optimal threshold
    precision, recall, thresholds = precision_recall_curve(gt_list, image_score)
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    print(f"F1 image:{f1.max()} threshold:{max_score}")
    # calculate per-pixel level ROCAUC
    gt_mask = np.asarray(gt_mask_list, dtype=np.int64).squeeze()
    #fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_auroc =  compute_roc_score(gt_mask.flatten(), score_map.flatten(), args.eval_threthold_step, args.non_partial_AUC)
    # get optimal threshold
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), score_map.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    print(f"F1 pixel:{f1.max()} threshold:{max_score}")
    
    # calculate Per-Region-Overlap Score
    total_PRO = compute_pro_score(gt_mask, score_map, args.eval_threthold_step, args.non_partial_AUC) if args.eval_PRO else None

    print([class_name, img_auroc, per_pixel_auroc, total_PRO])
    if args.save_pic:
        save_dir = os.path.join(args.save_path, class_name)
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(test_imgs, score_map, gt_mask_list, threshold, save_dir, class_name)
    return img_auroc, per_pixel_auroc, total_PRO

def plot_roc(fpr, tpr, score, save_dir, class_name, tag='pixel'):
    plt.plot(fpr, tpr, marker="o", color="k", label=f"AUROC Score: {score:.3f}")
    plt.xlabel("FPR: FP / (TN + FP)", fontsize=14)
    plt.ylabel("TPR: TP / (TP + FN)", fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{class_name}_{tag}_roc_curve.png")
    plt.close()

def plot_roc_all(fprs, tprs, scores, class_names, save_dir, tag='pixel'):
    plt.figure()
    for fpr,tpr,score,class_name in zip(fprs, tprs, scores,class_names):
        plt.plot(fpr, tpr, marker="o", color="k", label=f"{class_name} AUROC: {score:.3f}")
        plt.xlabel("FPR: FP / (TN + FP)", fontsize=14)
        plt.ylabel("TPR: TP / (TP + FN)", fontsize=14)
        plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{tag}_roc_curve.png")
    plt.close()


if __name__ == '__main__':
    main()
