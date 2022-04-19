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
from model import PaDiMPlus
from utils import compute_pro_score, plot_fig, str2bool

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
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--arch", type=str, default='resnet18', help="backbone model arch, one of [resnet18, resnet50, wide_resnet50_2]")
    parser.add_argument("--k", type=int, default=100, help="feature used")
    parser.add_argument("--method", type=str, default='sample', help="projection method, one of [sample,ortho]")
    parser.add_argument("--save_pic", type=str2bool, default=True)
    parser.add_argument('--eval_PRO', action='store_true')
    parser.add_argument("--seed", type=int, default=521)
    
    args, _ =  parser.parse_known_args()
    return args


def main():

    args = parse_args()
    args.save_path += f"/{args.method}_{args.arch}_{args.k}"
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)

    result = []
    assert (args.category in mvtec.CLASS_NAMES) or (args.category == 'all')
    class_names = mvtec.CLASS_NAMES if args.category == 'all' else [args.category]
    csv_columns = ['category','Image_AUROC','Pixel_AUROC', 'PRO_score']
    csv_name = os.path.join(args.save_path, '{}_seed{}.csv'.format(args.category, args.seed))
    for i,class_name in enumerate(class_names):
        print("Testing model {}/{} for {}".format(i, len(class_names), class_name))
        
        # build model
        model_path = args.model_path or args.save_path + '/{}.pdparams'.format(class_name)
        model = PaDiMPlus(arch=args.arch, pretrained=False, fout=args.k, method= args.method)
        state = paddle.load(model_path)
        model.model.set_dict(state["params"])
        model.projection = state["projection"]
        model.mean = state["mean"]
        model.inv_covariance = state["inv_covariance"]
        model.eval()
        #model.compute_inv(state["stats"])
        
        # build datasets
        test_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=False, cropsize=args.crop_size)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
        result.append([class_name, *eval(args, model, test_dataloader, class_name)])
        if args.category == 'all':
            pd.DataFrame(result, columns=csv_columns).set_index('category').to_csv(csv_name)
    result = pd.DataFrame(result, columns=csv_columns).set_index('category')
    if not args.eval_PRO: result = result.drop(columns="PRO_score")
    print("Evaluation result saved at{}:".format(csv_name))
    print(result)
    result.to_csv(csv_name)
    if args.category == 'all':
        print("=========Mean Performance========")
        print(result.mean(numeric_only=True))

def eval(args, model, test_dataloader, class_name):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' + "Starting eval model...")
    total_auroc = []
    total_pixel_auroc = []

    gt_list = []
    gt_mask_list = []
    test_imgs = []
    score_map = []
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
    score_map = np.concatenate(score_map, 0)
    
    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)

    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    #fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_auroc = roc_auc_score(gt_list, img_scores)
    total_auroc.append(img_auroc)
    total_auroc = np.mean(total_auroc)
    
    # get optimal threshold
    gt_mask = np.asarray(gt_mask_list, dtype=np.int64).squeeze()
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    # calculate per-pixel level ROCAUC
    #fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_auroc = roc_auc_score(gt_mask.flatten(), scores.flatten())
    total_pixel_auroc.append(per_pixel_auroc)
    total_pixel_auroc = np.mean(total_pixel_auroc)
    
    # calculate Per-Region-Overlap Score
    total_PRO = compute_pro_score(scores, gt_mask) if args.eval_PRO else None

    print([class_name, total_auroc, total_pixel_auroc, total_PRO])
    if args.save_pic:
        save_dir = os.path.join(args.save_path, class_name)
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)
    return total_auroc, total_pixel_auroc, total_PRO

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
