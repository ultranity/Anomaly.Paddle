import argparse
from re import T
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import time
import numpy as np
from skimage import measure
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

def cholesky_inverse(input, upper=False, out=None) :
    u = paddle.cholesky(input, upper)
    ui = paddle.linalg.triangular_solve(u, paddle.eye(u.shape[-1]), upper=upper)
    if len(u.shape)==2:
        uit = ui.T
    elif len(u.shape)==3:
        uit =paddle.transpose(ui, perm=(0, 2, 1))
    elif len(u.shape)==4:
        uit = paddle.transpose(ui, perm=(0, 1, 3, 2))
    out = ui@uit if upper else uit@ui
    return out

def mahalanobis(embedding, mean, inv_covariance):
    B,C,H,W = embedding.shape
    delta = (embedding - mean).reshape((B,C,H*W)).transpose((2, 0, 1))
    distances = ((delta @ inv_covariance) @ delta).sum(2).transpose((1, 0))
    distances = distances.reshape((B, H, W))
    distances = paddle.sqrt(distances)
    return distances

def mahalanobis_einsum(embedding, mean, inv_covariance):
    M = embedding - mean
    distances = paddle.einsum('nmhw,hwmk,nkhw->nhw', M, inv_covariance, M)
    distances = paddle.sqrt(distances)
    return distances

def svd_orthogonal(fin,fout, use_paddle=False):
    assert fin > fout, 'fin > fout'
    if use_paddle:
        X = paddle.rand((fout, fin))
        U, _, Vt = paddle.linalg.svd(X, full_matrices=False)
        #print(Vt.shape)
        #print(paddle.allclose(Vt@Vt.T, paddle.eye(Vt.shape[0])))
    else:
        X = np.random.random((fout, fin))
        U, _, Vt = np.linalg.svd(X, full_matrices=False)
        #print(Vt.shape)
        #print(np.allclose((Vt@ Vt.T), np.eye(Vt.shape[0])))
    W = paddle.to_tensor(Vt, dtype=paddle.float32).T
    return W

def orthogonal(rows,cols, gain=1):
    r"""return a (semi) orthogonal matrix, as
    described in `Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks` - Saxe, A. et al. (2013). 
    Args:
        rows: rows
        cols: cols
        gain: optional scaling factor
    Examples:
        >>> orthogonal_(5, 3)
    """
    flattened = paddle.randn((rows,cols))

    if rows < cols:
        flattened = flattened.T

    # Compute the qr factorization
    q, r = paddle.linalg.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = paddle.diag(r, 0)
    q *= d.sign()

    if rows < cols:
        q = q.T
    
    q *= gain
    return q


def compute_pro(binary_amaps:np.ndarray, masks:np.ndarray, method='mean') -> float:
    pros = []
    for binary_amap, mask in zip(binary_amaps, masks):
        per_region_tpr = []
        for region in measure.regionprops(measure.label(mask)):
            axes0_ids = region.coords[:, 0]
            axes1_ids = region.coords[:, 1]
            TP_pixels = binary_amap[axes0_ids, axes1_ids].sum()
            per_region_tpr.append(TP_pixels / region.area)
        if method=='mean' and per_region_tpr:
            pros.append(np.mean(per_region_tpr))
        else:
            pros.extend(per_region_tpr)
    return np.mean(pros)

def kthvalue(x:np.ndarray, k:int):
    return x[x.argpartition(k)[k]]

def get_thresholds(t:np.ndarray, num_samples=1000, reverse=False, opt=True):
    if opt:
        # use the worst-case for efficient determination of thresholds
        max_idx = t.reshape(t.shape[0], -1).max(1).argmax(0)
        t = t[max_idx].flatten()
        #return [kthvalue(t, max(1, math.floor(t.size * i / num_samples)-1)-1)
        #            for i in range(num_samples, 0, -1)]
        r = np.linspace(0, t.size-1, num=num_samples).astype(int)
        if reverse: r = r[::-1]
        t.sort()
        return t[r]
        #idx = np.argsort(t)
        #return [t[idx[max(1, math.floor(t.size * i / num_samples)-1)-1]] for i in range(num_samples, 0, -1)]
    else:
        #return [kthvalue(t.flatten(), max(1, math.floor(t.size * i / num_samples)))
        #            for i in range(num_samples, 0, -1)]
        
        r = np.linspace(t.min(), t.max(), num=num_samples)
        if reverse: r = r[::-1]
        return r

def compute_pro_score(amaps:np.ndarray, masks:np.ndarray) -> float:
    masks = masks.squeeze()
    
    pros = []
    fprs = []
    for th in (get_thresholds(amaps, 500, True, True)):#thresholds[::-1]:#
        binary_amaps = amaps.squeeze() > th
        """
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                TP_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pro.append(TP_pixels / region.area)
        pros.append(np.mean(pro))"""
        pros.append(compute_pro(binary_amaps, masks, 'mean'))
        
        inverse_masks = 1 - masks
        FP_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = FP_pixels / inverse_masks.sum()
        fprs.append(fpr)
        if fpr>0.3: break
        
    #print(np.array(list(zip(pros,fprs))))
    fprs = np.array(fprs)
    pros = np.array(pros)
    pro_auc_score = auc(rescale(fprs), rescale(pros)) # pros)#
    #thi = np.argmax([x[0] for x in df])
    #return df[thi]
    return pro_auc_score
