# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models.resnet import resnet18, resnet50
from net import wide_resnet50_2
from scipy.ndimage import gaussian_filter
from utils import cholesky_inverse, mahalanobis, mahalanobis_einsum, orthogonal, svd_orthogonal
from tqdm import tqdm
models = {"resnet18":resnet18,"resnet50":resnet50,
          #"resnet18_vd":resnet18_vd,
          "wide_resnet50_2":wide_resnet50_2,}
fins = {"resnet18":448,"resnet50":1792,
          "resnet18_vd":448,"wide_resnet50_2":1792,}

def get_projection(fin, fout, method='ortho'):
    if 'sample' == method:
        s = paddle.randperm(fin)[:fout].tolist()
        W = paddle.eye(fin)[s].T
    elif 'h_sample' == method:
        s = paddle.randperm(fin//7)[:fout//3].tolist()\
                +(fin//7+paddle.randperm(fin//7*2)[:fout//3]).tolist()\
                +(fin//7*3+paddle.randperm(fin//7*4)[:(fout-fout//3*2)]).tolist()
        W = paddle.eye(fin)[s].T
    elif 'ortho' == method:
        W = orthogonal(fin, fout)
    elif 'svd_ortho' == method:
        W = svd_orthogonal(fin, fout)
    elif 'gaussian' == method:
        W = paddle.randn(fin, fout)
    return W

class PaDiMPlus(nn.Layer):
    def __init__(self, arch='resnet18', pretrained=True, fout=100, method = 'sample'):
        super(PaDiMPlus, self).__init__()
        assert arch in models.keys(), 'arch {} not supported'.format(arch)
        
        self.model = models[arch](pretrained)
        del self.model.layer4, self.model.fc , self.model.avgpool
        self.model.eval()
        print(f'model {arch}, nParams {sum([w.size for w in self.model.parameters()])}')
        self.arch = arch
        self.method = method
        self.fin = fins[arch]
        self.k = fout
        self.projection = None
        self.mean = None
        self.inv_covariance = None
        #self.cov = None
        
    def init_projection(self):
        self.projection = get_projection(fins[self.arch], self.k, self.method)
    
    def clean_stats(self):
        self.mean = None
        self.inv_covariance = None
    
    def set_dist_params(self, mean, inv_cov):
        self.mean, self.inv_covariance = mean, inv_cov

    @paddle.no_grad()
    def project_einsum(self, x):
        return paddle.einsum('bchw, cd -> bdhw', x, self.projection)
        #if self.method == 'ortho':
        #    return paddle.einsum('bchw, cd -> bdhw', x, self.projection)
        #else: #self.method == 'PaDiM':
        #    return paddle.index_select(embedding,  self.projection, 1)
    
    @paddle.no_grad()
    def project(self, x):
        B, C, H, W = x.shape
        x = x.reshape((B, C, H*W))
        result = paddle.zeros((B, self.k, H, W))
        for i in range(B):
            #result[i] = paddle.einsum('chw, cd -> dhw', x[i], self.projection)
            result[i] = (self.projection.T @ x[i]).reshape((self.k, H, W))
        return result
    
    @paddle.no_grad()
    def forward_res(self, x):
        res = []
        with paddle.no_grad():
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            res.append(x)
            x = self.model.layer2(x)
            res.append(x)
            x = self.model.layer3(x)
            res.append(x)
        return res
    
    @paddle.no_grad()
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = [self.model.layer1(x).detach()]
        x.append(self.model.layer2(x[-1]).detach())
        x.append(self.model.layer3(x[-1]).detach())
        x[-2]=F.interpolate(x[-2], size=x[0].shape[-2:], mode="nearest")
        x[-1]=F.interpolate(x[-1], size=x[0].shape[-2:], mode="nearest")
        #print([i.shape for i in x])
        x = paddle.concat(x, 1)
        #x = self.project(x)
        return x

    @paddle.no_grad()
    def forward_score(self, x):
        return self.generate_scores_map(self.get_embedding(x), x.shape)
    
    @paddle.no_grad()
    def compute_distribution_einsum(self, outs):
        # calculate multivariate Gaussian distribution
        B, C, H, W = outs.shape
        mean = outs.mean(0)  # mean chw
        outs-= mean
        cov = paddle.einsum('bchw, bdhw -> hwcd', outs, outs)/(B-1) # covariance hwcc
        self.compute_inv(mean, cov)
    
    @paddle.no_grad()
    def compute_distribution_(self, embedding):
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding.shape
        mean = paddle.mean(embedding, axis=0)
        embedding = embedding.reshape((B, C, H * W))
        cov = np.empty((C, C, H * W))
        for i in range(H * W):
            cov[:, :, i] = np.cov(embedding[:, :, i].numpy(), rowvar=False)
        cov = paddle.to_tensor(cov.reshape(C,C,H,W).transpose((2,3, 0, 1)))
        self.compute_inv(mean, cov)
    
    @paddle.no_grad()
    def compute_distribution_np(self, embedding):
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding.shape
        mean = paddle.mean(embedding, axis=0)
        embedding = embedding.reshape((B, C, H * W)).numpy()
        inv_covariance = np.empty((H * W, C, C), dtype='float32')
        I = np.identity(C)
        for i in tqdm(range(H * W)):
            inv_covariance[i, :, :] = np.linalg.inv(np.cov(embedding[:, :, i], rowvar=False)  + 0.01 * I)
        inv_covariance = paddle.to_tensor(inv_covariance.reshape(H,W,C,C)).astype('float32')
        self.set_dist_params(mean, inv_covariance)
    
    @paddle.no_grad()
    def compute_distribution(self, embedding):
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding.shape
        mean = paddle.mean(embedding, axis=0)
        embedding -= mean
        embedding = embedding.transpose((2, 3, 0, 1)) #hwbc
        inv_covariance = paddle.zeros((H, W, C, C), dtype='float32')
        I = paddle.eye(C)
        for i in (range(H)):
            inv_covariance[i, :, :, :] = paddle.einsum('wbc, wbd -> wcd',embedding[i],embedding[i])/(B-1) + 0.01*I
            inv_covariance[i, :, :, :] = cholesky_inverse(inv_covariance[i, :, :, :])
        inv_covariance = paddle.to_tensor(inv_covariance.reshape((H,W,C,C))).astype('float32')
        self.set_dist_params(mean, inv_covariance)
    
    @paddle.no_grad()
    def compute_distribution_incremental(self, outs):
        # calculate multivariate Gaussian distribution
        B, C, H, W = outs.shape
        mean = outs.sum(0)  # mean chw
        cov = paddle.einsum('bchw, bdhw -> hwcd', outs, outs)# covariance hwcc
        return mean, cov, B
    
    def compute_inv_incremental(self,mean, covariance, B, eps=0.01):
        c = mean.shape[0]
        #if self.inv_covariance == None:
        mean = mean/B # chw
        #covariance hwcc  #.transpose((2,3, 0, 1)))
        covariance = (covariance - B*paddle.einsum('chw, dhw -> hwcd', mean, mean))/(B-1) # covariance hwcc
        inv_covariance = cholesky_inverse(covariance + eps * paddle.eye(c))
        #self.inv_covariance = paddle.linalg.inv(covariance)
        self.set_dist_params(mean, inv_covariance)
    
    def compute_inv(self,mean, covariance, eps=0.01):
        c = mean.shape[0]
        #if self.inv_covariance == None:
        #covariance hwcc  #.transpose((2,3, 0, 1)))
        #self.inv_covariance = paddle.linalg.inv(covariance)
        self.set_dist_params(mean, cholesky_inverse(covariance + eps * paddle.eye(c)))
    
    def generate_scores_map(self, embedding, out_shape, gaussian_blur = True):
        # calculate distance matrix
        B, C, H, W = embedding.shape
        #embedding = embedding.reshape((B, C, H * W))
        # calculate mahalanobis distances
        
        distances = mahalanobis_einsum(embedding, self.mean, self.inv_covariance)
        score_map = F.interpolate(distances.unsqueeze(1), size=out_shape, mode='bilinear',
                                align_corners=False).squeeze(1).numpy()

        if gaussian_blur:
            # apply gaussian smoothing on the score map
            for i in range(score_map.shape[0]):
                score_map[i] = gaussian_filter(score_map[i], sigma=4)

        return score_map


def generate_scores_map(mean, inv_covariance, embedding, out_shape, gaussian_blur = True):
    # calculate distance matrix
    B, C, H, W = embedding.shape
    #embedding = embedding.reshape((B, C, H * W))
    # calculate mahalanobis distances
    
    distances = mahalanobis_einsum(embedding, mean, inv_covariance)
    score_map = F.interpolate(distances.unsqueeze(1), size=out_shape, mode='bilinear',
                            align_corners=False).squeeze(1).numpy()

    if gaussian_blur:
        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

    return score_map

if __name__ == '__main__':
    model = PaDiMPlus()
    print(model)
