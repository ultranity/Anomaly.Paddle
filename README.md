# Anomaly.Paddle

## 简介
基于 Paddle 的异常检测方法复现
参考: [anomalib](https://github.com/openvinotoolkit/anomalib)


目前已实现|Including paper:<br>
[x] [PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization](https://arxiv.org/abs/2011.08785)
[x] [Semi-orthogonal Embedding for Efficient Unsupervised Anomaly Segmentation](https://arxiv.org/abs/2105.14737)
[ ] [Modeling the Distribution of Normal Data in Pre-Trained Deep Features for Anomaly Detection](https://arxiv.org/abs/2005.14140)
[ ] [PatchCore: ]
and so on...

数据集|Datasets:
[x] [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
[ ] [KolektorSDD]()
[ ] [mSTC]()

评估指标|Metrics:
[x] Image AUROC
[x] Pixel AUROC
[x] PRO score (Per Region Overlap Score)

## 快速开始

### 第一步：克隆本项目
```bash
# clone this repo
git clone git@github.com/ultranity/Anomaly.Paddle.git
cd Anomaly.Paddle
```

### 第二步：训练/验证模型

见各算法单独文档
[PaDiM](PaDiM.md)

[OrthoAD](OrthoAD.md)

更多日志及预训练模型见 [AIStudio notebook](https://aistudio.baidu.com/aistudio/projectdetail/3824965)

## 代码结构与说明
**代码结构**
```
├── data
├── datasets
├── logs
├── output
├── test_tpic
├── PaDiM.md
├── OrthoAD.md
├── README.md
├── eval.py
├── train.py
├── predict.py
├── utils.py
├── net.py
├── export_model.py
├── infer.py
└── requirements.txt
```

## 模型信息

相关信息:

| 信息 | 描述 |
| --- | --- |
| 作者 | ultranity|
| 框架 | PaddlePaddle==2.2.1 |
| 应用场景 | 异常检测 |
| 硬件支持 | GPU、CPU |

## 说明

- 感谢百度提供的算力支持。