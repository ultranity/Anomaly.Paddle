# Anomaly.Paddle

##  简介

基于 Paddle 的异常检测方法复现


**[Logs&Tricks](Logs&Tricks.md)**

**参考repo:** 
- [PaDiM-Anomaly-Detection-Localization-master](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master)
- [PaDiM-Paddle](https://github.com/CuberrChen/PaDiM-Paddle)
- [anomalib](https://github.com/openvinotoolkit/anomalib)

### 复现论文 |Including paper:<br>

- [X] [PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization](https://arxiv.org/abs/2011.08785)
- [x] [Semi-orthogonal Embedding for Efficient Unsupervised Anomaly Segmentation](https://arxiv.org/abs/2105.14737)
- [x] [PatchCore: Towards Total Recall in Industrial Anomaly Detection ](https://arxiv.org/abs/2106.08265)
- [ ] [Modeling the Distribution of Normal Data in Pre-Trained Deep Features for Anomaly Detection](https://arxiv.org/abs/2005.14140)
and so on...

## 数据集|Datasets:

- [x] [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [ ] [KolektorSDD]()
- [ ] [mSTC]()

## 评估指标|Metrics:

- [x] Image AUROC
- [x] Pixel AUROC
- [x] PRO score (Per Region Overlap Score)

## 快速开始

### 第一步：克隆本项目
```bash
# clone this repo
git clone git@github.com/ultranity/Anomaly.Paddle.git
cd Anomaly.Paddle
```

### 第二步：安装依赖
```bash
pip install -r requirements.txt
```

### 后续使用：训练/验证/导出模型，TIPC测试等

见各算法单独文档
- [PaDiM](PaDiM.md)
- [OrthoAD](OrthoAD.md)
- [PatchCore](PatchCore.md)

## 代码结构

```
Anomaly.Paddle
├── datasets #数据集定义
├── logs #复现日志
├── output #输出目录
├── test_tpic # TPIC自动化测试
├── eval.py # 模型验证
├── export_model.py #模型导出
├── infer.py # 模型推理脚本
├── net.py # resnet网络结构补充（wide_resnet50_2）
├── OrthoAD.md # OrthoAD说明文件
├── PaDiM.md # PaDiM说明文件
├── predict.py # 模型单独预测
├── README.md # 主说明文件
├── train.py # 模型训练
├── utils.py # 工具函数
└── requirements.txt #库依赖
```

## 说明

在此非常感谢`CuberrChen`贡献的[PaDiM-Paddle](https://github.com/CuberrChen/PaDiM-Paddle)项目，提高了本repo复现论文的效率。

感谢百度 AIStudio 提供的算力支持

##  LICENSE

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。

##  参考链接与文献

- Simonjan, Jennifer  and  Unluturk, Bige D.  and  Akyildiz, Ian F. [PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization](https://arxiv.org/pdf/2011.08785)
- Jin-Hwa Kim, Do-Hyeong Kim, Saehoon Yi, Taehoon Lee. [Semi-orthogonal Embedding for Efficient Unsupervised Anomaly Segmentation](https://arxiv.org/abs/2105.14737)
- [Modeling the Distribution of Normal Data in Pre-Trained Deep Features for Anomaly Detection](https://arxiv.org/abs/2005.14140)
- [anomalib](https://github.com/openvinotoolkit/anomalib)
- [PaDiM-Anomaly-Detection-Localization-master](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master)
- [Semi-Orthogonal Embedding for Efficient Unsupervised Anomaly Segmentation(github.com)](https://github.com/jnhwkim/orthoad)
- [PaDiM-Paddle](https://github.com/CuberrChen/PaDiM-Paddle)
- [Knowledge_Distillation_AD_Paddle](https://github.com/txyugood/Knowledge_Distillation_AD_Paddle)
- [DFR](https://github.com/YoungGod/DFR)
