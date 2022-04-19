# PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization

## 1. 简介

本项目基于PaddlePaddle框架复现了PaDiM算法，并在MvTec数据集上进行了实验。
![arch](assets/PaDiM_arch.png)

PaDiM是一种基于图像Patch的无监督异常检测算法，在训练过程中只是用正常样本进行建模。PaDiM算法首先使用resnet等预训练模型作为特征提取器，提取图像的向量表示，为提升模型效果，采用拼接的方式得到包含不同语义层次和分辨率信息的嵌入向量，然而由于生成的嵌入向量可能携带冗余信息，实验表明利用随机抽样降维的方法就能保持较高精度同时极大提升模型速度。
在训练过程中，对每个patch的嵌入向量表示使用多元高斯分布建模，统计其均值和协方差矩阵作为模型参数并保存。
在推理过程中，使用马氏距离对测试图像的每个面片位置进行评分，马氏距离矩阵上采样至原图尺寸并高斯滤波后生成异常图，使用阈值分割异常区域。
由于马氏距离计算中需要用到精度矩阵（也即协方差矩阵的逆），实际训练过程保存的是精度矩阵以减少后续冗余运算。
考虑到协方差矩阵是非负定实对称矩阵，求逆过程可以使用cholesky分解加速。
相关优化详见 [Logs&Tricks](Logs&Tricks.md)

**论文:** [PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization](https://arxiv.org/pdf/2011.08785)

**参考repo:**

- [anomalib](https://github.com/openvinotoolkit/anomalib)
- [PaDiM-Anomaly-Detection-Localization-master](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master)
- [PaDiM-Paddle](https://github.com/CuberrChen/PaDiM-Paddle)
在此非常感谢`xiahaifeng1995`贡献的[PaDiM-Anomaly-Detection-Localization-master](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master)，以及`CuberrChen`贡献的[PaDiM-Paddle](https://github.com/CuberrChen/PaDiM-Paddle)项目，提高了本repo复现论文的效率。

感谢百度 AIStudio 提供的算力支持。

**aistudio体验教程:** [地址](https://aistudio.baidu.com/aistudio/projectdetail/3824965)

## 2. 复现精度

- [MvTec数据集说明](#3.2 准备数据)

按复现考核标准，使用resnet18 在[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)数据集的测试效果如下表。

### Image-Level AUC

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| anomalib      | 0.891 | 0.945  | 0.857 |  0.982  | 0.950 | 0.976 | 0.994  | 0.844 |  0.901  |  0.750   |   0.961   | 0.863 | 0.759 |   0.889    |   0.920    | 0.780  |
|    Win端复现      | 0.919702 |  0.993178 | 0.879699 | 1.0      | 0.958874 | 0.986842 | 0.996825 | 0.819153 | 0.904268 | 0.875714 | 0.98045  | 0.878069 | 0.786637 | 0.897222 | 0.95625  | 0.882353 |
|    AIStudio复现      | 0.904630 |0.994783 | 0.896408 | 1.000000 | 0.934343 | 0.989474 | 0.994444 | 0.810907 | 0.900279 | 0.782143 | 0.960411 | 0.896890 | 0.810822 | 0.858333 | 0.890417 | 0.849790 |


### Pixel-Level AUC

|                |  Avg  | Carpet | Grid  | Leather | Tile  | Wood  | Bottle | Cable | Capsule | Hazelnut | Metal Nut | Pill  | Screw | Toothbrush | Transistor | Zipper |
| -------------- | :---: | :----: | :---: | :-----: | :---: | :---: | :----: | :---: | :-----: | :------: | :-------: | :---: | :---: | :--------: | :--------: | :----: |
| anomalib     | 0.968 | 0.984  | 0.918 |  0.994  | 0.934 | 0.947 | 0.983  | 0.965 |  0.984  |  0.978   |   0.970   | 0.957 | 0.978 |   0.988    |   0.968    | 0.979  |
|   Win端复现      |  0.968535| 0.991483 | 0.938801 | 0.991653 | 0.902967 | 0.940542 | 0.984302 | 0.953526 | 0.987435 | 0.980303 | 0.971491 | 0.957839 | 0.982025 | 0.988719 | 0.970605 | 0.986341 |
| AIStudio复现  |  0.968134 | 0.990992 | 0.934089 | 0.991474 | 0.904523 | 0.947558 | 0.983002 | 0.945316 | 0.984991 | 0.979178 | 0.978208 | 0.956459 | 0.981292 | 0.990511 | 0.972025 | 0.982397 |

达到论文复现验收标准.

训练及预测日志：[PaDiM](./logs/PaDiM.log)

AIStudio预训练权重：[notebook](https://aistudio.baidu.com/aistudio/projectdetail/3824965)

注意：该算法不需要模型训练，没有学习率设置和损失log，设定seed相同即可复现所有输出。

## 3. 准备数据与环境

### 3.1 准备环境

- 硬件：CPU\GPU

- 框架：
  
  - PaddlePaddle >= 2.2.0
  
  包依赖参见[requirements.txt](requirements.txt)
  
  在安装完PaddlePaddle之后，直接使用`pip install -r requirements.txt`安装依赖即可。

### 3.2 准备数据

数据集网站：[MvTec数据集](https://www.mvtec.com/company/research/datasets/mvtec-ad/)

AIStudio 中对应数据集 [MVTec-AD](https://aistudio.baidu.com/aistudio/datasetdetail/116034)

MVTec AD 是 MVtec 公司提出的一个用于异常检测的数据集，发布于 2019CVPR。与之前的异常检测数据集不同，该数据集模仿了工业实际生产场景，并且主要用于 unsupervised anomaly detection。数据集为异常区域都提供了像素级标注，是一个全面的、包含多种物体、多种异常的数据集。

训练集中只包含正常样本，测试集中包含正常样本与缺陷样本，因此需要使用无监督方法学习正常样本的特征表示，并用其检测缺陷样本。这是符合现实的做法，因为异常情况不可预知并无法归纳。

数据集包含不同领域中的五种纹理以及十种物体。
```
textures = ['carpet', 'grid', 'leather', 'tile', 'wood']
objects = ['bottle','cable', 'capsule','hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
```
### 3.3 准备模型


该算法使用resnet18等预训练模型作为特征提取器，可以直接调用paddle官方预训练权重。

## 4. 开始使用

可使用 [AIStudio notebook](https://aistudio.baidu.com/aistudio/projectdetail/3824965) 快速体验

### 4.1 模型训练

MVTec共有15个类别的子数据集，每个类别都需要单独提取训练集分布数据。
可用参数：
category指定数据类别，可用all代表全部类别，objects代表物体类别，textures代表所有纹理类别。
data_path指定数据集路径**PATH/TO/MVTec**
method 指定所用算法，PaDiM对应`--method=sample`
arch 指定所用backbone，复现任务为`--arch=resnet18`
k 指定所用特征数量，复现任务为`--k=100`
save_path指定模型保存路径
seed 设定随机数种子以便复现
eval表示是否在训练时评估模型表现

####全部训练并验证：
```bash
python train.py --data_path=PATH/TO/MVTec/ --category all --method=sample --arch=resnet18 --k=100 --eval
```

####单独训练某一类别（以carpet为例）：
```bash
python train.py --data_path=PATH/TO/MVTec/ --category carpet --method=sample --arch=resnet18 --k=100 --eval
```

### 4.2 模型评估
```bash
python eval.py --data_path=PATH/TO/MVTec/ --category all --method=sample --arch=resnet18 --k=100
```
也可以指定模型参数路径`--model_path` 及 类别 `--category`
```bash
python eval.py --data_path=PATH/TO/MVTec/ --category carpet --method=sample --arch=resnet18 --k=100
```
![验证](assets/carpet_eval.png)

### 4.3 模型预测
指定单张图片路径，生成预测结果
```shell
python predict.py PATH/TO/MVTec/carpet/test/color/000.png --category carpet --method=sample --arch=resnet18 --k=100
```

输出图像如下：
![检测](assets/carpet_predict.png)

## 5. 模型推理部署：预训练模型的静态图导出与推理测试

```shell
python export_model.py --depth 18 --img_size=224 --model_path=./output/carpet/best.pdparams --save_dir=./output
```
注意：该算法导出分为两个部分，一部分是预训练模型`model.pdiparams,model.pdmodel`，一部分是训练集获得的分布数据（平均值矩阵和精度矩阵）`stats`。

```shell
!python infer.py --use_gpu=True --model_file=output/model.pdmodel --input_file=/home/aistudio/data/carpet/test/color/000.png --params_file=output/model.pdiparams --category=carpet  --stats=./output/stats --save_path=./output
```
可正常导出与推理。
推理结果与动态图一致。
![infer](assets/carpet_infer.png)

## 6. 自动化测试脚本

**详细日志在[test_tipc/output](test_tipc/output/PaDiM)**

TIPC: [TIPC: test_tipc/README.md](test_tipc/README.md)

首先安装auto_log，需要进行安装，安装方式如下：
auto_log的详细介绍参考https://github.com/LDOUBLEV/AutoLog。
```shell
git clone https://github.com/LDOUBLEV/AutoLog
cd AutoLog/
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-1.2.0-py3-none-any.whl
```
进行TIPC：
```bash
bash test_tipc/prepare.sh test_tipc/configs/PaDiM/train_infer_python.txt 'lite_train_lite_infer'

bash test_tipc/test_train_inference_python.sh test_tipc/configs/PaDiM/train_infer_python.txt 'lite_train_lite_infer'
```
TIPC结果：
[输出日志](test_tipc/output/PaDiM.log)

## 7 参考链接与文献
- [PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization](https://arxiv.org/pdf/2011.08785)
- [anomalib](https://github.com/openvinotoolkit/anomalib)
- [PaDiM-Anomaly-Detection-Localization-master](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master)
- [PaDiM-Paddle](https://github.com/CuberrChen/PaDiM-Paddle)
- [Knowledge_Distillation_AD_Paddle](https://github.com/txyugood/Knowledge_Distillation_AD_Paddle)