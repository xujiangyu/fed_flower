# 环境要求：

**可稳定支持Linux和Windows操作系统，其他操作系统未测试。**

python >= 3.8

pytorch >= 1.11+cu115





# 环境配置：

```shell
pip install -r requirements.txt
```





# 数据集下载：

#### nii.gz格式二分类数据集COVID2019 CT扫描图像

链接：https://pan.baidu.com/s/1mSo6ufV53zVwYLrKuDXhVg 
提取码：no0r 



#### nii.gz格式4模态4类别脑部肿瘤分割数据集BraTs2017 CT扫描图像

链接：https://pan.baidu.com/s/1NDRm8t2kRrqE79ZRa4c5nQ 
提取码：g377 



#### nii.gz格式单模态单通道2类别胰脏分割分割数据集Pancreas CT扫描图像

链接：https://pan.baidu.com/s/15Aa3lZfriNW7csJ8MH1fZA 
提取码：h7j0 



**请严格按照下面数据存放格式修改存放数据。**





# 数据存放：

## 图像分类

`./data/`（无论是RGB图像还是CT图像，需保证最后一级子目录名字为分类标签，分类标签需用可转为int型的字符表示，如0、1、2、……）

## 图像分割

待分割图像路径：`./data/images`

已标定标签路径：`./data/labels`

**需严格保证images路径下的图像与labels下的标签是一一对应的，不允许存在有未标定的图像及没有图像对应的标签。**





# 配置文件：

以`./config/UNet3D.json`为例

```json
{
    "task": "segmentation",
    "model":"UNet3D",
    "data_param":{
      "data_random_split": true,
      "data":"E:/01_work/02_data/010_FL_data/05_Task01_BrainTumour",
      "trainset": "E:/01_work/02_data/010_FL_data/06_BrainTumour_part/train",
      "testset": "E:/01_work/02_data/010_FL_data/06_BrainTumour_part/test",
      "data_type":"CT",
      "channel": 4,
      "num_classes": 4,
      "img_size": [128, 128, 64],
      "rotate": false
    },
    "bench_param":{
        "server_address": "localhost:8090",
        "device": "cuda:0",
		"num_rounds": 20
    },
    "training_param": {
        "epochs": 20,
        "batch_size": 2,
        "learning_rate": 0.001,
        "loss_func": "dice_loss",
        "optimizer": "adam",
        "optimizer_param": {
            "betas1": 0.9,
            "betas2": 0.999,
            "eps": 1e-08,
            "weight_decay": 0,
            "amsgrad": false
        }
    },
  "testing_param": {
        "model_path": "save_model/test_model_segmentation_UNet3D_BraTs.pth",
        "test_path": "E:/01_work/02_data/010_FL_data/05_Task01_BrainTumour",
        "test_save_path": "prediction/"
        }
}

```

`task`: 联邦学习任务，可选“`classification`”和“`segmentation`”

`model`：网络模型。

`data_param`：数据参数。

`data_random_split`：bool变量，是否为随机划分数据集的方式。若为`true`，加载`data`路径下数据集，并按照4:1的比例随机划分训练集和测试集；若为`flase`，需提前手动划分好训练集和测试集，并将训练集和测试集路径分别填入下面“`trainset`”和“`test`”中。

`data`：数据存放根目录。

`trainset`：已手动划分好的训练集路径。

`testset`：已手动划分好的测试集路径。

`data_type`：训练及测试的图像数据类型，可选"`RGB`"或“`CT`”图像。

`channel`：图像通道，若为RGB图像则为3通道，若为单模态CT图像则为1通道，若为n模态CT图像则为n通道，此处采用BraTS 2017数据集，该数据集采用FLAIR、T1W、T1GD、T2W四种造影方式采集同一部位CT图像，对应4种模态的CT图像信息，故通道设置为4。

`img_size`：为训练图像初始分辨率，对应二维RGB图像的H、W，三维CT图像的H、W、D，越大图像信息越丰富，同时越占用显存，训练越慢。

`rotate`：bool变量，是否做图像90°旋转。若显示CT扫描图像为竖，可选择该参数为`true`，若为正常图像可选`false`不做旋转。

`bench_param`：联邦学习参数。

​	`server_address`为服务器IP地址。

​	`device`为训练设备，可选`cuda:0`或`cuda:1`，`cuda:…`或`cup`。

​    `num_rounds`为联邦聚合训练轮次。

`training_param`：训练参数。

​	`epochs`为训练轮次；

​	`batch_size`为一次训练塞入的图片数量，越大越占用显存。

​	`learning_rate`为学习率。

​	`loss_func`为训练损失，分类模型支持`cross_entropy`、`mse`损失，分割模型支持`dice_loss`及`cross_entropy_3d`损失，暂不支持其他选项。

​	`optimizer`为优化器，可选`sgd`和`adam`。

​	`optimizer_param`为优化器参数；

`testing_param`：前向推理测试参数。

`model_path`：模型路径；

`test_path`：测试集或待推理数据存放路径；

`test_save_path`：分割图像推理结果存放路径（分割任务独有）；

`labeled`：是否存在标签，若存在则计算精度，若不存在则只做前向推理（分类任务独有）。





# 快速联邦训练：

```shell
sh run.sh
```

训练完成后模型保存在`save_model`目录下





# 正常联邦训练：

在服务器上执行

```shell
python server.py --config=./config/DenseNet.json
```

在每台客户端上执行

```shell
python client.py --config=./config/DenseNet.json
```

参数可以在`config`下的`DenseNet.json`中修改，**也可以选择用默认参数直接**：

在服务器上执行

```shell
python server.py
```

在每台客户端上执行

```shell
python client.py
```

训练完成后模型保存在`save_model`目录下





# 集中式训练：

```shell
python train.py --config=./config/DenseNet.json
```





# 前向推理与测试：

```shell
python test.py --config=./config/DenseNet.json
```





# 可视化：

终端模式在工程目录下

```shell
tensorboard --logdir=./logs/visualization/年月日时分秒/
```