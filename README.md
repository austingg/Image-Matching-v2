# Image-Matching

## 数据集
共计 9935 个方案，373471个照片。

## 依赖
- Python 3.6.8
- PyTorch 1.3.0

## 如何使用

### 数据准备
数据解压缩并预处理:
```bash
$ python extract.py
$ python pre_process.py
```

### 训练
```bash
$ python train.py
```

可视化训练过程：
```bash
$ tensorboard --logdir=runs
```

## 效果评估

### 测试数据集

#### 开始评估
```bash
$ python test.py
```

#### 准确度
|Accuracy|Threshold|
|---|---|
|99.69%|19.82047499705621|


#### 角度 theta 的分布

![image](https://github.com/foamliu/Image-Matching/raw/master/images/theta_dist.png)

##### False Positive
16 false positives:

小样图|上刊照|
|---|---|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/0_fp_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/0_fp_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/1_fp_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/1_fp_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/2_fp_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/2_fp_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/3_fp_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/3_fp_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/4_fp_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/4_fp_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/5_fp_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/5_fp_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/6_fp_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/6_fp_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/7_fp_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/7_fp_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/8_fp_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/8_fp_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/9_fp_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/9_fp_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/10_fp_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/10_fp_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/11_fp_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/11_fp_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/12_fp_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/12_fp_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/13_fp_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/13_fp_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/14_fp_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/14_fp_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/15_fp_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/15_fp_1.jpg)|


##### False Negative
15 false negative:

1|2|1|2|
|---|---|---|---|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/0_fn_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/0_fn_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/1_fn_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/1_fn_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/2_fn_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/2_fn_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/3_fn_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/3_fn_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/4_fn_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/4_fn_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/5_fn_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/5_fn_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/6_fn_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/6_fn_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/7_fn_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/7_fn_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/8_fn_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/8_fn_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/9_fn_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/9_fn_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/10_fn_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/10_fn_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/11_fn_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/11_fn_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/12_fn_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/12_fn_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/13_fn_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/13_fn_1.jpg)|
|![image](https://github.com/foamliu/Image-Matching/raw/master/images/14_fn_0.jpg)|![image](https://github.com/foamliu/Image-Matching/raw/master/images/14_fn_1.jpg)|
