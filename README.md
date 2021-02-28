# 前情提要

先看题目吧，看经验帖子，看数据，模型不急。

1. [原文地址](https://tianchi.aliyun.com/competition/entrance/531871/information)
2. [他人baseline](https://github.com/datawhalechina/team-learning-data-mining/tree/master/WeatherOceanForecasts)
3. [他人EDA](https://github.com/ydup/ENSO-Forecasting/blob/master/EDA.ipynb)
4. [时空序列建模系列文章](https://www.zhihu.com/column/c_1208033701705162752)

# 任务计划

数据挖据四步走了。

1. EDA(Exploratory Data Analysis)
2. 特征工程
3. 模型构建
4. 集成

# EDA 进度

- [x] [读取nc格式文件方法](http://www.clarmy.net/2018/11/01/python%E8%AF%BB%E5%8F%96nc%E6%96%87%E4%BB%B6%E7%9A%84%E5%85%A5%E9%97%A8%E7%BA%A7%E6%93%8D%E4%BD%9C/)
- [x] 转换成 `pandas` 或 `numpy`，处理与分析容易
- [x] 填充值需要注意

## SODA EDA

无填充值。

- [x] 数据集透视完毕，包括数据维度、填充值分析与数据分布
- [x] 标签集透视完毕，包括填充值分析、去除重叠数据后的可视化

## CMIP EDA

有填充值，`t300`缺失3055032条，`ua`缺失13921123条，`ua`和`va`的缺失数据分布相同。

- [x] 数据集透视完毕，包括数据维度、填充值分析与数据分布。`t300`缺失的数据年份为[1812, 1962]，即第13个模式缺失；`ua, va`缺失的数据年份为[755, 1962]，第6到第13个模式消失。共755条。
- [x] 标签集透视完毕，包括填充值分析、去除重叠数据后的可视化。

# 模型构建进度

- `seq_length` 原始序列长度
- `custom_length` 指定窗口大小

- [x] Conv2dLSTM 神经单元实现与适配到多层网络（训练读入的是单个元素）
```
Input  x : (batch_size, seq_length, channels, height, width)
Output h : (batch_size, seq_length, hidden_dim, height, width)
       c : (batch_size, seq_length, hidden_dim, height, width)
```
- [x] Conv3dLSTM 神经单元实现与适配到多层网络（训练读入一个序列的元素）
```
Input  x : (batch_size, channels, seq_length, height, width)
Output h : (batch_size, hidden_dim, seq_length, height, width)
       c : (batch_size, hidden_dim, seq_length, height, width)
```