# Voxelnet

确定每个voxel的边界，从而将整体仿真空间划分为三维网体空间。

**grouping**:

将粒子根据坐标（全部转为正向坐标）将粒子划分入网体（voxel）内,feature shape(K, T， 7) K为当前粒子所能划分的网体数目，T为每个网体内最大的粒子容量，7为每个粒子编码的特征长度。

实际处理中是将网体按出现顺序编号，粒子已将以出现顺序编码在每个网体中。

训练中batch划分分为两个层次：

1. 首先对所有训练数据划分batch，每个batch将会有batch_size个文件
2. 根据GPU数目在每个batch内划分子batch，每个batch内的batch_size个文件将会被拼接，拼接的时候会在cordinate（标示网体位置的长度为3的一维元组）前添加不同文件的表识。

**VFE(Voxel Feature Encoding) layer**:

每个VFEdense全连接+max pooling

maxpoling的作用元素包括哪些呢？

FeatureNet mask

在FeatureNet之后，得到的是tf.scatter_nd根据batch和网体坐标生成的5-D全空间张量