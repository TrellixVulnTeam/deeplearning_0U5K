# 用于图像分割的卷积神经网络

使用深度学习进行目标检测最大的困难可能是生成一个长度可变的边框列表。诚如是？

卷积特征图将图片的所有信息编码到深度的维度上，同时保留着原始图片上目标物体的相对位置信息。可以通过变换的关系，找到原始图片对应特征图上的像素区域（很重要的一点是即使我们是在特征图上生成的锚点，这些锚点最终是要映射回原始图片的尺寸），如果只用到卷积和池化，那么最终特征图的维度与原始图片是呈比例的，**数学上，如果图片的尺寸是 w×h，那么特征图最终会缩小到尺寸为 w/r 和 h/r，其中 r 是次级采样率。如果我们在特征图上每个空间位置上都定义一个锚点，那么最终图片的锚点会相隔 r 个像素，在 VGG 中，r=16**。



**RCNN**

R-CNN的三个步骤：

1. 为边界框生成一组提案(选择性搜索算法)
2. 将边界框输入CNN进行特征提取，并通过svm判定边界框内图像的目标(How many object)
3. 将边框代入线性回归模型，一旦目标完成分类，输出边框的更紧密坐标，（即在边界框中找到了标，能否使边框适应真实目标的尺寸）

R-CNN的缺陷，性能很棒（预测结果），但是运行很慢，主要体现在两个方面：

1. 需要CNN针对图像的每个region proposal进行约2000次的前向传递；
2. 分别训练3个不同的模型，CNN生成图像特征，预测类别的分类器和收紧边界框的的回归模型。




>- 借助一个可以生成约2000个region proposal 的「选择性搜索」（Selective Search）算法，R-CNN 可以对输入图像进行扫描，来获取可能出现的目标。
>- 在每个region proposal上都运行一个卷积神经网络（CNN）。
>- 将每个CNN的输出都输入进：a）一个支持向量机（SVM），以对上述区域进行分类。b）一个线性回归器，以收缩目标周围的边界框，前提是这样的目标存在。

![/images_md/rcnn.png](/images_md/rcnn.png)

**Fast RCNN**

为解决这一问题，R-CNN 的第一作者 Ross Girshick提出fast RCNN

1. 在推荐区域之前，先对图像执行特征提取工作，通过这种办法，后面只用对整个图像使用一个 CNN（之前的 R-CNN 网络需要在 2000 个重叠的区域上分别运行 2000 个 CNN）。
2. 将支持向量机替换成了一个 softmax 层，这种变化并没有创建新的模型，而是将神经网络进行了扩展以用于预测工作。

不用针对每一个类训练很多的svm分类器，但仍然有一个瓶颈，即生成region proposal的selective search algorithm.

![/images_md/frcnn.png](/images_md/frcnn.png)






**Faster RCNN**

为解决区域提案的生成问题，Faster RCNN提出重复使用CNN提取的特征图以取代单独运行选择性搜索算法。region proposal network，得到包含目标的边界框（即寻找可能包含目标的区域），并将这些候选边界框传递给后续的分类器和收紧边界框的回归模型。

（第一个真正意义上的端到端训练，主要是指RPN和主干网络一起训练）

![/images_md/faster_rcnn.png](/images_md/faster_rcnn.png)

**Mask R-CNN**
扩展Faster RCNN以用于像素级分割

RoiAlign——**准确地将原始图像的相关区域映射到特征图上**

**F-RCN**



那么我们继续深入一些细节

RPN

如何生成anchors

```python
feature_map_shape (batch, height, width, depth)
all_anchors (num_anchors_per_points * feature_width * feature_height, 4)
output_stride spatial shape [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
```

针对每个锚点，我们需要考虑，这个锚点包含目标吗以及如何调整锚点以更好的拟合目标。



**非极大抑制（Non-maximum suppression）**:

解决一个目标被多次检测的问题

![/images_md/sc.png](/images_md/sc.png)

总算弄清楚一个问题，RPN网络针对一次卷积生成的特征图（512 for VCG-16）进行卷积，每个滑动窗口都会得到一个512-d的向量，而这一层次的核参数维度应当是 （512，3，3，512），同时新一层特征图有多大，就有几个这样的512-d向量



**关于固定大小的问题：**

经典的卷积神经网络有一个问题是它只能接受固定大小的输入图像，这是因为**第一个全连接层**和**它之前的卷积层之间（最后一个卷积层）**的权重矩阵大小是固定的，而卷积层、全连接层本身对输入图像的大小并没有限制。而在做目标检测时，卷积网络面临的输入候选区域图像大小尺寸是不固定的。

如何解决这个问题？

如果在最后一个卷积层和第一个全连接层之间做一些处理，将不同大小的图像变为固定大小的全连接层输入就可以解决问题。SPPNet引入了Spatial Pyramid pooling层



**RoI Pooling**

根据预选框的位置坐标，在特征图中将相应区域池化为固定尺寸的特征图，以便在最后的回归和分类问题中使用全连接层。



**Mask RCNN的RoI Align**

降低池化像素偏差



## 实现

For Faster RCNN, we can get a pipeline like this:

Image -(

pertrained cnn

) ->feature map-(

RPN(rpn proposals - rpn targets)

)->proposals-(

fast RCNN(rcnn targets - roi pooling - rcnn proposals)

)->classification(type), regression(box)

### rpn



```python
input：
conv_feature_map: [1, feature_map_height, feature_map_width, depth] 预训练CNN网络的输出，depth 512 for the default layer in VGG and 1024 for the defaultlayer in ResNet.
im_shape: A Tensor with the shape of the original image.
all_anchors: [feature_map_height * feature_map_width * total_anchors, 4]
gt_boxes: A Tensor with the ground-truth boxes for the image.
    Its dimensions should be `[total_gt_boxes, 5]`, and it should
    consist of [x1, y1, x2, y2, label], being (x1, y1) -> top left
    point, and (x2, y2) -> bottom right point of the bounding box.
```

初始anchors数为WxHxn^ 2（W,H分别特征图的宽和高，本文为3）

#### rpn proposals



### rcnn

#### **rcnn targets**



```python
input:
    proposals (num_proposals, 4)
    gt_boxes (num_gt, 5) index 4 for truth label 
```



```python
output:
    proposals_label 从rpn选取的proposals中与基准值gt_boxes相比较，给定标签从-1，0，···，n，
-1表示未达到设定阈值的区域，将被忽略，0表示bg，1，···，n表示真实类别
	bbox_targets (num_proposals, 4)为正类标签的proposals返回边界回归结果，其他返回0
```

判别标准：IoU（overlap）

并且在每一次得到新的proposals之后，都需要重新判定该边界是否满足边界的约束条件

#### RoI Pooling

```
resize to 14x14 and then stack a (2, 2)max pooling to get 7x7 feature
```



#### rcnn proposals

主要的操作是NMS(No Maxium Suppresion)

Step by Step

使用pointnet训练适应流体粒子个数的网络，我需要得到粒子下一时刻加速度

将粒子以加载模型，使用随机生成的数据填充tensor，得到粒子的输出（对于这种粒子数可变的情况，还是建议使用voxelnet，对某一帧粒子数据进行分组）

**从vexel对粒子进行分组开始。**
