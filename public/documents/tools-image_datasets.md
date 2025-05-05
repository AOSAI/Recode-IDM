## 1. load_data

### 1.1 递归获取所有图像文件路径

改写的都写在注释里了，包括调用的递归函数也是。

### 1.2 类别条件生成的索引映射

这里一共做了三件事，第一句代码是：假设类是文件名的第一部分，在下划线之前，把这些类名全部提取出来。

第二句代码是，先把这些类名，装进 set 容器中去重，然后对齐进行排序，再通过 enumerate 加上编号索引。假设排序之后的标签为 ['bird', 'cat', 'dog']，加上编号索引就变成了 [('bird', 0), ('cat', 1), ('dog', 2)]。最后再转换成字典的形式。

第三句代码表示，sorted_classes 是键值对的字典，所以 sorted_classes[x]就是提取索引 num，然后映射到 class_names 中的所有图像上去，相当于把单词的分类标签，转换成了数字标签，重新贴给了图像。

### 1.3 构造自定义数据集

这里原本还有两个参数，是多卡训练所使用的，但是我只用单卡，并且 dist_utils 文件已经删掉了，所以这里也删除不用了。

```py
shard=MPI.COMM_WORLD.Get_rank(),
num_shards=MPI.COMM_WORLD.Get_size(),
```

但是 ImageDataset 中还是保留了这两个默认参数，因为该默认参数就表示使用单卡训练：

- shard=0：就表示训练进程是唯一的；
- num_shards=1：表示总共只有一份数据，没有分片。

### 1.4 是否打乱数据的顺序

deterministic 这部分原本是这么写的，但是只有 shuffle 进行了修改，并且 shuffle 还是 deterministic 的相反值，所以我优化了一下代码。

```py
if deterministic:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    )
else:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    )
```

### 1.5 生成器函数

可以无限循环产出数据 batch，让训练循环永远不会“停”在数据加载上。训练代码里会像这样用它：

```py
data = next(data_iter)
```

## 2. ImageDataset

构造函数中的参数 shard 和 num_shards 已经在 1.3 小节部分讲过了。resolution 是我们设置的 image_size，用来调整实际图像的尺寸。image_paths 和 classes 就是传入的所有图像的路径 all_files 和 所有类别 classes。

```py
# 两种写法完全等价
self.local_images = image_paths[shard:][::num_shards]
self.local_images = image_paths[shard::num_shards]
```

local_classes 部分和 local_images 基本是一致的，这是多卡分片中的常用写法：

1. image_paths[shard:]：从第 shard 个元素开始（跳过前面 shard 个样本）；
2. [::num_shards]：每隔 num_shards 个元素取一个，也就是按间隔采样。

假设我们有 8 张图像，命名从 0 开始递增到 7，类似 img0、...、img7。现在我们有 2 张 GPU，那么就设置：

- num_shards = 2，步长为 2
- shard = 0, 1（分别代表 2 个进程）

这样就能让每个 GPU 加载不重复的样本，实现 分布式数据并行训练。切片结果的表格：

| 进程编号 | 切片方法             | 切片结果                         |
| -------- | -------------------- | -------------------------------- |
| 0        | image_paths[0:][::2] | ['img0', 'img2', 'img4', 'img6'] |
| 1        | image_paths[1:][::2] | ['img1', 'img3', 'img5', 'img7'] |

从这个图可以看出来，确定有几张 GPU 之后 num_shards 这个步长是绝对固定下来的。但是 shard，每个进程都自己调用一次数据加载方法，所有进程执行的是同一个代码逻辑，但由于 shard 不同，每个进程只加载属于自己的那部分图像。

当 shard = 0，并且 num_shards = 1 的时候，相当于什么都没有切，完全保留所有数据。

### 2.1 getitem

这个函数是 Dataset 类必须实现的，它定义了：“当你调用 dataset[i] 时，实际上发生了什么”。

首先是 idx，这个参数表示的是已经分好了的，各个进程的图像路径的局部索引，比如我们有 5000 张图像，rank0 有 2500 张，rank1 有 2500 张，idx 属于 [0, 2499]，表示不同进程下所包含图像的索引。

另外，据 ChatGPT 所说，blobfile 是 OpenAI 自家写的一个库，作用是：统一处理本地、远程和压缩包中的文件操作，就像 open() 的超进阶版。你可以用它读取：

- 本地磁盘文件（如 ./data/cat.jpg）；
- 各种云端文件，Google Cloud Storage 路径（如 gs://bucket/image.jpg）；
- zip、tar、嵌套路径中的文件。

优点是：

1. 接口统一：你不用管文件在哪里，代码统一写 with bf.BlobFile(...)；
2. 支持流式读取，不会因为打开大文件或网络资源导致内存爆炸；
3. 自动解码远程资源、缓存。

其他的部分基本注释里都写明了，需要注意的是，resample=Image.BOX，这个 BOX 是 Pillow 库中的一种重采样方法，用于下采样（缩小图像）时的高质量、抗锯齿的插值，约等于均值池化。

另外，这种处理方式，emmmm，也不是完全适用于所有图像吧，所以还是最好，自己先处理好图像。
