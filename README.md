# Codebase of NN-training based on Pytorch
# 项目clone
```
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:StoneAgeJL/NNBase.git
cd NNBase
git lfs pull
```
- 上述 `git lfs pull` 慢的话, 可直接下载 `./data/AVA/clip_b32.tar` 到本地对应路径

# 样例测试
- 单机单卡
```
python training.py ./configs/ava.yaml
```

# 各文件夹介绍
```
.
├── README.md
├── configs  # 训练的配置文件, 一次训练实验对应一个配置文件
│   └── ava.yaml
├── data  # 数据集存档位置, 放数据集的文本/图片/视频/标注, 数据集太大不好cp/mv的时候, 在这里新建个超链接也行
│   └── AVA  # 某个数据集的文件夹名
├── datasets  # pytorch的dataset类实现的地方, name里面一个py对应一个数据集的读取功能, 参考样例py文件, 注意需要在class前面加个registery, 从而实现类在build_dataset里面的记录注册
│   ├── __init__.py
│   ├── build.py
│   └── names # 各个数据集dataset实现代码
│       ├── advid.py
│       ├── ava.py
│       ├── koniq10k.py
│       ├── ssv2.py
│       ├── tevis.py
│       ├── vlp.py
│       └── vspd.py
├── experiments  # 建议将配置文件里面的模型保存路径、日志记录打印路径 和 tensorboard等的训练过程记录填这块的路径, 方便实验结果管理
│   ├── ckpt # 模型文件保存路径, 配置文件的tag标签字段会在这下面新建文件夹用于模型保存
│   ├── mess # 保存日志打印结果, 续写模式, 不改变日志打印路径的话，会在这里的txt文件中一直写
│   └── vis  # 一些可视化的结果可以保存在这里
├── models  # pytorch的nn.Module类实现的地方, blocks + loss -> nets, nets下面一个py文件对应一个模型的pipeline, 参考样例代码文件, 为了方便后续的训练流程和评价流程代码的编写, 我们这里的模型代码里面实现了 cal_loss 和 cal_metric 函数, 通过forward函数传入的参数来自动辨别我们模型此刻应当进行训练还是评估(出了指标计算不同, 对应的数据集也都不同了); 单纯的一次模型的推理可使用 single_inferrence 函数来实现. 这样也方便模型部署
│   ├── __init__.py
│   ├── blocks # 一些网络模块
│   │   ├── attn.py
│   │   ├── matcher.py
│   │   ├── position_encoding.py
│   │   ├── transformer.py
│   │   ├── transformer_encoder_droppath.py
│   │   └── vit_layer.py
│   ├── build.py
│   ├── loss # 一些损失函数
│   │   ├── info_nce.py
│   │   └── moco.py
│   └── nets # 模块框架代码, 就是把网络的积木搭起来的代码, 注意使用registery注册下, 便于后续使用build_model函数加载对应名称的模型
│       └── aesthetic_predictor.py
├── pipelines
│   ├── __init__.py
│   ├── arrange.py  # 训练评测代码框架, 包含日志记录,模型保存,训练迭代等
│   ├── build.py
│   ├── dist # 使用的分布式框架的代码
│   │   ├── ddp.py
│   │   └── ds.py
│   ├── log # 使用的日志记录的代码
│   │   ├── tb.py
│   │   └── wb.py
│   └── optim # TBD
├── tensorboard  # tensorboard日志记录结果会保存在这下面, 会以实验开启的时间戳为文件夹名
├── training.py  # 执行训练/评估认为的主函数
└── utils  # 一些有用的小段函数/功能代码都可以扔到这里
```
# 此项目的正确打开方式
- 整理好你的数据, 把对应的数据文件和标注在 `./data/` 下新建一个超链接
- 按照你的数据存储格式, 参考 `./datasets/names/` 下的代码, 好好实现下 `__init__`、`__len__` 和 `__getitem__`，分别用于数据&标注文件的加载、数据集长度读取 以及 单个训练/测试样本的读取
    - PS: 如果有一些特殊的batch化的数据加载, 可以在utils里面参考 `custom_collate_fn` 等函数实现一个自定义的 数据样本批量化合并 的函数, 在 `./pipelines/arrange.py` 里面的 `model_data_wrapper` 函数中的 `dataloader` 实现中加上
- 按照你的算法设计模型框图, 参考 `./models/nets/aesthetic_predictor.py`, 好好实现下 `__init__`、`single_inference`、`cal_loss`、`cal_metric` 以及 `forward`, 分别用于定义模型参数, 单次前向推理过程, 结合标注GT计算损失值, 结合标注计算度量评测值 以及 模块使用流程的挑选，具体参考 `aesthetic_predictor.py` 代码的写法
- 按照数据/模型部分的代码，基于你添加的超参数值(如resnet网络多少个block啊这类) 以及 训练过程中的超参(如训练使用的optim类型，learning rate大小等), 好好写下配置文件, 放到`./config/xx.yaml` 里面
- 按照下节的训练命令，建议先在单机单卡上面Debug下, 然后在上分布式

# 训练命令
- 单机单卡`调试`时: 
```
python training.py ./configs/xxx.yaml
```
- pytorch官方的`torchrun`分布式: 
```
torchrun \
    --nproc_per_node=4 \ # 你一台服务器上面有几张能用的显卡
    --nnodes=1 \  # 你准本使用几台服务器来炼丹
    --node_rank=0 \  # 你运行当前命令的这台服务器的进程编号(主进程填0,其他进程从1开始)
    --master_addr=xxx.xxx.xxx.xxx \ # 你主进程的服务器的网络IP, 命令行`ip addr`查看下IP地址, 用于多台机子训练时候的通信
    --master_port=xxxx \  # 和`master_addr`作用一样, 端口号
    training.py \ # 和单机单卡模式下一样, 得把训练/评估脚本文件加上
        ./configs/xxx.yaml \ # 此次训练/评估实验的配置文件
        --dist_type=ddp # 记得加上, 用于在torchrun分布式训练下修改代码里的一些行为
``` 
- deepspeed分布式:
    - TBD 待更新, 更加适合大模型的训练和部署

# 依赖环境
- requirements.txt仅供参考
- 不建议直接 pip install -r requirements.txt(因为每个人的设备不一样)
- 可以看看 requirements.txt 里面的几个关键库 (如pytorch, transformers, pyyaml等)
- 缺啥装啥, 上述txt环境是基于python 3.9的

# 其他
- 有问题欢迎issure或者邮箱联系我
