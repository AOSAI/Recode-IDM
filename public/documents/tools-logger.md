IDM 的 logger 文件原本是从 OpenAI 公司某一个产品的 baseline 中拆出来的，很多的内容都是老版本直接打包过来的，而且有一些我用不上，所以我在学习的同时，对其做了一个重构改写。

文件中的内容主要分了三大块，对各种日志文件保存形式的格式化、方便外部调用的 API 函数接口、Logger 的核心组件。

## 1. 各种类型的日志的 Format

第一个部分的切入点在 make_output_format 函数，它将 log 的类型分了 5 种：

| format 类型   | 作用区域     | 是否保留？ | 推荐用途          |
| ------------- | ------------ | ---------- | ----------------- |
| "stdout"      | 控制台       | ✅ 必留    | 实时调试          |
| "log"         | log.txt 文件 | ✅ 推荐    | 保存标准日志      |
| "json"        | json 文件    | ✅ 推荐    | 可视化分析        |
| "csv"         | csv 文件     | 🔄 可选    | Excel/Pandas 分析 |
| "tensorboard" | TensorBoard  | 🔄 训练用  | 图表展示（训练）  |

json 和 csv 可以二选一，效果都差不多，TensorBoard 主要用于训练时，实时进行的可视化分析。感觉都挺重要，就全部保留了。

另外，原本的文件名是用的 %s 这种旧字符串格式化方式：osp.join(ev_dir, "progress%s.csv" % log_suffix)。

我改写成了 Python3.6 版本之后的最推荐写法（f-string）：f"progress{log_suffix}.csv"

### 1.1 TensorBoardOutputFormat

IDM 所用的是 tensorflow.core.util 中的 event_pb2，这是老版本的遗产。TensorBoard 本身是 TensorFlow 开发的产物，但 PyTorch 社区专门做了一层适配 —— 所以我们不必用 tensorflow 模块也能写 TensorBoard 日志。

新代码真的异常的清爽简洁，PyTorch 的 SummaryWriter 直接处理日志写入，无需手动管理 Event 或 Summary 对象。并且由于 PyTorch 已经提供了自动的时间戳和步数处理，所以我们在 writekvs (add_scalar) 函数中使用 self.step 来管理每次的写入。

同时，通过 add_scalar 可以方便地扩展其他类型的日志，比如图像、音频、标量等。

### 1.2 CSVOutputFormat

这一块的主要逻辑是：

1. 记录目前已知的所有字段名（列名）
2. 后面每次写入前，会检查是否有新的 key（比如前面写的没有 loss, 后面突然来了）
3. 如果有，就重写表头，并把之前的记录也补上对应的空格

原本的代码是纯手搓的，厉害是真厉害，不过我还是按照 PyTorch 风格重新写了一遍。

csv.DictWriter 是 Python 的标准库 csv 模块提供的一个类，用来将字典写入 CSV 文件。基本用法为：

```py
import csv

with open("log.csv", "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "acc"])
    writer.writeheader()  # 写入表头
    writer.writerow({"epoch": 1, "loss": 0.35, "acc": 0.9})
    writer.writerow({"epoch": 2, "loss": 0.25, "acc": 0.92})
```

这个会输出一个标准 CSV 文件。优点是，不用手动拼字符串或加逗号；顺序自动匹配 fieldnames；缺失字段会留空，安全稳定。

### 1.3 JSONOutputFormat

JSON 保存前的格式化写的非常简单，具体的用法已经注释在代码里了。

### 1.4 HumanOutputFormat

这个函数也是纯手搓的 **控制台输出 + txt 文件日志** 两用格式化方法，写的挺好的，没啥要改的，就改动了老版本的字符串格式化方式。

（1）构造函数 init

支持两种初始化方式，**字符串路径**：比如 "log.txt"，那就自己去 open() 一个文件。**已经打开的文件对象**：比如 sys.stdout，就直接拿来用。

self.own_file 是标志位，如果是自己打开的文件，close 的时候要负责关掉；不是自己的，就不关。（因为 stdout 不能自己随便关掉）

（2）writekvs

```py
key2str = {}
for (key, val) in sorted(kvs.items()):
    if hasattr(val, "__float__"):
        valstr = "%-8.3g" % val
    else:
        valstr = str(val)
    key2str[self._truncate(key)] = self._truncate(valstr)
```

第一段的循环，先对字典 kvs.items() 按名字排序，然后获取字典的键值对 key-val；再判断值 val 是否是数值类型（能转 float），就格式化成短字符串，最多显示 8 个字符宽、3 位有效数字。

self.\_truncate 是下面写的，字符串长度控制的方法，如果字符串超过 30 字符，就只保留前 27 个字符，末尾补上 ...。防止一行内内容太多，造成爆行。

```py
if len(key2str) == 0:
    print("WARNING: tried to write empty key-value dict")
    return
else:
    keywidth = max(map(len, key2str.keys()))
    valwidth = max(map(len, key2str.values()))
```

第二段的条件判断是说，如果字典是空的，提醒一声，然后什么也不写；否则，找出最长的 key 和 value 的长度，做对齐用（美化输出）。

```py
dashes = "-" * (keywidth + valwidth + 7)
lines = [dashes]
for (key, val) in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
    lines.append(
        "| %s%s | %s%s |"
        % (key, " " * (keywidth - len(key)), val, " " * (valwidth - len(val)))
    )
lines.append(dashes)
self.file.write("\n".join(lines) + "\n")
self.file.flush()
```

这里就是一个美化输出的内容，打印成一个小表格的样子，两边加 |，中间对齐，表头和底部用 ----- 分隔，最后刷到控制台或文件里去。

（3）writeseq

这里是另一种模式，直接写一串值（比如一堆字符串），用空格隔开，也是写完就马上 flush。

也有能用到的时候，比如：1）训练初期提示；2）超参数设置回显；3）异常警告（比如模型炸了）；4）评估指标的一些特殊说明；等等

（4）close

用 own_file 标志来判断文件时自己打开的还是外部来的，我们只关闭自己打开的文件，如果是 sys.stdout 这种外部传进来的，啥也不做。

### 1.5 KVWriter 和 SeqWriter

这两个是抽象基类，KVWriter 规范了 "你得能写 key-value"，SeqWriter 规范了 "你得能写序列"，老版的写法是类似这样的：

```py
class KVWriter(object):
    def writekvs(self, kvs):
        raise NotImplementedError
```

这种写法从 Python 很早期就有了，它属于一种非正式的抽象基类，靠的是约定俗成：大家看到 raise NotImplementedError，就知道 "哦，这是个接口，需要子类去实现"。

如果这个方法没有被子类重写，但是调用了，就会报错。只不过 Python 解释器不会强制检查，你漏了写，程序到运行时才会报错。

而 ABC (Abstract Base Class) 是 Python 后来加的标准模块，大概是在 Python 2.6 / 3.0 那一代（2008 年左右）引入的。它让抽象基类变成了正式的第一类公民，可以：

1. 用 @abstractmethod 装饰器，明确标记哪些方法必须实现。
2. 子类如果漏了实现，在子类定义阶段就会报错（不是运行时报）。
3. 让 Python 解释器能帮你在代码写的时候就做强校验。

## 2. Logger API（原名 API）

这一块的理解说简单也简单，但是需要先看第三部分。实际上这里的前几个函数都是在调用 Logger 类中的方法，只不过封装成了函数，在外部调用会更方便一些。

中间的看到的各个等级，都是调用的 log 函数，而 log 函数又调用的是 Logger.log 函数，表示输出一句话，但是等级可能不同（debug、info、warning、error）。

```py
record_tabular = logkv
dump_tabular = dumpkvs
```

而两句话，只是简单的起了一个别名的意思，也许是为了给外部调用者多提供一个友好的名字，而不是 logger 自己用的。但是因为目前没有遇见，所以先注释掉。

### 2.1 自定义的 profile 装饰器

先看 profile_kv，它是一个计时器工具，用来统计一段代码运行了多久。开始时 tstart = time.time() 记录当前时间，代码块运行完成后，finally 中计算用时，然后把用时加到日志里。

它不会立刻 dump，只是把 时间差 记录到当前的 Logger.name2val 中，以 "wait\_" + scopename 作为 key。

再来看 profile 函数装饰器，用来方便的给函数计时，使用方法如下：

```py
@profile("train_step")
def train_step():
    # do training
```

这样，当我们调用 train_step() 时，会自动在内部打一个 profile_kv("train_step") 的计时，
不用手动写 start = time.time() 之类的了。

那么它是如何实现的呢？四层函数的嵌套，有点难懂啊。它使用的过程是这样的：

1. @profile("train_step")：调用外层函数。会被立即执行，接受参数 n="train_step"，然后返回一个新的函数 decorator_with_name。

2. decorator_with_name(func)：装饰目标函数。接收你的原函数（比如 train_step），然后返回一个包裹版的函数 func_wrapper。这个 func_wrapper 是真正被执行的版本。

3. func_wrapper(\*args, \*\*kwargs)：执行计时 + 调用原函数。当 train_step() 函数被调用时：

   - 先执行 with profile_kv(n)，也就是开启计时
   - 再执行 return func(\*args, \*\*kwargs)，也就是执行你的原函数内容
   - 最后退出 with 块，自动记录用时

## 3. Logger Management（原名 Backend）

### 3.1 定义 Logger.CURRENT 和 Logger.DEFAULT 的三件套

主类是 Logger 没错，但是我们从 API 中调用最多的 get_current 开始说起。它先是判断了 Logger 中的 CURRENT 是否为空，为空的话，调用 \_configure_default_logger 函数，返回 Logger.CURRENT。

\_configure_default_logger 中又调用了 configure 函数，这个函数是真正决定 Logger.CURRENT 参数的内容。首先先看原版的第一段：

```py
if dir is None:
    dir = os.getenv("OPENAI_LOGDIR")
if dir is None:
    dir = osp.join(
        tempfile.gettempdir(),
        datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"),
    )
```

这里首先检查是否传递了 dir 参数。如果没有，则尝试从环境变量 OPENAI_LOGDIR 中读取目录路径。如果 dir 仍然没有值，则使用 tempfile.gettempdir() 获取系统的临时文件夹路径，并根据当前的时间戳生成一个以日期时间为名称的目录。这可以确保每次运行时都会创建一个独立的日志文件夹。

虽然大多数大公司都是这么做的，不过我们小型项目，我觉得输出在本身项目内是最方便的，所以我设置了一个 output 文件夹，至于上传 github，用 gitignore 文件把输出忽略掉就行。

至于时间戳里面的 -%f 是一个 6 位数微秒的计数，虽然我觉得用不到这么精确的，但是也不影响使用，就留着吧。

```py
assert isinstance(dir, str)
dir = os.path.expanduser(dir)
os.makedirs(os.path.expanduser(dir), exist_ok=True)
```

接下来的三行，首先是判断文件夹路径 dir 是否为一个字符串形式，不符合就报错；expanduser 这个函数，是针对 linux 和 mac 系统的，它们的路径有的时候会是 “~/xxxx”，expanduser 就负责将 “~” 展开，获取到一个绝对路径。

而我写的 "../output/" 是一个相对路径的形式，makedirs 就能正确的识别判断，但是为了兼容性，这里不改动。

```py
rank = get_rank_without_mpi_import()
if rank > 0:
    log_suffix = log_suffix + "-rank%03i" % rank

def get_rank_without_mpi_import():
    # 在此处检查环境变量，而不是导入 mpi4py，以避免在导入此模块时调用 MPI_Init()。
    for varname in ["PMI_RANK", "OMPI_COMM_WORLD_RANK"]:
        if varname in os.environ:
            return int(os.environ[varname])
    return 0
```

原版中使用了 get_rank_without_mpi_import 获取多张显卡同时训练的 rank。而我只用单卡，所以这里删除。同时，下面的这段代码也可以简化：

```py
if format_strs is None:
    # if rank == 0:
    format_strs = os.getenv("OPENAI_LOG_FORMAT", "stdout,log,csv").split(",")
    # else:
        # format_strs = os.getenv("OPENAI_LOG_FORMAT_MPI", "log").split(",")
format_strs = filter(None, format_strs)
output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]
```

这段代码的意思是，如果 format_strs 为空，先从 OPENAI_LOG_FORMAT 这个环境变量中找值，如果没找到，就使用默认的 "stdout,log,csv"。filter 函数会移除空字符串，以确保只保留有效的格式字符串。

output_formats 调用了 make_output_format 函数，我们自定义的不同类型日志格式化的函数，用来创建相应的输出格式对象。

```py
Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, comm=comm)
if output_formats:
    log("Logging to %s" % dir)
```

最后两句话很简单，使用 Logger 类创建一个日志记录器对象 Logger.CURRENT，并传入日志目录 dir、输出格式 output_formats 以及通信对象 comm（如果有的话）。此外，如果配置了输出格式，则输出一条日志，表示日志将被保存到 dir 目录下。

### 3.2 主类 Logger

（1）构造函数，成员变量

configure 中调用 Logger 类传入的参数为 dir 和 output_formats，分别代表输出文件夹和输出方式的列表（在 configure 中的默认设置为 stdout, log, csv）。level 是用来设置日志等级的，在第二部分设置了 5 个值。

- self.name2val = defaultdict(float)：记录 每个 key（名字）对应的数值，如果访问某个 key 时没设置过，默认是 0.0。
- self.name2cnt = defaultdict(int)：记录 每个 key（名字）出现的次数，如果访问某个 key 时没设置过，默认是 0。

他们来自 Python 的 collections.defaultdict，是 带默认值 的字典。这么写的好处是，在训练时，我们会不断地记录各种指标（比如 loss、accuracy、reward 等），有些指标可能一开始没出现，但依然可以直接用，不需要自己先判断是否存在。

举个例子，加入前三轮训练，loss 没有出现过，但是第 4 轮开始出现了，那么这个字典也不会报错，而是直接从 0 加上：

```py
logger.name2val["loss"] += 0.5
logger.name2cnt["loss"] += 1
```

（2）日志记录 API-1

| 方法                 | 说明                                                           |
| -------------------- | -------------------------------------------------------------- |
| logkv(key, val)      | 直接记录一组 key 和 val（后面可以 dump，写入日志、清空缓存区） |
| logkv_mean(key, val) | 如果一个 key 多次出现，取均值（比如一次训练多个 batch）        |
| dumpkvs()            | 把 name2val 里的所有数据交给 output_format 写入                |

logkv_mean 是说，如果你每次训练都调用 logkv_mean("loss", 当前这次的 loss)，它会自动帮你算到目前为止的平均 loss，而不是直接覆盖原本的 loss。

（3）普通日志记录 API-2

- log(\*args, level=INFO) 是 一般打印用的，不是 key-value。
- \_do_log(args) 里面遍历所有输出器，只要它是 SeqWriter 就把整个 args 当一行写进去。而所有类型的日志里，只有 HumanOutputFormat 中继承了 SeqWriter，所以这个输出，是美化后的终端输出。

（4）Configuration

- set_level：控制日志的等级
- get_dir：拿到日志目录
- close：关闭所有打开的文件

### 3.3 日志重置 & 上下文管理

reset 函数的作用是将当前使用的日志记录器（Logger.CURRENT）重置为默认的日志记录器（Logger.DEFAULT）。如果当前日志记录器不是默认的，就会关闭当前日志记录器，恢复为默认日志记录器。最后通过 log("Reset logger") 输出日志，表示日志记录器已经重置。

scoped_configure 是一个上下文管理器（使用 @contextmanager 装饰器），用于在特定的代码块中配置和使用一个新的日志记录器。在 yield 语句之前，它会配置一个新的日志记录器（通过调用 configure()）。在 yield 之后，无论是正常退出还是抛出异常，它都会关闭当前日志记录器，并恢复为之前的日志记录器。

这两个函数在单卡训练中是用不到的，所以注释掉了。
