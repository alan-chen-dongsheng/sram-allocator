# SRAM Allocator — Strategy Pattern Neural Network Memory Allocator

[![C++](https://img.shields.io/badge/C%2B%2B-17-blue)]()
[![Python](https://img.shields.io/badge/Python-3.12+-green)]()
[![pybind11](https://img.shields.io/badge/pybind11-3.0-purple)]()

基于**策略模式**的神经网络 SRAM 内存分配器。给定计算图（DAG），自动计算每个中间张量的内存偏移，通过**内存复用**最小化峰值 SRAM 使用量。

## 目标需求

- [x] 解析神经网络 DAG，自动推导张量生命周期
- [x] 策略模式架构：Linear Scan / Best Fit / Largest First 三种分配算法
- [x] 运行时切换策略，一行代码不影响调用方
- [x] **内存上限溢出检测**：设定 `max_memory`，超限不停止，最终汇总报告（含每个 liveTime 点的违规张量列表）
- [x] **Perfetto Trace 输出**：分配过程持续产生 Chrome Trace JSON，可直接导入 perfetto UI 可视化
- [x] C++ 通过 pybind11 暴露给 Python，纯 Python 命令即可使用

## 架构设计

### 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    Python Layer                          │
│  SRAMAllocator / DAGBuilder / PerfettoTracer            │
│  AllocatorResult (overflow / violations / timeline)      │
├──────────────────────┬──────────────────────────────────┤
│                   pybind11                               │
│  _core.cpython-*.so                                      │
├──────────────────────┼──────────────────────────────────┤
│                    C++ Layer                             │
│                                                          │
│  ┌────────────────────────────────────────────┐         │
│  │        AllocationStrategy (interface)       │         │
│  │  + name()      + allocate(tensors, config) │         │
│  └──────────┬─────────────┬─────────────┬────┘         │
│             │             │             │               │
│  ┌──────────▼──┐ ┌───────▼──────┐ ┌────▼──────────┐   │
│  │LinearScan    │ │BestFit       │ │LargestFirst   │   │
│  │(First-Fit)   │ │(最小碎片)    │ │(按大小降序)   │   │
│  └──────────────┘ └──────────────┘ └───────────────┘   │
│             │             │             │               │
│  ┌──────────▼─────────────▼─────────────▼──────────┐   │
│  │          SRAMAllocator (Context)                 │   │
│  │  + set_strategy()  + set_max_memory()           │   │
│  │  + set_tracer()    + allocate()                 │   │
│  └─────────────────────────────────────────────────┘   │
│             │                                          │
│  ┌──────────▼──────────────┐                          │
│  │    PerfettoTracer       │                          │
│  │  on_alloc / on_free     │                          │
│  │  on_peak / on_overflow  │                          │
│  │  → Chrome Trace JSON    │                          │
│  └─────────────────────────┘                          │
└─────────────────────────────────────────────────────────┘
```

### 数据结构

| 结构体 | 用途 |
|--------|------|
| `TensorLifetime` | 张量生命周期：name, size, alive_from → alive_to |
| `OpNode` | 算子节点：name, op_type, inputs, output, depends_on |
| `DAG` | 计算图：ops 列表 |
| `AllocationResult` | 分配结果：offsets, sizes, peak_memory, timeline, overflow_report |
| `OverflowReport` | 溢出报告：violations 列表（每个包含 time, active, limit, excess, live_tensors） |
| `AllocConfig` | 配置：alignment, sram_size, max_memory, verbose, tracer |

### 设计原则

1. **开闭原则**：新增策略只需继承 `AllocationStrategy` 并注册，不改现有代码
2. **依赖倒置**：`SRAMAllocator` 依赖抽象接口，不依赖具体策略
3. **零开销可选**：`tracer=nullptr` 时零额外开销，`max_memory=-1` 时不检测溢出

## 快速开始

### 编译

```bash
cd sram-allocator
mkdir -p build && cd build
cmake .. && cmake --build . -- -j$(nproc)
```

编译产物：
- `build/libsram_alloc.so` — C++ 共享库
- `build/_core.cpython-312-x86_64-linux-gnu.so` — pybind11 Python 扩展

将 `.so` 文件拷贝到 Python 包目录：

```bash
cp build/_core.cpython-312-x86_64-linux-gnu.so src/python/sram_allocator/
cp build/libsram_alloc.so src/python/sram_allocator/
```

### 安装

```bash
pip install -e . --break-system-packages
```

### 运行测试

```bash
python3 test.py              # 5 个基础场景测试
python3 test_strategies.py   # 策略对比 + 隔离测试
```

## Python API 文档

### 基础分配

```python
from sram_allocator import SRAMAllocator, DAGBuilder

# 构建 DAG
builder = DAGBuilder.build_simple_cnn("my_cnn")
dag = builder.dag

# 默认 Linear Scan 策略
alloc = SRAMAllocator(alignment=64)
result = alloc.allocate(dag)

print(result.summary())
# Peak SRAM: 1.16 MB
# Total (no reuse): 1.22 MB
# Reuse ratio: 5.5%
# Reused tensors: 2
```

### 切换策略

```python
from sram_allocator import SRAMAllocator, BestFitStrategy, LargestFirstStrategy

alloc = SRAMAllocator()

# 方式 1：构造函数注入
alloc = SRAMAllocator(strategy=BestFitStrategy())

# 方式 2：运行时切换
alloc.set_strategy(LargestFirstStrategy())

# 方式 3：通过名称切换
alloc.set_strategy_by_name("linear_scan")
```

### 内存上限 + 溢出检测

```python
from sram_allocator import SRAMAllocator, PerfettoTracer

# 设置 200KB 上限
alloc = SRAMAllocator(max_memory=200 * 1024)
result = alloc.allocate(dag)

if result.overflow:
    print(f"⚠️ 峰值超出限制！")
    print(f"   峰值: {result.peak_memory}B")
    print(f"   限制: {result.violations[0].limit}B")
    print(f"   超出: {result.violations[-1].excess}B")

    for v in result.violations:
        print(f"   t={v.time}: 活跃={v.active_memory}B, 存活张量={v.live_tensors}")
```

**关键行为**：超限后分配**不会停止**，正常完成所有张量分配，最终在 `result.overflow` 和 `result.violations` 中汇总所有违规点。

### Perfetto Trace 可视化

```python
from sram_allocator import SRAMAllocator, PerfettoTracer

# 创建 tracer
tracer = PerfettoTracer()

# 注入到 allocator
alloc = SRAMAllocator(tracer=tracer)
result = alloc.allocate(dag)

# 保存 Chrome Trace JSON
alloc.save_perfetto_trace("trace.json")
```

打开 [perfetto UI](https://ui.perfetto.dev/)，拖入 `trace.json` 即可查看：
- **SRAM: xxx** — 内存分配区间（B=begin / E=end）
- **free: xxx** — 内存释放点（instant event）
- **PeakMemory** — 峰值内存计数器（counter）
- **OVERFLOW** — 溢出事件点（instant event）

### DAG 构建

```python
from sram_allocator import DAGBuilder

# 手动构建
builder = DAGBuilder("my_network")
builder.add_op("conv1", "Conv", [], 65536)
builder.add_op("conv2", "Conv", ["conv1"], 65536)
builder.add_op("fc1", "FC", ["conv2"], 131072)
dag = builder.dag

# 或使用预设模板
builder = DAGBuilder.build_resnet_block("b1", 64, 64, 56, 56)
builder = DAGBuilder.build_linear_chain("chain", num_ops=10, tensor_size=4096)
builder = DAGBuilder.build_fork_join("fork_join")

dag = builder.dag
```

### 注册自定义策略

```python
from sram_allocator import SRAMAllocator, register_strategy

register_strategy("my_new_strategy", lambda: MyCustomStrategy())
alloc = SRAMAllocator()
alloc.set_strategy_by_name("my_new_strategy")
```

## 策略对比

| 场景 | Linear Scan | Best Fit | Largest First |
|------|------------|----------|---------------|
| Linear Chain (10 ops) | 8KB (80%) | 8KB (80%) | 8KB (80%) |
| Simple CNN | 1184KB (6%) | 1184KB (6%) | 1184KB (6%) |
| ResNet Block | 3136KB (43%) | 3136KB (43%) | 3136KB (43%) |
| Fork-Join | 32KB (20%) | 32KB (20%) | 32KB (20%) |

## 项目结构

```
sram-allocator/
├── CMakeLists.txt                      # CMake 构建配置
├── pyproject.toml                      # Python 包配置
├── README.md                           # 本文档
├── .gitignore
├── test.py                             # 基础功能测试 (5 cases)
├── test_strategies.py                  # 策略对比测试
├── src/
│   ├── cpp/
│   │   ├── linear_scan.h               # 核心头文件：接口 + 数据结构 + Context + PerfettoTracer
│   │   ├── linear_scan.cpp             # 三种策略实现 + PerfettoTracer + 溢出检测
│   │   └── bindings.cpp                # pybind11 绑定
│   └── python/
│       └── sram_allocator/
│           ├── __init__.py
│           ├── allocator.py            # Python 高层 API
│           ├── builder.py              # DAG 构建器
│           └── visualizer.py           # 报告 + 可视化
└── build/                              # 编译产物（gitignore）
```

## 如何新增策略

```cpp
// 1. 头文件中声明
class MyNewStrategy : public AllocationStrategy {
public:
    std::string name() const override;
    AllocationResult allocate(
        const std::vector<TensorLifetime>& tensors,
        const AllocConfig& config) const override;
};

// 2. cpp 文件中实现
std::string MyNewStrategy::name() const { return "MyNew"; }

AllocationResult MyNewStrategy::allocate(
    const std::vector<TensorLifetime>& tensors,
    const AllocConfig& config) const {
    // 你的算法
    // 注意：在循环中调用 config.tracer->on_alloc/on_free
    // 在后处理中调用 config.tracer->on_peak/on_overflow
}

// 3. 自动注册
struct AutoRegisterMyNew { AutoRegisterMyNew() {
    SRAMAllocator::register_strategy("my_new",
        []() { return std::make_shared<MyNewStrategy>(); });
}};
static AutoRegisterMyNew auto_reg_my_new;
```

## 技术栈

- **C++17** + pybind11 3.0
- **Python 3.12+** + numpy
- **CMake 3.28+**
- **Chrome Trace Event Format** — Perfetto 兼容

## License

MIT
