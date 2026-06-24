# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Strategy-pattern SRAM memory allocator for neural network DAGs. C++17 core exposed to Python via pybind11. Given a computation graph, automatically calculates memory offsets for intermediate tensors and minimizes peak SRAM usage through memory reuse.

## Build and Test Commands

### Install (builds C++ extension via scikit-build-core)

```bash
pip install -e . --break-system-packages
```

This uses scikit-build-core + Ninja + ccache. The build produces:
- `build/libsram_alloc.so` — C++ shared library
- `build/_core.cpython-*.so` — pybind11 Python extension

### Manual C++ build

```bash
mkdir -p build && cd build
cmake .. && cmake --build . -- -j$(nproc)
cp build/*.so src/python/sram_allocator/
```

### Run tests

```bash
python3 test.py              # 5 basic functional scenarios
python3 test_strategies.py   # Strategy comparison + isolation tests
```

Tests are standalone scripts with `test_*` functions. Each builds a DAG with `DAGBuilder`, allocates via `SRAMAllocator`, and asserts on `result.success` / `result.reuse_ratio`.

## Architecture

### Two-layer design

**C++ layer** (`src/cpp/`):
- `linear_scan.h` — Core interfaces, data structures, `SRAMAllocator` context, `PerfettoTracer`
- `linear_scan.cpp` — Three allocation strategies + tracer + overflow detection
- `bindings.cpp` — pybind11 bindings exposing C++ to Python

**Python layer** (`src/python/sram_allocator/`):
- `allocator.py` — High-level `SRAMAllocator` API wrapping C++ core
- `builder.py` — `DAGBuilder` for constructing computation graphs
- `visualizer.py` — Reporting and visualization utilities

### Strategy pattern

Three allocation strategies implement `AllocationStrategy::allocate()`:
- **LinearScan** — First-fit allocation
- **BestFit** — Minimizes fragmentation
- **LargestFirst** — Allocates largest tensors first

Strategies self-register via static `AutoRegister*` structs, making them accessible through `set_strategy_by_name()`.

### Adding a new strategy

1. Declare in `linear_scan.h`:
```cpp
class MyStrategy : public AllocationStrategy {
public:
    std::string name() const override;
    AllocationResult allocate(
        const std::vector<TensorLifetime>& tensors,
        const AllocConfig& config) const override;
};
```

2. Implement in `linear_scan.cpp`. Must call `config.tracer->on_alloc/on_free` during allocation and `on_peak/on_overflow` in post-processing.

3. Auto-register:
```cpp
struct AutoRegisterMy { AutoRegisterMy() {
    SRAMAllocator::register_strategy("my_strategy",
        []() { return std::make_shared<MyStrategy>(); });
}};
static AutoRegisterMy auto_reg_my;
```

### Key data structures

- `TensorLifetime` — Tensor lifecycle: name, size, alive_from, alive_to
- `OpNode` — Operator node: name, op_type, inputs, output, depends_on
- `DAG` — Computation graph: list of ops
- `AllocationResult` — Allocation output: offsets, sizes, peak_memory, timeline, overflow_report
- `OverflowReport` — Violations list with time, active memory, limit, excess, live_tensors

## Coding Conventions

**C++**: C++17, compiled with `-O3 -Wall`, no GNU extensions. Core types in `sram` namespace. `PascalCase` for classes/structs, `snake_case` for fields/methods. `CMAKE_EXPORT_COMPILE_COMMANDS` is on for clangd/LSP.

**Python**: 3.10+, type hints, `@dataclass` for result/struct types, `snake_case` naming.

## Key features

- **Memory reuse** — Minimizes peak SRAM by reusing memory blocks
- **Overflow detection** — Set `max_memory` limit; violations reported but allocation continues
- **Perfetto tracing** — Optional Chrome Trace JSON output for visualization in perfetto UI
- **Zero-overhead optional** — `tracer=nullptr` means no tracing overhead; `max_memory=-1` disables overflow detection
