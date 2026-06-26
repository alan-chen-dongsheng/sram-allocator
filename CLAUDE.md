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
# All tests in a file
python3 test.py              # 5 basic functional scenarios
python3 test_strategies.py   # Strategy comparison + isolation tests

# A single test function (scripts are pytest-compatible)
python3 -m pytest test.py::test_linear_chain -v
# Or without pytest:
python3 -c "from test import test_linear_chain; test_linear_chain()"
```

Tests are standalone scripts with `test_*` functions. Each builds a DAG with `DAGBuilder`, allocates via `SRAMAllocator`, and asserts on `result.success` / `result.reuse_ratio`.

## Architecture

### Two-layer design

**C++ layer** (`src/cpp/`):
- `linear_scan.h` — Core interfaces, data structures, `SRAMAllocator` context, `PerfettoTracer`
- `linear_scan.cpp` — Three allocation strategies + tracer + overflow detection
- `bindings.cpp` — pybind11 bindings exposing C++ to Python

**Python layer** (`src/python/sram_allocator/`):
- `allocator.py` — High-level `SRAMAllocator` API wrapping C++ core, plus `AllocationResult` / `OverflowViolation` dataclasses
- `builder.py` — `DAGBuilder` for constructing computation graphs
- `visualizer.py` — `print_report()` and `visualize_allocation()` ASCII reporting

### Strategy pattern

Three allocation strategies implement `AllocationStrategy::allocate()`:

| Strategy | Class | Registered name | Behavior |
|---|---|---|---|
| Linear Scan | `LinearScanStrategy` | `linear_scan` | First-fit allocation (default) |
| Best Fit | `BestFitStrategy` | `best_fit` | Minimizes fragmentation |
| Largest First | `LargestFirstStrategy` | `largest_first` | Allocates largest tensors first |

Strategies self-register via static `AutoRegister*` structs, making them accessible through `set_strategy_by_name()`.

### Two allocation entry points

- `alloc.allocate(dag)` — takes a `DAG`, internally runs topological sort + lifetime computation, then allocates.
- `alloc.allocate_from_tensors(tensors)` — takes a pre-built `list[TensorLifetime]`, bypasses DAG construction. Used when lifetimes are known externally.

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

4. Rebuild (`pip install -e .`) and use via `alloc.set_strategy_by_name("my_strategy")` or from Python with `register_strategy("my_strategy", lambda: MyCustomStrategy())`.

### Key data structures

- `TensorLifetime` — Tensor lifecycle: name, size, producer, consumers, alive_from, alive_to
- `OpNode` — Operator node: name, op_type, inputs, output, output_size, depends_on
- `DAG` — Computation graph: name + list of ops
- `AllocationResult` — Allocation output (Python dataclass): `tensor_offsets`, `tensor_sizes`, `peak_memory`, `total_without_reuse`, `reuse_ratio`, `num_reuses`, `timeline`, `overflow`, `violations`
- `OverflowViolation` — Single overflow point: `time`, `active_memory`, `limit`, `excess`, `live_tensors`
- `TimelineSnapshot` — Per-event memory snapshot: `time`, `event` (string: "alloc"/"free"/"peak"), `tensor_name`, `peak_memory`, `active_memory`, `live_tensors`

C++-only types (not exposed to Python): `AllocConfig`, `MemoryBlock`, `OverflowReport`/`OverflowEvent` (mapped to `OverflowViolation` in Python).

### Key Python API

```python
from sram_allocator import (
    SRAMAllocator, DAGBuilder, PerfettoTracer,
    AllocationStrategy, AllocationResult,
    LinearScanStrategy, BestFitStrategy, LargestFirstStrategy,
    register_strategy, create_strategy,
    visualize_allocation, print_report,
)

# Constructor options
alloc = SRAMAllocator(
    strategy=BestFitStrategy(),   # optional, defaults to LinearScan
    alignment=64,                 # byte alignment; rounds each tensor's SIZE up to this multiple, inflating memory usage
    max_memory=200 * 1024,        # peak limit; violations recorded, alloc continues
    verbose=True,
    tracer=PerfettoTracer(),      # optional; enables Chrome Trace output
)

# DAG builder presets
DAGBuilder.build_simple_cnn(name)
DAGBuilder.build_resnet_block(name, in_ch, out_ch, h, w)
DAGBuilder.build_linear_chain(name, num_ops, tensor_size)
DAGBuilder.build_fork_join(name, branch_size)

# DAG builder primitives
builder = DAGBuilder("net")
builder.add_op(name, op_type, inputs, output_size, depends_on)
builder.add_conv / add_matmul / add_add / add_relu / add_pool / add_reshape / add_concat

# Custom Python strategy
class MyStrategy(AllocationStrategy):
    def name(self): return "my_strategy"
    def allocate(self, tensors, config):
        result = AllocationResult(success=True, error_msg="", ...)
        # populate tensor_offsets, tensor_sizes, peak_memory, etc.
        return result

register_strategy("my_strategy", lambda: MyStrategy())
alloc.set_strategy_by_name("my_strategy")

# Perfetto trace: open https://ui.perfetto.dev/ and drag in the JSON
alloc.save_perfetto_trace("trace.json")
```

## Coding Conventions

**C++**: C++17, compiled with `-O3 -Wall`, no GNU extensions. Core types in `sram` namespace. `PascalCase` for classes/structs, `snake_case` for fields/methods. `CMAKE_EXPORT_COMPILE_COMMANDS` is on — `compile_commands.json` is generated for clangd/LSP; re-run cmake after CMakeLists.txt changes to keep it in sync.

**Python**: 3.10+, 4-space indent, type hints, `@dataclass` for result/struct types, `snake_case` naming.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'sram_allocator._core'`** — the C++ extension hasn't been built. Run `pip install -e .` or the manual build steps above and ensure `*.so` files exist under `src/python/sram_allocator/` or in the installed package.
- **Stale `compile_commands.json`** — re-run `cmake ..` in `build/` after editing `CMakeLists.txt` or adding source files.
- **Strategy name not found** — built-in names are `linear_scan`, `best_fit`, `largest_first`. Custom strategies must be registered before `set_strategy_by_name()` is called.
- **Overflow not reported** — `max_memory` defaults to `-1` (disabled). Pass a positive value to enable; violations are recorded but allocation still completes.
- **Trace file empty** — a `PerfettoTracer` must be passed to the `SRAMAllocator` constructor *before* `allocate()` is called; events are not backfilled.
