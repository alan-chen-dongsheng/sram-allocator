# Repository Guidelines

Contributor guide for **sram-allocator** — a strategy-pattern SRAM memory allocator for neural-network DAGs, implemented in C++17 and exposed to Python via pybind11.

## Project Structure & Module Organization

- `src/cpp/` — C++ core: `linear_scan.h` (interfaces, data structs, `SRAMAllocator` context, `PerfettoTracer`), `linear_scan.cpp` (three strategies + tracer + overflow detection), `bindings.cpp` (pybind11 bindings).
- `src/python/sram_allocator/` — high-level Python API: `allocator.py`, `builder.py` (`DAGBuilder`), `visualizer.py`, `__init__.py`.
- `test.py` — five functional scenarios; `test_strategies.py` — strategy comparison + isolation.
- `CMakeLists.txt`, `pyproject.toml` — build/packaging config; `build/` holds artifacts (gitignored).

## Build, Test, and Development Commands

Install (builds the C++ extension via scikit-build-core + Ninja + ccache):

```bash
pip install -e . --break-system-packages
```

Manual C++-only build:

```bash
mkdir -p build && cd build && cmake .. && cmake --build . -- -j$(nproc)
cp build/*.so src/python/sram_allocator/
```

Run tests:

- `python3 test.py` — five basic scenarios.
- `python3 test_strategies.py` — strategy comparison + isolation.

## Coding Style & Naming Conventions

- C++17, compiled with `-O3 -Wall`, GNU extensions off. Core types live in the `sram` namespace; `PascalCase` for classes/structs, `snake_case` for fields and methods.
- Python 3.10+: 4-space indentation, type hints, `@dataclass` for result/struct types, `snake_case` naming.
- `CMAKE_EXPORT_COMPILE_COMMANDS` is on; `compile_commands.json` feeds clangd/LSP — keep it in sync when editing CMake.

## Testing Guidelines

Tests are standalone scripts with `test_*` functions invoked via `if __name__ == "__main__"` (also pytest-compatible). Each test builds a DAG with `DAGBuilder`, allocates via `SRAMAllocator`, prints a report, and asserts on `result.success` / `result.reuse_ratio`. Add scenarios by extending `DAGBuilder` presets or the `make_tensor` helpers in `test.py`.

## Architecture & Adding Strategies

The allocator follows the Strategy pattern. New strategies implement `AllocationStrategy::allocate()` and self-register through a static `AutoRegister*` struct so they're reachable via `set_strategy_by_name()`. Strategies must call `config.tracer->on_alloc/on_free` during allocation and `on_peak/on_overflow` in post-processing to keep trace and overflow reporting consistent.

## Commit & Pull Request Guidelines

Follow Conventional Commits (e.g. `feat: ...`, `build: ...`); keep the subject short and imperative. Reference linked issues in the PR description and include before/after peak-memory or reuse-ratio figures when changing allocator behavior.
