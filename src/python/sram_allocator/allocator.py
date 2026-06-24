"""High-level Python wrapper — Strategy Pattern Edition."""

from dataclasses import dataclass, field
from typing import Optional, Dict, List

from ._core import (
    SRAMAllocator as _CPPAllocator,
    DAG, OpNode, TensorLifetime,
    LinearScanStrategy, BestFitStrategy, LargestFirstStrategy,
    AllocationStrategy, PerfettoTracer,
    register_strategy, create_strategy,
)

# Export strategies
__all__ = [
    "SRAMAllocator", "AllocationResult",
    "LinearScanStrategy", "BestFitStrategy", "LargestFirstStrategy",
    "PerfettoTracer",
    "register_strategy", "create_strategy",
]


@dataclass
class OverflowViolation:
    """Single memory limit violation."""
    time: int
    active_memory: int
    limit: int
    excess: int
    live_tensors: List[str]


@dataclass
class AllocationResult:
    """Python-friendly allocation result."""
    success: bool
    error_msg: str
    peak_memory: int
    total_without_reuse: int
    reuse_ratio: float
    num_reuses: int
    tensor_offsets: Dict[str, int]
    tensor_sizes: Dict[str, int]
    timeline: list = field(default_factory=list)
    overflow: bool = False
    violations: List[OverflowViolation] = field(default_factory=list)
    _cpp_result: object = field(default=None, repr=False)

    def summary(self) -> str:
        if not self.success:
            return f"Allocation FAILED: {self.error_msg}"
        lines = [
            f"Peak SRAM: {self._fmt(self.peak_memory)}",
            f"Total (no reuse): {self._fmt(self.total_without_reuse)}",
            f"Reuse ratio: {self.reuse_ratio:.1%}",
            f"Reused tensors: {self.num_reuses}",
            f"Tensor count: {len(self.tensor_offsets)}",
        ]
        if self.overflow:
            lines.append(f"⚠️  OVERFLOW! Peak exceeded limit by {self._fmt(self.violations[-1].excess) if self.violations else 'N/A'}")
            lines.append(f"   Violations at {len(self.violations)} time point(s)")
        return "\n".join(lines)

    @staticmethod
    def _fmt(size: int) -> str:
        if size >= 1024 * 1024:
            return f"{size / (1024*1024):.2f} MB"
        elif size >= 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size} B"


class SRAMAllocator:
    """SRAM memory allocator with pluggable strategies.

    Usage:
        # Default: Linear Scan (First-Fit)
        alloc = SRAMAllocator()
        result = alloc.allocate(dag)

        # Switch strategy
        alloc.set_strategy(BestFitStrategy())
        result = alloc.allocate(dag)

        # With memory limit + Perfetto trace
        tracer = PerfettoTracer()
        alloc = SRAMAllocator(max_memory=1024*1024, tracer=tracer)
        result = alloc.allocate(dag)
        tracer.save("trace.json")  # open in perfetto ui
    """

    def __init__(
        self,
        strategy: Optional[AllocationStrategy] = None,
        alignment: int = 16,
        sram_size: Optional[int] = None,
        max_memory: Optional[int] = None,
        verbose: bool = False,
        tracer: Optional[PerfettoTracer] = None,
    ):
        self._cpp = _CPPAllocator()
        if strategy is not None:
            self._cpp.set_strategy(strategy)
        self._cpp.set_alignment(alignment)
        if sram_size is not None:
            self._cpp.set_sram_size(sram_size)
        if max_memory is not None:
            self._cpp.set_max_memory(max_memory)
        self._cpp.set_verbose(verbose)
        if tracer is not None:
            self._cpp.set_tracer(tracer)
        self._tracer = tracer

    def set_strategy(self, strategy: AllocationStrategy):
        """Set allocation strategy by instance."""
        self._cpp.set_strategy(strategy)

    def set_strategy_by_name(self, name: str):
        """Set allocation strategy by registered name.

        Built-in names:
            - 'linear_scan'   (default, First-Fit)
            - 'best_fit'      (Best-Fit)
            - 'largest_first' (Sort by size descending)
        """
        strategy = create_strategy(name)
        if strategy is None:
            raise ValueError(f"Unknown strategy: '{name}'. "
                             f"Available: linear_scan, best_fit, largest_first")
        self._cpp.set_strategy(strategy)

    def get_strategy_name(self) -> str:
        """Get current strategy name."""
        s = self._cpp.get_strategy()
        return s.name() if s else "None"

    def allocate(self, dag: DAG) -> AllocationResult:
        """Allocate SRAM for a neural network DAG."""
        cpp_result = self._cpp.allocate(dag)
        return self._wrap_result(cpp_result)

    def allocate_from_tensors(self, tensors: list) -> AllocationResult:
        """Allocate SRAM from a list of TensorLifetime objects."""
        cpp_result = self._cpp.allocate_from_tensors(tensors)
        return self._wrap_result(cpp_result)

    def save_perfetto_trace(self, path: str):
        """Save Perfetto trace JSON to file.

        Requires a PerfettoTracer to have been set at construction.
        """
        if self._tracer is None:
            raise RuntimeError(
                "No PerfettoTracer set. Create one and pass it as "
                "tracer=PerfettoTracer() to SRAMAllocator()"
            )
        self._tracer.save(path)

    def _wrap_result(self, cpp_result) -> AllocationResult:
        # Parse violations
        violations = []
        if cpp_result.overflow:
            report = cpp_result.overflow_report
            for v in report.violations:
                violations.append(OverflowViolation(
                    time=v.time,
                    active_memory=v.active_memory,
                    limit=v.limit,
                    excess=v.excess,
                    live_tensors=list(v.live_tensors),
                ))

        result = AllocationResult(
            success=cpp_result.success,
            error_msg=cpp_result.error_msg,
            peak_memory=cpp_result.peak_memory,
            total_without_reuse=cpp_result.total_tensor_bytes,
            reuse_ratio=cpp_result.reuse_ratio,
            num_reuses=cpp_result.num_reuses,
            tensor_offsets=dict(cpp_result.tensor_offsets),
            tensor_sizes=dict(cpp_result.tensor_sizes),
            timeline=list(cpp_result.timeline),
            overflow=cpp_result.overflow,
            violations=violations,
        )
        result._cpp_result = cpp_result
        return result
