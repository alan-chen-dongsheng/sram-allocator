"""SRAM Allocator — Strategy Pattern Edition"""

from .allocator import (
    SRAMAllocator,
    AllocationResult,
    AllocationStrategy,
    LinearScanStrategy,
    BestFitStrategy,
    LargestFirstStrategy,
    PerfettoTracer,
    register_strategy,
    create_strategy,
)
from ._core import DAG, OpNode, TensorLifetime
from .builder import DAGBuilder
from .visualizer import visualize_allocation, print_report

__version__ = "0.3.0"
__all__ = [
    "SRAMAllocator", "AllocationResult", "AllocationStrategy",
    "LinearScanStrategy", "BestFitStrategy", "LargestFirstStrategy",
    "PerfettoTracer",
    "register_strategy", "create_strategy",
    "DAG", "OpNode", "TensorLifetime",
    "DAGBuilder", "visualize_allocation", "print_report",
]
