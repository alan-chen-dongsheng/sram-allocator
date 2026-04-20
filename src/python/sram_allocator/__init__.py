"""SRAM Allocator — Strategy Pattern Edition"""

from .allocator import (
    SRAMAllocator,
    AllocationResult,
    LinearScanStrategy,
    BestFitStrategy,
    LargestFirstStrategy,
    PerfettoTracer,
    register_strategy,
    create_strategy,
)
from .builder import DAGBuilder
from .visualizer import visualize_allocation, print_report

__version__ = "0.3.0"
__all__ = [
    "SRAMAllocator", "AllocationResult",
    "LinearScanStrategy", "BestFitStrategy", "LargestFirstStrategy",
    "PerfettoTracer",
    "register_strategy", "create_strategy",
    "DAGBuilder", "visualize_allocation", "print_report",
]
