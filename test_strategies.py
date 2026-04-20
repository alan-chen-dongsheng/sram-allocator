#!/usr/bin/env python3
"""Strategy pattern tests — 3 built-in strategies across 4 scenarios."""

import os

from sram_allocator import (
    SRAMAllocator, DAGBuilder, print_report, visualize_allocation,
    LinearScanStrategy, BestFitStrategy, LargestFirstStrategy,
)

STRATEGIES = {
    "Linear Scan": LinearScanStrategy(),
    "Best Fit": BestFitStrategy(),
    "Largest First": LargestFirstStrategy(),
}

SCENARIOS = {
    "Linear Chain": lambda: DAGBuilder.build_linear_chain("chain", 10, 4096),
    "Simple CNN": lambda: DAGBuilder.build_simple_cnn(),
    "ResNet Block": lambda: DAGBuilder.build_resnet_block("b1", 64, 64, 56, 56),
    "Fork-Join": lambda: DAGBuilder.build_fork_join("fj", 8192),
}


def test_strategy_comparison():
    """所有策略 × 所有场景的对比矩阵。"""
    print("=" * 70)
    print("  Strategy Comparison Matrix")
    print("=" * 70)

    print(f"\n{'Scenario':<18}", end="")
    for name in STRATEGIES:
        print(f"  {name:<18}", end="")
    print()
    print(f"{'─' * 18}", end="")
    for _ in STRATEGIES:
        print(f"  {'─' * 18}", end="")
    print()

    for scenario_name, dag_factory in SCENARIOS.items():
        print(f"{scenario_name:<18}", end="")
        for strat_name, strategy in STRATEGIES.items():
            alloc = SRAMAllocator(strategy=strategy, alignment=16)
            result = alloc.allocate(dag_factory().dag)
            label = f"{result.peak_memory / 1024:.0f}KB ({result.reuse_ratio:.0%})"
            print(f"  {label:<18}", end="")
        print()

    print("\n  ✓ Comparison matrix complete")


def test_strategy_switching():
    """运行时切换策略。"""
    print("\n" + "=" * 70)
    print("  Strategy Switching Demo")
    print("=" * 70)

    alloc = SRAMAllocator()
    dag = DAGBuilder.build_simple_cnn().dag

    for name in ["linear_scan", "best_fit", "largest_first"]:
        alloc.set_strategy_by_name(name)
        current = alloc.get_strategy_name()
        result = alloc.allocate(dag)
        print(f"  [{current}] peak={result.peak_memory / 1024:.0f}KB, "
              f"reuse={result.reuse_ratio:.0%}, reuses={result.num_reuses}")

    print("\n  ✓ Strategy switching works")


def test_strategy_isolation():
    """不同策略独立运行。"""
    print("\n" + "=" * 70)
    print("  Strategy Isolation Test")
    print("=" * 70)

    dag = DAGBuilder.build_resnet_block("b1", 64, 64, 56, 56)

    for strat_name, strategy in STRATEGIES.items():
        print(f"\n--- {strat_name} ---")
        alloc = SRAMAllocator(strategy=strategy, alignment=256, verbose=True)
        result = alloc.allocate(dag.dag)
        print_report(result)
        assert result.success, f"Strategy {strat_name} failed"

    print("\n  ✓ All strategies produce valid results")


def test_strategy_by_constructor():
    """通过构造函数指定策略。"""
    print("\n" + "=" * 70)
    print("  Constructor Strategy Test")
    print("=" * 70)

    dag = DAGBuilder.build_fork_join("fj", 8192).dag

    for cls in [LinearScanStrategy, BestFitStrategy, LargestFirstStrategy]:
        alloc = SRAMAllocator(strategy=cls(), alignment=16)
        sname = alloc.get_strategy_name()
        assert len(sname) > 0, "Strategy name should not be empty"
        result = alloc.allocate(dag)
        print(f"  [{alloc.get_strategy_name()}] peak={result.peak_memory / 1024:.0f}KB, "
              f"reuse={result.reuse_ratio:.0%}")
        assert result.success

    print("\n  ✓ Constructor strategy injection works")


def test_detailed_view():
    """单个场景的详细可视化。"""
    print("\n" + "=" * 70)
    print("  Detailed View: Linear Chain (best reuse scenario)")
    print("=" * 70)

    dag = DAGBuilder.build_linear_chain("chain", 10, 4096)

    for strat_name, strategy in STRATEGIES.items():
        print(f"\n--- {strat_name} ---")
        alloc = SRAMAllocator(strategy=strategy, alignment=16)
        result = alloc.allocate(dag.dag)
        visualize_allocation(result)

    print("\n  ✓ Detailed view complete")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_strategy_comparison()
    test_strategy_switching()
    test_strategy_isolation()
    test_strategy_by_constructor()
    test_detailed_view()

    print("\n" + "=" * 70)
    print("  All strategy tests passed! ✓")
    print("=" * 70)
