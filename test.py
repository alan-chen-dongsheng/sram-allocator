#!/usr/bin/env python3
"""Test cases for SRAM linear scan allocator."""

import os

from sram_allocator import SRAMAllocator, DAGBuilder, visualize_allocation, print_report


def test_linear_chain():
    print("\n" + "=" * 60)
    print("  Test 1: Linear Chain (完美复用场景)")
    print("=" * 60)
    dag = DAGBuilder.build_linear_chain("chain", num_ops=10, tensor_size=4096)
    alloc = SRAMAllocator(alignment=16, verbose=False)
    result = alloc.allocate(dag.dag)
    print_report(result)
    visualize_allocation(result)
    assert result.success
    assert result.reuse_ratio >= 0.8, f"Expected >=80% reuse, got {result.reuse_ratio:.1%}"
    print("  ✓ PASS")


def test_simple_cnn():
    print("\n" + "=" * 60)
    print("  Test 2: Simple CNN")
    print("=" * 60)
    dag = DAGBuilder.build_simple_cnn("simple_cnn")
    alloc = SRAMAllocator(alignment=64, verbose=False)
    result = alloc.allocate(dag.dag)
    print_report(result)
    visualize_allocation(result)
    assert result.success
    print(f"  Peak SRAM: {result.peak_memory / 1024:.1f} KB")
    print("  ✓ PASS")


def test_resnet_block():
    print("\n" + "=" * 60)
    print("  Test 3: ResNet Block (skip connection)")
    print("=" * 60)
    dag = DAGBuilder.build_resnet_block("block1", 64, 64, 56, 56)
    alloc = SRAMAllocator(alignment=256, verbose=True)
    result = alloc.allocate(dag.dag)
    print_report(result)
    visualize_allocation(result)
    assert result.success
    print(f"  Peak SRAM: {result.peak_memory / (1024*1024):.2f} MB")
    print("  ✓ PASS")


def test_fork_join():
    print("\n" + "=" * 60)
    print("  Test 4: Fork-Join Pattern")
    print("=" * 60)
    dag = DAGBuilder.build_fork_join("fork_join", branch_size=8192)
    alloc = SRAMAllocator(alignment=16, verbose=True)
    result = alloc.allocate(dag.dag)
    print_report(result)
    visualize_allocation(result)
    assert result.success
    print("  ✓ PASS")


def test_custom_tensors():
    print("\n" + "=" * 60)
    print("  Test 5: Direct Tensor List")
    print("=" * 60)

    from sram_allocator._core import TensorLifetime

    def make_tensor(name, size, producer, consumers, alive_from, alive_to):
        t = TensorLifetime()
        t.name = name
        t.size = size
        t.producer = producer
        t.consumers = consumers
        t.alive_from = alive_from
        t.alive_to = alive_to
        return t

    tensors = [
        make_tensor("t0", 1024, "op0", ["op2"], 0, 2),
        make_tensor("t1", 2048, "op1", ["op3"], 1, 3),
        make_tensor("t2", 512, "op2", ["op4"], 2, 4),
        make_tensor("t3", 1024, "op3", ["op5"], 3, 5),
    ]

    alloc = SRAMAllocator(alignment=16, verbose=True)
    result = alloc.allocate_from_tensors(tensors)
    print_report(result)
    visualize_allocation(result)

    assert result.success
    print("  ✓ PASS")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    test_linear_chain()
    test_simple_cnn()
    test_resnet_block()
    test_fork_join()
    test_custom_tensors()
    print("\n" + "=" * 60)
    print("  All tests passed! ✓")
    print("=" * 60)
