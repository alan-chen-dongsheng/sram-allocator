"""DAG builder — construct neural network computation graphs."""

import math
from typing import Optional, List, Dict, Tuple

from ._core import DAG, OpNode


class DAGBuilder:
    """Helper to construct neural network DAGs for SRAM allocation."""

    def __init__(self, name: str = "network"):
        self.dag = DAG()
        self.dag.name = name
        self._ops = []  # Python list for building

    def add_op(
        self,
        name: str,
        op_type: str,
        inputs: List[str],
        output_size: int,
        depends_on: Optional[List[str]] = None,
    ) -> str:
        """Add an operator node to the DAG."""
        node = OpNode()
        node.name = name
        node.op_type = op_type
        node.inputs = inputs
        node.output = name
        node.output_size = output_size
        node.depends_on = depends_on or []
        self._ops.append(node)
        self.dag.ops = self._ops  # Reassign to C++ vector
        return name

    def add_conv(
        self, name: str, input_tensor: str, out_channels: int,
        h: int, w: int, depends_on: Optional[List[str]] = None,
    ) -> str:
        size = out_channels * h * w * 4
        return self.add_op(name, "Conv", [input_tensor], size, depends_on)

    def add_matmul(
        self, name: str, input_a: str, input_b: str, m: int, n: int,
        depends_on: Optional[List[str]] = None,
    ) -> str:
        size = m * n * 4
        return self.add_op(name, "MatMul", [input_a, input_b], size, depends_on)

    def add_add(
        self, name: str, input_a: str, input_b: str, output_size: int,
        depends_on: Optional[List[str]] = None,
    ) -> str:
        return self.add_op(name, "Add", [input_a, input_b], output_size, depends_on)

    def add_relu(self, name: str, input_tensor: str, size: int,
                 depends_on: Optional[List[str]] = None) -> str:
        return self.add_op(name, "Relu", [input_tensor], size, depends_on)

    def add_pool(self, name: str, input_tensor: str, out_size: int,
                 depends_on: Optional[List[str]] = None) -> str:
        return self.add_op(name, "Pool", [input_tensor], out_size, depends_on)

    def add_reshape(self, name: str, input_tensor: str, out_size: int,
                    depends_on: Optional[List[str]] = None) -> str:
        return self.add_op(name, "Reshape", [input_tensor], out_size, depends_on)

    def add_concat(self, name: str, input_a: str, input_b: str, out_size: int,
                   depends_on: Optional[List[str]] = None) -> str:
        return self.add_op(name, "Concat", [input_a, input_b], out_size, depends_on)

    @classmethod
    def build_resnet_block(cls, name: str, in_channels: int, out_channels: int,
                           h: int, w: int) -> 'DAGBuilder':
        """ResNet bottleneck block with skip connection."""
        b = cls(f"resnet_block_{name}")
        out_size = out_channels * h * w * 4

        b.add_conv("conv1", "input", out_channels, h, w)
        b.add_relu("relu1", "conv1", out_size)
        b.add_conv("conv2", "relu1", out_channels, h, w)
        b.add_relu("relu2", "conv2", out_size)
        b.add_conv("conv3", "relu2", out_channels, h, w)
        b.add_add("add", "conv3", "input", out_size)
        b.add_relu("relu_out", "add", out_size)
        return b

    @classmethod
    def build_simple_cnn(cls, name: str = "simple_cnn") -> 'DAGBuilder':
        """Simple CNN: Conv→Relu→Pool→Conv→Relu→FC."""
        b = cls(name)
        b.add_conv("conv1", "input", 16, 32, 32)
        b.add_relu("relu1", "conv1", 16*32*32*4)
        b.add_pool("pool1", "relu1", 16*16*16*4)
        b.add_conv("conv2", "pool1", 32, 16, 16)
        b.add_relu("relu2", "conv2", 32*16*16*4)
        b.add_pool("pool2", "relu2", 32*8*8*4)
        b.add_reshape("flatten", "pool2", 32*8*8*4)
        b.add_matmul("fc1", "flatten", "fc1_w", 2048, 128)
        b.add_relu("relu3", "fc1", 128*4)
        b.add_matmul("fc2", "relu3", "fc2_w", 128, 10)
        return b

    @classmethod
    def build_linear_chain(cls, name: str = "chain", num_ops: int = 10,
                           tensor_size: int = 1024) -> 'DAGBuilder':
        """Linear chain: each op depends on previous (perfect reuse)."""
        b = cls(name)
        current_input = "input"
        prev_op = None

        for i in range(num_ops):
            out_name = f"op{i}"
            deps = [prev_op] if prev_op else []
            b.add_op(out_name, "MatMul", [current_input], tensor_size, deps)
            current_input = out_name
            prev_op = out_name
        return b

    @classmethod
    def build_fork_join(cls, name: str = "fork_join", branch_size: int = 4096) -> 'DAGBuilder':
        """Fork-join: two parallel branches, tests concurrent memory."""
        b = cls(name)
        b.add_op("fork", "Split", ["input"], branch_size)
        b.add_op("branch1", "MatMul", ["fork"], branch_size, depends_on=["fork"])
        b.add_op("branch2", "MatMul", ["fork"], branch_size, depends_on=["fork"])
        b.add_op("join", "Concat", ["branch1", "branch2"], branch_size * 2,
                 depends_on=["branch1", "branch2"])
        return b
