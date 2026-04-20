"""Visualization and reporting for SRAM allocation results."""


def print_report(result) -> str:
    """Print a human-readable allocation report."""
    lines = []
    lines.append("=" * 60)
    lines.append("  SRAM Allocation Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append(result.summary())
    lines.append("")
    lines.append("-" * 60)
    lines.append("  Tensor Layout")
    lines.append("-" * 60)
    lines.append(f"  {'Tensor':<15} {'Offset':>10} {'Size':>10} {'Lifetime':>12}")
    lines.append(f"  {'─'*15} {'─'*10} {'─'*10} {'─'*12}")

    cpp_result = result._cpp_result
    if cpp_result is not None:
        for name, offset in result.tensor_offsets.items():
            size = result.tensor_sizes.get(name, 0)
            if name in cpp_result.tensor_lifetimes:
                lt = cpp_result.tensor_lifetimes[name]
                lifetime = f"[{lt.alive_from},{lt.alive_to}]"
            else:
                lifetime = ""
            lines.append(f"  {name:<15} {offset:>10} {size:>10} {lifetime:>12}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("  Memory Timeline")
    lines.append("-" * 60)

    if cpp_result is not None and cpp_result.timeline:
        for snap in cpp_result.timeline:
            bar_len = min(40, snap.active_memory * 40 // max(result.peak_memory, 1))
            bar = "█" * bar_len + "░" * (40 - bar_len)
            lines.append(
                f"  t={snap.time:3d} {snap.event:>5} {snap.tensor_name:<12} "
                f"mem={snap.active_memory:>8}B  {bar}"
            )

    lines.append("")
    report = "\n".join(lines)
    print(report)
    return report


def visualize_allocation(result, output_path: str = None) -> str:
    """Generate ASCII memory layout visualization."""
    if not result.success:
        return f"Allocation failed: {result.error_msg}"

    lines = []
    lines.append("SRAM Memory Layout")
    lines.append("=" * 50)

    # Sort tensors by offset
    sorted_tensors = sorted(result.tensor_offsets.items(), key=lambda x: x[1])

    for name, offset in sorted_tensors:
        size = result.tensor_sizes.get(name, 0)
        lifetime = ""
        cpp_result = result._cpp_result
        if cpp_result and name in cpp_result.tensor_lifetimes:
            lt = cpp_result.tensor_lifetimes[name]
            lifetime = f" [{lt.alive_from},{lt.alive_to}]"

        # Bar representation (scaled to 40 chars)
        scale = 40 / max(result.peak_memory, 1)
        start_char = int(offset * scale)
        end_char = int((offset + size) * scale)

        prefix = " " * start_char
        bar = "█" * max(1, end_char - start_char)

        lines.append(f"  {prefix}{bar} {name} ({size}B){lifetime}")

    # Summary
    lines.append("")
    lines.append(f"  Total allocated: {_fmt(result.peak_memory)}")
    lines.append(f"  Peak usage:      {_fmt(result.peak_memory)}")
    lines.append(f"  Without reuse:   {_fmt(result.total_without_reuse)}")
    lines.append(f"  Savings:         {result.reuse_ratio:.1%}")

    output = "\n".join(lines)
    if output_path:
        with open(output_path, "w") as f:
            f.write(output)
    print(output)
    return output


def _fmt(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024*1024):.2f} MB"
    elif size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"
