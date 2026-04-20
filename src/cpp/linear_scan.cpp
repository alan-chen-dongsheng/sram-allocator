#include "linear_scan.h"
#include <algorithm>
#include <queue>
#include <iostream>
#include <numeric>
#include <functional>
#include <sstream>
#include <fstream>
#include <cmath>

namespace sram {

// ============================================================
//  PerfettoTracer 实现
// ============================================================

void PerfettoTracer::on_alloc(int time_step, const std::string& tensor,
                              int64_t offset, int64_t size) {
    events_.push_back({time_step * 1000, "B", "SRAM: " + tensor, "memory",
                       1, 1, size, offset, 0, 0, ""});
}

void PerfettoTracer::on_free(int time_step, const std::string& tensor,
                             int64_t offset, int64_t size) {
    // Close the duration event started by on_alloc
    events_.push_back({time_step * 1000, "E", "SRAM: " + tensor, "memory",
                       1, 1, 0, 0, 0, 0, ""});
    // Instant marker to make the free point visible
    events_.push_back({time_step * 1000, "i", "free: " + tensor, "memory",
                       1, 1, size, offset, 0, 0, "t"});
}

void PerfettoTracer::on_peak(int time_step, int64_t peak_bytes, int64_t active_bytes) {
    events_.push_back({time_step * 1000, "C", "PeakMemory", "memory",
                       1, 1, 0, 0, peak_bytes, 0, ""});
}

void PerfettoTracer::on_overflow(int time_step, int64_t peak_bytes, int64_t limit) {
    events_.push_back({time_step * 1000, "i", "OVERFLOW", "memory",
                       1, 1, 0, 0, peak_bytes, limit, "t"});
}

static std::string escape_json_string(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            default:   out += c;
        }
    }
    return out;
}

std::string PerfettoTracer::to_json() const {
    std::ostringstream j;
    j << "[\n";
    for (size_t i = 0; i < events_.size(); i++) {
        const auto& e = events_[i];
        j << "  {\"name\":\"" << escape_json_string(e.name)
          << "\",\"ph\":\"" << e.ph
          << "\",\"ts\":" << e.ts
          << ",\"pid\":" << e.pid
          << ",\"tid\":" << e.tid
          << ",\"cat\":\"" << escape_json_string(e.cat) << "\"";

        // args
        j << ",\"args\":{";
        bool first = true;
        if (e.size > 0) {
            j << "\"size\":" << e.size; first = false;
        }
        if (e.offset > 0 || e.ph == "B") {
            if (!first) j << ",";
            j << "\"offset\":" << e.offset; first = false;
        }
        if (e.bytes > 0) {
            if (!first) j << ",";
            j << "\"bytes\":" << e.bytes; first = false;
        }
        if (e.limit > 0) {
            if (!first) j << ",";
            j << "\"limit\":" << e.limit; first = false;
        }
        j << "}";

        // scope for instant events
        if (!e.s.empty()) {
            j << ",\"s\":\"" << e.s << "\"";
        }

        j << "}";
        if (i + 1 < events_.size()) j << ",";
        j << "\n";
    }
    j << "]\n";
    return j.str();
}

void PerfettoTracer::save(const std::string& path) const {
    std::ofstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("PerfettoTracer::save: cannot open file: " + path);
    }
    f << to_json();
    f.close();
}

// ============================================================
//  策略注册表（静态，全局）
// ============================================================
static std::map<std::string, SRAMAllocator::StrategyFactory>& strategy_registry() {
    static std::map<std::string, SRAMAllocator::StrategyFactory> reg;
    return reg;
}

void SRAMAllocator::register_strategy(const std::string& name, StrategyFactory factory) {
    strategy_registry()[name] = factory;
}

std::shared_ptr<AllocationStrategy> SRAMAllocator::create_strategy(const std::string& name) {
    auto& reg = strategy_registry();
    auto it = reg.find(name);
    if (it != reg.end()) return it->second();
    return nullptr;
}

// 自动注册内置策略
struct AutoRegister {
    AutoRegister() {
        SRAMAllocator::register_strategy("linear_scan",
            []() { return std::make_shared<LinearScanStrategy>(); });
        SRAMAllocator::register_strategy("best_fit",
            []() { return std::make_shared<BestFitStrategy>(); });
        SRAMAllocator::register_strategy("largest_first",
            []() { return std::make_shared<LargestFirstStrategy>(); });
    }
};
static AutoRegister auto_register;

// ============================================================
//  Context：SRAMAllocator
// ============================================================

SRAMAllocator::SRAMAllocator()
    : strategy_(std::make_shared<LinearScanStrategy>()) {}

void SRAMAllocator::set_strategy(std::shared_ptr<AllocationStrategy> strategy) {
    strategy_ = strategy;
}

std::shared_ptr<AllocationStrategy> SRAMAllocator::get_strategy() const {
    return strategy_;
}

void SRAMAllocator::set_alignment(int64_t align) { config_.alignment = align; }
void SRAMAllocator::set_sram_size(int64_t size) { config_.sram_size = size; }
void SRAMAllocator::set_max_memory(int64_t max) { config_.max_memory = max; }
void SRAMAllocator::set_verbose(bool v) { config_.verbose = v; }
void SRAMAllocator::set_tracer(PerfettoTracer* t) { config_.tracer = t; }

// ============================================================
//  辅助工具函数（策略内部使用）
// ============================================================

static int64_t align_up(int64_t size, int64_t alignment) {
    return (size + alignment - 1) / alignment * alignment;
}

// ===== 拓扑排序 =====
std::vector<std::string> SRAMAllocator::topo_sort(const DAG& dag) {
    std::map<std::string, std::vector<std::string>> adj;
    std::map<std::string, int> in_degree;
    std::set<std::string> all_ops;

    for (const auto& op : dag.ops) {
        all_ops.insert(op.name);
        in_degree[op.name];  // ensure entry exists (default-initialized to 0)
        for (const auto& dep : op.depends_on) {
            adj[dep].push_back(op.name);
            in_degree[op.name]++;
            if (!in_degree.count(dep)) in_degree[dep] = 0;
        }
    }

    std::queue<std::string> q;
    for (const auto& name : all_ops) {
        if (in_degree[name] == 0) q.push(name);
    }

    std::vector<std::string> result;
    while (!q.empty()) {
        std::string curr = q.front();
        q.pop();
        result.push_back(curr);
        for (const auto& next : adj[curr]) {
            in_degree[next]--;
            if (in_degree[next] == 0) q.push(next);
        }
    }

    if (result.size() != all_ops.size()) return {};
    return result;
}

// ===== 生命周期计算 =====
std::vector<TensorLifetime> SRAMAllocator::compute_lifetimes(
    const DAG& dag,
    const std::vector<std::string>& order) {

    std::map<std::string, TensorLifetime> lifetimes;
    std::map<std::string, int> op_order;
    for (int i = 0; i < (int)order.size(); i++) {
        op_order[order[i]] = i;
    }

    for (const auto& op : dag.ops) {
        int prod_idx = op_order.count(op.name) ? op_order[op.name] : 0;

        TensorLifetime lt;
        lt.name = op.output;
        lt.size = op.output_size;
        lt.producer = op.name;
        lt.consumers = {};
        lt.alive_from = prod_idx;
        lt.alive_to = prod_idx;
        lifetimes[op.output] = lt;
    }

    for (const auto& op : dag.ops) {
        int cons_idx = op_order.count(op.name) ? op_order[op.name] : 0;
        for (const auto& inp : op.inputs) {
            if (lifetimes.count(inp)) {
                lifetimes[inp].alive_to = std::max(lifetimes[inp].alive_to, cons_idx);
                lifetimes[inp].consumers.push_back(op.name);
            }
        }
    }

    std::vector<TensorLifetime> result;
    for (const auto& [name, lt] : lifetimes) {
        result.push_back(lt);
    }
    return result;
}

// ===== 分配：DAG → 结果 =====
AllocationResult SRAMAllocator::allocate(const DAG& dag) {
    auto order = topo_sort(dag);
    if (order.empty()) {
        AllocationResult r;
        r.success = false;
        r.error_msg = "Cycle detected in DAG";
        r.sched_order = order;
        return r;
    }

    auto tensors = compute_lifetimes(dag, order);
    auto result = allocate_from_tensors(tensors);
    result.sched_order = order;
    return result;
}

// ===== 分配：张量列表 → 结果（委托给策略）=====
AllocationResult SRAMAllocator::allocate_from_tensors(
    const std::vector<TensorLifetime>& tensors) {

    if (!strategy_) {
        AllocationResult r;
        r.success = false;
        r.error_msg = "No allocation strategy set";
        return r;
    }

    if (config_.verbose) {
        std::cout << "[SRAMAllocator] Using strategy: " << strategy_->name() << "\n";
    }

    return strategy_->allocate(tensors, config_);
}

// ============================================================
//  内部辅助函数（策略内部使用）
// ============================================================

// 计算给定时间点的活跃内存和存活张量列表
static void compute_active_at(
    int time_point,
    const std::map<std::string, TensorLifetime>& lifetimes,
    const std::map<std::string, int64_t>& sizes,
    int64_t& active_memory,
    std::vector<std::string>& live_tensors) {

    active_memory = 0;
    live_tensors.clear();
    for (const auto& [name, lt] : lifetimes) {
        if (lt.alive_from <= time_point && lt.alive_to >= time_point) {
            if (sizes.count(name)) {
                active_memory += sizes.at(name);
                live_tensors.push_back(name);
            }
        }
    }
}

// 后处理：构建 timeline、检测溢出、组装 AllocationResult（三种策略共享）
static AllocationResult finalize_alloc_result(
    std::map<std::string, int64_t>&& offsets,
    std::map<std::string, int64_t>&& sizes,
    std::map<std::string, TensorLifetime>&& lifetimes,
    int64_t total_without_reuse,
    int num_reuses,
    const std::vector<TensorLifetime>& sorted_tensors,
    const AllocConfig& config) {

    std::vector<TimelineSnapshot> timeline;
    std::vector<OverflowEvent> violations;
    int64_t global_peak = 0;

    std::set<int> time_points;
    for (const auto& t : sorted_tensors) {
        time_points.insert(t.alive_from);
        time_points.insert(t.alive_to);
    }

    for (int tp : time_points) {
        int64_t active_mem = 0;
        std::vector<std::string> live;
        compute_active_at(tp, lifetimes, sizes, active_mem, live);

        if (active_mem > global_peak) global_peak = active_mem;

        TimelineSnapshot snap;
        snap.time = tp;
        snap.event = "SNAPSHOT";
        snap.tensor_name = "";
        snap.peak_memory = global_peak;
        snap.active_memory = active_mem;
        snap.live_tensors = live;
        timeline.push_back(snap);

        if (config.max_memory > 0 && active_mem > config.max_memory) {
            OverflowEvent evt;
            evt.time = tp;
            evt.active_memory = active_mem;
            evt.limit = config.max_memory;
            evt.excess = active_mem - config.max_memory;
            evt.live_tensors = live;
            violations.push_back(evt);
        }

        if (config.tracer) {
            config.tracer->on_peak(tp, global_peak, active_mem);
            if (config.max_memory > 0 && active_mem > config.max_memory) {
                config.tracer->on_overflow(tp, active_mem, config.max_memory);
            }
        }
    }

    bool overflow = !violations.empty();

    AllocationResult result;
    result.success = true;
    result.tensor_offsets = std::move(offsets);
    result.tensor_sizes = std::move(sizes);
    result.tensor_lifetimes = std::move(lifetimes);
    result.peak_memory = global_peak;
    result.total_tensor_bytes = total_without_reuse;
    result.reuse_ratio = total_without_reuse > 0
        ? 1.0 - static_cast<double>(global_peak) / total_without_reuse
        : 0.0;
    result.num_reuses = num_reuses;
    result.timeline = std::move(timeline);
    result.overflow = overflow;

    if (overflow) {
        result.overflow_report.max_memory_limit = config.max_memory;
        result.overflow_report.peak_memory = global_peak;
        result.overflow_report.peak_excess = global_peak - config.max_memory;
        result.overflow_report.violations = std::move(violations);
    }

    return result;
}

// ============================================================
//  策略 1：Linear Scan (First-Fit)
// ============================================================

std::string LinearScanStrategy::name() const {
    return "LinearScan (First-Fit)";
}

AllocationResult LinearScanStrategy::allocate(
    const std::vector<TensorLifetime>& tensors,
    const AllocConfig& config) const {

    if (tensors.empty()) {
        AllocationResult r;
        r.success = true;
        return r;
    }

    auto sorted = tensors;
    std::sort(sorted.begin(), sorted.end(), [](const TensorLifetime& a, const TensorLifetime& b) {
        if (a.alive_from != b.alive_from) return a.alive_from < b.alive_from;
        return a.size > b.size;
    });

    struct ActiveTensor {
        std::string name;
        int64_t size;
        int64_t start;
        int alive_to;
    };
    std::vector<ActiveTensor> active;
    int64_t total_allocated = 0;
    int64_t total_without_reuse = 0;
    int num_reuses = 0;

    std::map<std::string, int64_t> offsets;
    std::map<std::string, int64_t> sizes;
    std::map<std::string, TensorLifetime> lifetimes;

    for (const auto& tensor : sorted) {
        int64_t aligned_size = align_up(tensor.size, config.alignment);
        total_without_reuse += aligned_size;

        // 释放过期张量，使用张量自身的到期时刻作为 free 时间
        for (auto it = active.begin(); it != active.end(); ) {
            if (it->alive_to < tensor.alive_from) {
                if (config.tracer) {
                    config.tracer->on_free(it->alive_to + 1, it->name, it->start, it->size);
                }
                it = active.erase(it);
            } else {
                ++it;
            }
        }

        // First-Fit：找第一个能容纳的空闲区间
        int64_t offset = -1;

        auto active_sorted = active;
        std::sort(active_sorted.begin(), active_sorted.end(),
            [](const ActiveTensor& a, const ActiveTensor& b) {
                return a.start < b.start;
            });

        int64_t cursor = 0;
        for (const auto& at : active_sorted) {
            if (at.start > cursor && (at.start - cursor) >= aligned_size) {
                offset = cursor;
                break;
            }
            cursor = std::max(cursor, at.start + at.size);
        }

        if (offset == -1) {
            // 没有内部间隙：尝试使用尾部空闲空间，否则扩展高水位
            offset = cursor;
            if (cursor + aligned_size <= total_allocated) {
                num_reuses++;  // 复用了已释放的尾部空间
            } else {
                total_allocated = cursor + aligned_size;
            }
        } else {
            num_reuses++;
        }

        active.push_back({tensor.name, aligned_size, offset, tensor.alive_to});
        offsets[tensor.name] = offset;
        sizes[tensor.name] = aligned_size;
        lifetimes[tensor.name] = tensor;

        if (config.tracer) {
            config.tracer->on_alloc(tensor.alive_from, tensor.name, offset, aligned_size);
        }

        if (config.verbose) {
            std::cout << "  ALLOC " << tensor.name << " (size=" << aligned_size
                      << ", offset=" << offset << ", alive=["
                      << tensor.alive_from << "," << tensor.alive_to << "])\n";
        }
    }

    return finalize_alloc_result(
        std::move(offsets), std::move(sizes), std::move(lifetimes),
        total_without_reuse, num_reuses, sorted, config);
}

// ============================================================
//  策略 2：Best Fit
// ============================================================

std::string BestFitStrategy::name() const {
    return "Best-Fit";
}

AllocationResult BestFitStrategy::allocate(
    const std::vector<TensorLifetime>& tensors,
    const AllocConfig& config) const {

    if (tensors.empty()) {
        AllocationResult r;
        r.success = true;
        return r;
    }

    auto sorted = tensors;
    std::sort(sorted.begin(), sorted.end(), [](const TensorLifetime& a, const TensorLifetime& b) {
        if (a.alive_from != b.alive_from) return a.alive_from < b.alive_from;
        return a.size > b.size;
    });

    struct ActiveTensor {
        std::string name;
        int64_t size;
        int64_t start;
        int alive_to;
    };
    std::vector<ActiveTensor> active;
    int64_t total_allocated = 0;
    int64_t total_without_reuse = 0;
    int num_reuses = 0;

    std::map<std::string, int64_t> offsets;
    std::map<std::string, int64_t> sizes;
    std::map<std::string, TensorLifetime> lifetimes;

    for (const auto& tensor : sorted) {
        int64_t aligned_size = align_up(tensor.size, config.alignment);
        total_without_reuse += aligned_size;

        // 释放过期张量，使用张量自身的到期时刻作为 free 时间
        for (auto it = active.begin(); it != active.end(); ) {
            if (it->alive_to < tensor.alive_from) {
                if (config.tracer) {
                    config.tracer->on_free(it->alive_to + 1, it->name, it->start, it->size);
                }
                it = active.erase(it);
            } else {
                ++it;
            }
        }

        // Best-Fit：在所有空闲区间（含尾部剩余空间）中找剩余量最小的
        int64_t best_offset = -1;
        int64_t best_remaining = -1;

        auto active_sorted = active;
        std::sort(active_sorted.begin(), active_sorted.end(),
            [](const ActiveTensor& a, const ActiveTensor& b) {
                return a.start < b.start;
            });

        int64_t cursor = 0;
        for (const auto& at : active_sorted) {
            if (at.start > cursor) {
                int64_t gap = at.start - cursor;
                if (gap >= aligned_size) {
                    int64_t remaining = gap - aligned_size;
                    if (best_remaining == -1 || remaining < best_remaining) {
                        best_remaining = remaining;
                        best_offset = cursor;
                    }
                }
            }
            cursor = std::max(cursor, at.start + at.size);
        }

        // 将尾部空闲空间也作为候选（正确比较剩余量而非空间总量）
        int64_t trailing_space = total_allocated - cursor;
        if (trailing_space >= aligned_size) {
            int64_t trailing_remaining = trailing_space - aligned_size;
            if (best_remaining == -1 || trailing_remaining < best_remaining) {
                best_offset = cursor;
                best_remaining = trailing_remaining;
            }
        }

        if (best_offset == -1) {
            // 没有任何合适的空闲区间，必须扩展高水位
            best_offset = cursor;
            total_allocated = cursor + aligned_size;
        } else {
            num_reuses++;
        }

        active.push_back({tensor.name, aligned_size, best_offset, tensor.alive_to});
        offsets[tensor.name] = best_offset;
        sizes[tensor.name] = aligned_size;
        lifetimes[tensor.name] = tensor;

        if (config.tracer) {
            config.tracer->on_alloc(tensor.alive_from, tensor.name, best_offset, aligned_size);
        }

        if (config.verbose) {
            std::cout << "  ALLOC " << tensor.name << " (size=" << aligned_size
                      << ", offset=" << best_offset << ", alive=["
                      << tensor.alive_from << "," << tensor.alive_to << "])\n";
        }
    }

    return finalize_alloc_result(
        std::move(offsets), std::move(sizes), std::move(lifetimes),
        total_without_reuse, num_reuses, sorted, config);
}

// ============================================================
//  策略 3：Largest First（按张量大小降序排列）
// ============================================================

std::string LargestFirstStrategy::name() const {
    return "Largest-First";
}

AllocationResult LargestFirstStrategy::allocate(
    const std::vector<TensorLifetime>& tensors,
    const AllocConfig& config) const {

    if (tensors.empty()) {
        AllocationResult r;
        r.success = true;
        return r;
    }

    auto sorted = tensors;
    std::sort(sorted.begin(), sorted.end(), [](const TensorLifetime& a, const TensorLifetime& b) {
        return a.size > b.size;
    });

    struct ActiveTensor {
        std::string name;
        int64_t size;
        int64_t start;
        int alive_to;
    };
    std::vector<ActiveTensor> active;
    int64_t total_allocated = 0;
    int64_t total_without_reuse = 0;
    int num_reuses = 0;

    std::map<std::string, int64_t> offsets;
    std::map<std::string, int64_t> sizes;
    std::map<std::string, TensorLifetime> lifetimes;

    for (const auto& tensor : sorted) {
        int64_t aligned_size = align_up(tensor.size, config.alignment);
        total_without_reuse += aligned_size;

        // 释放过期张量，使用张量自身的到期时刻作为 free 时间
        for (auto it = active.begin(); it != active.end(); ) {
            if (it->alive_to < tensor.alive_from) {
                if (config.tracer) {
                    config.tracer->on_free(it->alive_to + 1, it->name, it->start, it->size);
                }
                it = active.erase(it);
            } else {
                ++it;
            }
        }

        // First-Fit：找第一个能容纳的空闲区间
        int64_t offset = -1;
        auto active_sorted = active;
        std::sort(active_sorted.begin(), active_sorted.end(),
            [](const ActiveTensor& a, const ActiveTensor& b) {
                return a.start < b.start;
            });

        int64_t cursor = 0;
        for (const auto& at : active_sorted) {
            if (at.start > cursor && (at.start - cursor) >= aligned_size) {
                offset = cursor;
                break;
            }
            cursor = std::max(cursor, at.start + at.size);
        }

        if (offset == -1) {
            // 没有内部间隙：尝试使用尾部空闲空间，否则扩展高水位
            offset = cursor;
            if (cursor + aligned_size <= total_allocated) {
                num_reuses++;  // 复用了已释放的尾部空间
            } else {
                total_allocated = cursor + aligned_size;
            }
        } else {
            num_reuses++;
        }

        active.push_back({tensor.name, aligned_size, offset, tensor.alive_to});
        offsets[tensor.name] = offset;
        sizes[tensor.name] = aligned_size;
        lifetimes[tensor.name] = tensor;

        if (config.tracer) {
            config.tracer->on_alloc(tensor.alive_from, tensor.name, offset, aligned_size);
        }

        if (config.verbose) {
            std::cout << "  ALLOC " << tensor.name << " (size=" << aligned_size
                      << ", offset=" << offset << ", alive=["
                      << tensor.alive_from << "," << tensor.alive_to << "])\n";
        }
    }

    return finalize_alloc_result(
        std::move(offsets), std::move(sizes), std::move(lifetimes),
        total_without_reuse, num_reuses, sorted, config);
}

} // namespace sram
