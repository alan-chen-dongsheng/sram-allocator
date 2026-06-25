#include "linear_scan.h"
#include <algorithm>
#include <queue>
#include <iostream>
#include <numeric>
#include <functional>
#include <utility>
#include <sstream>
#include <fstream>
#include <cmath>

namespace sram {

// ============================================================
//  PerfettoTracer 实现
// ============================================================

void PerfettoTracer::on_alloc(int time_step, const std::string& tensor,
                              int64_t offset, int64_t size, int duration_steps) {
    // 使用 complete 事件（X）而非 begin（B）：自带 dur，无需配对 end，
    // 避免生命周期非嵌套时 B/E 配对错乱导致的"永不结束"切片。
    int64_t dur = static_cast<int64_t>(std::max(1, duration_steps)) * 1000;
    events_.push_back({time_step * 1000, "X", "SRAM: " + tensor, "memory",
                       1, 1, size, offset, 0, 0, "", dur});
}

void PerfettoTracer::on_free(int time_step, const std::string& tensor,
                             int64_t offset, int64_t size) {
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

        if (e.ph == "X" && e.dur > 0) {
            j << ",\"dur\":" << e.dur;
        }

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

int64_t SRAMAllocator::align_up(int64_t size, int64_t alignment) {
    if (alignment <= 1) return size;
    return (size + alignment - 1) / alignment * alignment;
}

// ===== 拓扑排序 =====
std::vector<std::string> SRAMAllocator::topo_sort(const DAG& dag) {
    std::map<std::string, std::vector<std::string>> adj;
    std::map<std::string, int> in_degree;
    std::set<std::string> all_ops;
    std::map<std::string, std::string> producer_of;

    for (const auto& op : dag.ops) {
        all_ops.insert(op.name);
        in_degree[op.name] = 0;
        if (!op.output.empty()) producer_of[op.output] = op.name;
    }

    // 添加依赖边：既包括显式 depends_on，也包括由 inputs 推导的
    // 数据依赖（输入张量由同名算子产出），保证拓扑序尊重数据流。
    auto add_edge = [&](const std::string& from, const std::string& to) {
        if (from == to || !all_ops.count(from) || !all_ops.count(to)) return;
        adj[from].push_back(to);
        in_degree[to]++;
    };

    for (const auto& op : dag.ops) {
        for (const auto& dep : op.depends_on) add_edge(dep, op.name);
        for (const auto& inp : op.inputs) {
            auto it = producer_of.find(inp);
            if (it != producer_of.end()) add_edge(it->second, op.name);
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
    if (dag.ops.empty()) {
        AllocationResult r;
        r.success = true;
        return r;
    }

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
//  辅助工具函数（策略内部使用）
// ============================================================

static int64_t align_up(int64_t size, int64_t alignment) {
    if (alignment <= 1) return size;
    return (size + alignment - 1) / alignment * alignment;
}

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


// ============================================================
//  共享辅助：基于时间重叠的冲突检测放置
//  算法无关、处理顺序无关 —— 两个张量当且仅当生命周期重叠
//  且地址区间重叠时才冲突。这样即使按大小降序（Largest-First）
//  这种非时间单调的顺序处理，也不会把同时存活的张量放到重叠地址。
// ============================================================

struct PlacedBlock {
    int64_t offset;
    int64_t size;
    int alive_from;
    int alive_to;
};

// 为生命周期 [alive_from, alive_to]、对齐后大小 aligned_size 的张量
// 寻找放置偏移。best_fit=false 取最低可用偏移（First-Fit）；
// best_fit=true 取剩余最小的可用间隙（Best-Fit）。
// reused 输出该张量是否复用了既有空间（未推高高水位）。
static int64_t find_offset(
    const std::vector<PlacedBlock>& placed,
    int64_t aligned_size,
    int alive_from, int alive_to,
    bool best_fit,
    int64_t high_water,
    bool& reused) {

    // 收集所有时间重叠的已放置块作为地址障碍
    std::vector<std::pair<int64_t, int64_t>> obstacles;
    for (const auto& b : placed) {
        if (alive_from <= b.alive_to && b.alive_from <= alive_to) {
            obstacles.emplace_back(b.offset, b.offset + b.size);
        }
    }
    std::sort(obstacles.begin(), obstacles.end());

    // 合并重叠/相邻区间
    std::vector<std::pair<int64_t, int64_t>> merged;
    for (const auto& o : obstacles) {
        if (!merged.empty() && o.first <= merged.back().second)
            merged.back().second = std::max(merged.back().second, o.second);
        else
            merged.push_back(o);
    }

    int64_t best_offset = -1;
    int64_t best_remaining = -1;  // -1 = 尚无候选
    int64_t cursor = 0;

    for (const auto& m : merged) {
        int64_t gap_start = cursor;
        int64_t gap = m.first - gap_start;
        if (gap >= aligned_size) {
            if (!best_fit) {
                reused = (gap_start < high_water);
                return gap_start;  // First-Fit：最低偏移
            }
            int64_t remaining = gap - aligned_size;
            if (best_remaining == -1 || remaining < best_remaining) {
                best_remaining = remaining;
                best_offset = gap_start;
            }
        }
        cursor = std::max(cursor, m.second);
    }

    // 尾部空闲区 [cursor, ∞)
    if (best_fit && best_offset != -1) {
        reused = (best_offset < high_water);
        return best_offset;
    }
    reused = (cursor < high_water);
    return cursor;
}

// 共享后处理：构建时间线快照 + 溢出检测 + tracer peak/overflow 回调
static void build_timeline_and_overflow(
    const std::vector<TensorLifetime>& sorted,
    const std::map<std::string, int64_t>& sizes,
    const std::map<std::string, TensorLifetime>& lifetimes,
    const AllocConfig& config,
    int64_t total_without_reuse,
    int num_reuses,
    AllocationResult& result) {

    std::vector<TimelineSnapshot> timeline;
    std::vector<OverflowEvent> violations;
    int64_t global_peak = 0;

    std::set<int> time_points;
    for (const auto& t : sorted) {
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

    result.success = true;
    result.peak_memory = global_peak;
    result.total_tensor_bytes = total_without_reuse;
    result.reuse_ratio = total_without_reuse > 0
        ? 1.0 - (double)global_peak / total_without_reuse : 0.0;
    result.num_reuses = num_reuses;
    result.timeline = std::move(timeline);
    result.overflow = overflow;

    if (overflow) {
        result.overflow_report.max_memory_limit = config.max_memory;
        result.overflow_report.peak_memory = global_peak;
        result.overflow_report.peak_excess = global_peak - config.max_memory;
        result.overflow_report.violations = std::move(violations);
    }
}

// 共享放置主循环：按给定顺序放置张量，返回完整分配结果
static AllocationResult run_placement(
    std::vector<TensorLifetime> sorted,
    const AllocConfig& config,
    bool best_fit) {

    std::vector<PlacedBlock> placed;
    std::map<std::string, int64_t> offsets, sizes;
    std::map<std::string, TensorLifetime> lifetimes;
    int64_t high_water = 0;
    int64_t total_without_reuse = 0;
    int num_reuses = 0;

    for (const auto& tensor : sorted) {
        int64_t aligned_size = align_up(tensor.size, config.alignment);
        total_without_reuse += aligned_size;

        bool reused = false;
        int64_t offset = find_offset(placed, aligned_size,
                                     tensor.alive_from, tensor.alive_to,
                                     best_fit, high_water, reused);
        if (reused) num_reuses++;
        if (offset + aligned_size > high_water) high_water = offset + aligned_size;

        placed.push_back({offset, aligned_size, tensor.alive_from, tensor.alive_to});
        offsets[tensor.name] = offset;
        sizes[tensor.name] = aligned_size;
        lifetimes[tensor.name] = tensor;

        if (config.tracer) {
            int dur = std::max(1, tensor.alive_to - tensor.alive_from + 1);
            config.tracer->on_alloc(tensor.alive_from, tensor.name,
                                    offset, aligned_size, dur);
            config.tracer->on_free(tensor.alive_to, tensor.name,
                                   offset, aligned_size);
        }
        if (config.verbose) {
            std::cout << "  ALLOC " << tensor.name << " (size=" << aligned_size
                      << ", offset=" << offset << ", alive=["
                      << tensor.alive_from << "," << tensor.alive_to << "])\n";
        }
    }

    AllocationResult result;
    result.tensor_offsets = offsets;
    result.tensor_sizes = sizes;
    result.tensor_lifetimes = lifetimes;
    build_timeline_and_overflow(sorted, sizes, lifetimes, config,
                                total_without_reuse, num_reuses, result);
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

    auto sorted = tensors;
    std::sort(sorted.begin(), sorted.end(), [](const TensorLifetime& a, const TensorLifetime& b) {
        if (a.alive_from != b.alive_from) return a.alive_from < b.alive_from;
        return a.size > b.size;
    });
    return run_placement(std::move(sorted), config, /*best_fit=*/false);
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

    auto sorted = tensors;
    std::sort(sorted.begin(), sorted.end(), [](const TensorLifetime& a, const TensorLifetime& b) {
        if (a.alive_from != b.alive_from) return a.alive_from < b.alive_from;
        return a.size > b.size;
    });
    return run_placement(std::move(sorted), config, /*best_fit=*/true);
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

    auto sorted = tensors;
    std::sort(sorted.begin(), sorted.end(), [](const TensorLifetime& a, const TensorLifetime& b) {
        if (a.size != b.size) return a.size > b.size;
        return a.alive_from < b.alive_from;
    });
    return run_placement(std::move(sorted), config, /*best_fit=*/false);
}

} // namespace sram
