#ifndef LINEAR_SCAN_H
#define LINEAR_SCAN_H

#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <functional>

namespace sram {

// Forward declaration
class PerfettoTracer;

// ============================================================
//  数据结构（算法无关，所有策略共享）
// ============================================================

// 张量生命周期
struct TensorLifetime {
    std::string name;
    int64_t size;
    std::string producer;
    std::vector<std::string> consumers;
    int alive_from;
    int alive_to;
};

// 内存块
struct MemoryBlock {
    int64_t start;
    int64_t size;
    std::string tensor_name;
    int freed_at;
    bool is_free() const { return tensor_name.empty(); }
};

// 时间线快照
struct TimelineSnapshot {
    int time;
    std::string event;
    std::string tensor_name;
    int64_t peak_memory;
    int64_t active_memory;
    std::vector<std::string> live_tensors;
};

// 溢出事件（单点违规记录）
struct OverflowEvent {
    int time;              // liveTime 索引
    int64_t active_memory; // 当时活跃内存
    int64_t limit;         // 限制值
    int64_t excess;        // 超出量
    std::vector<std::string> live_tensors; // 当时存活的张量
};

// 溢出报告
struct OverflowReport {
    int64_t max_memory_limit;
    int64_t peak_memory;
    int64_t peak_excess;   // 峰值超出量
    std::vector<OverflowEvent> violations;
};

// 算子节点
struct OpNode {
    std::string name;
    std::string op_type;
    int64_t output_size;
    std::vector<std::string> inputs;
    std::string output;
    std::vector<std::string> depends_on;
};

// DAG
struct DAG {
    std::string name;
    std::vector<OpNode> ops;
};

// 分配结果（统一格式）
struct AllocationResult {
    bool success;
    std::string error_msg;

    std::map<std::string, int64_t> tensor_offsets;
    std::map<std::string, int64_t> tensor_sizes;
    std::map<std::string, TensorLifetime> tensor_lifetimes;

    int64_t peak_memory;
    int64_t total_tensor_bytes;
    double reuse_ratio;
    int num_reuses;

    std::vector<TimelineSnapshot> timeline;
    std::vector<std::string> sched_order;

    // 溢出相关
    bool overflow = false;
    OverflowReport overflow_report;
};

// 分配配置（所有策略共享）
struct AllocConfig {
    int64_t alignment = 16;
    int64_t sram_size = -1;    // -1 = unlimited
    int64_t max_memory = -1;   // -1 = no limit, 超限不停止只记录
    bool verbose = false;
    PerfettoTracer* tracer = nullptr; // nullptr = 不生成 trace
};

// ============================================================
//  策略接口 —— 核心设计模式
// ============================================================

class AllocationStrategy {
public:
    virtual ~AllocationStrategy() = default;

    // 策略名称
    virtual std::string name() const = 0;

    // 执行分配
    virtual AllocationResult allocate(
        const std::vector<TensorLifetime>& tensors,
        const AllocConfig& config) const = 0;
};

// ============================================================
//  内置策略：线性扫描（First-Fit）
// ============================================================

class LinearScanStrategy : public AllocationStrategy {
public:
    std::string name() const override;
    AllocationResult allocate(
        const std::vector<TensorLifetime>& tensors,
        const AllocConfig& config) const override;
};

// ============================================================
//  内置策略：最佳适配（Best-Fit）
// ============================================================

class BestFitStrategy : public AllocationStrategy {
public:
    std::string name() const override;
    AllocationResult allocate(
        const std::vector<TensorLifetime>& tensors,
        const AllocConfig& config) const override;
};

// ============================================================
//  内置策略：按大小排序（Largest-First）
// ============================================================

class LargestFirstStrategy : public AllocationStrategy {
public:
    std::string name() const override;
    AllocationResult allocate(
        const std::vector<TensorLifetime>& tensors,
        const AllocConfig& config) const override;
};

// ============================================================
//  Perfetto Trace 生成器
// ============================================================

class PerfettoTracer {
public:
    // 记录事件（分配策略内部调用）
    void on_alloc(int time_step, const std::string& tensor,
                  int64_t offset, int64_t size, int duration_steps);
    void on_free(int time_step, const std::string& tensor,
                 int64_t offset, int64_t size);
    void on_peak(int time_step, int64_t peak_bytes, int64_t active_bytes);
    void on_overflow(int time_step, int64_t peak_bytes, int64_t limit);

    // 输出
    std::string to_json() const;
    void save(const std::string& path) const;

private:
    struct TraceEvent {
        int ts;         // 时间步（微秒级，= time_step * 1000）
        std::string ph; // B=begin, E=end, i=instant, C=counter
        std::string name;
        std::string cat;
        int64_t pid = 1;
        int64_t tid = 1;
        // args
        int64_t size = 0;
        int64_t offset = 0;
        int64_t bytes = 0;
        int64_t limit = 0;
        std::string s; // scope for instant
        int64_t dur = 0; // 持续时长（complete 事件 X，= duration_steps * 1000）
    };
    std::vector<TraceEvent> events_;
};

// ============================================================
//  Context 类 —— SRAMAllocator（持有策略，随时可换）
// ============================================================

class SRAMAllocator {
public:
    SRAMAllocator();

    // ===== 策略切换 =====
    void set_strategy(std::shared_ptr<AllocationStrategy> strategy);
    std::shared_ptr<AllocationStrategy> get_strategy() const;

    // ===== 配置 =====
    void set_alignment(int64_t align);
    void set_sram_size(int64_t size);
    void set_max_memory(int64_t max);
    void set_verbose(bool v);
    void set_tracer(PerfettoTracer* t);
    PerfettoTracer* get_tracer() const { return config_.tracer; }
    const AllocConfig& get_config() const { return config_; }

    // ===== 分配入口 =====
    AllocationResult allocate(const DAG& dag);
    AllocationResult allocate_from_tensors(
        const std::vector<TensorLifetime>& tensors);

    // ===== 工具：拓扑排序 + 生命周期计算 =====
    static std::vector<std::string> topo_sort(const DAG& dag);
    static std::vector<TensorLifetime> compute_lifetimes(
        const DAG& dag,
        const std::vector<std::string>& order);

    // ===== 注册自定义策略工厂（函数式扩展）=====
    using StrategyFactory = std::function<std::shared_ptr<AllocationStrategy>()>;
    static void register_strategy(const std::string& name, StrategyFactory factory);
    static std::shared_ptr<AllocationStrategy> create_strategy(const std::string& name);

private:
    AllocConfig config_;
    std::shared_ptr<AllocationStrategy> strategy_;

    // 内部工具
    static int64_t align_up(int64_t size, int64_t alignment);
};

} // namespace sram

#endif // LINEAR_SCAN_H
