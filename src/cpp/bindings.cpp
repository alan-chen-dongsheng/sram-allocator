#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "linear_scan.h"

namespace py = pybind11;

// Trampoline class to allow Python subclasses of AllocationStrategy
class PyAllocationStrategy : public sram::AllocationStrategy {
public:
    using sram::AllocationStrategy::AllocationStrategy;

    std::string name() const override {
        PYBIND11_OVERRIDE_PURE(
            std::string,
            sram::AllocationStrategy,
            name
        );
    }

    sram::AllocationResult allocate(
        const std::vector<sram::TensorLifetime>& tensors,
        const sram::AllocConfig& config) const override {
        // Acquire GIL for calling back into Python
        py::gil_scoped_acquire gil;
        py::function py_allocate = py::get_override(this, "allocate");
        if (!py_allocate) {
            throw std::runtime_error(
                "Python AllocationStrategy subclass must override allocate()");
        }
        py::object py_result = py_allocate(tensors, config);

        // If it's already a C++ AllocationResult, return it directly
        if (py::isinstance<py::class_<sram::AllocationResult>::type>(py_result)) {
            try {
                return py_result.cast<sram::AllocationResult>();
            } catch (...) {
                // Fall through to dataclass extraction
            }
        }

        // Otherwise, extract fields from a Python object (dataclass or duck-typed)
        sram::AllocationResult result;
        result.success = py_result.attr("success").cast<bool>();
        result.error_msg = py_result.attr("error_msg").cast<std::string>();
        result.peak_memory = py_result.attr("peak_memory").cast<int64_t>();
        result.reuse_ratio = py_result.attr("reuse_ratio").cast<double>();
        result.num_reuses = py_result.attr("num_reuses").cast<int>();
        result.tensor_offsets = py_result.attr("tensor_offsets")
            .cast<std::map<std::string, int64_t>>();
        result.tensor_sizes = py_result.attr("tensor_sizes")
            .cast<std::map<std::string, int64_t>>();
        // total_tensor_bytes maps from Python's total_without_reuse
        if (py::hasattr(py_result, "total_without_reuse")) {
            result.total_tensor_bytes =
                py_result.attr("total_without_reuse").cast<int64_t>();
        }
        // Optional fields
        if (py::hasattr(py_result, "overflow")) {
            result.overflow = py_result.attr("overflow").cast<bool>();
        }
        return result;
    }
};

PYBIND11_MODULE(_core, m) {
    m.doc() = "Linear Scan SRAM Allocator — Strategy Pattern Edition";
    m.attr("__version__") = "0.3.0";

    // ===== Data Structures =====
    py::class_<sram::TensorLifetime>(m, "TensorLifetime")
        .def(py::init<>())
        .def_readwrite("name", &sram::TensorLifetime::name)
        .def_readwrite("size", &sram::TensorLifetime::size)
        .def_readwrite("producer", &sram::TensorLifetime::producer)
        .def_readwrite("consumers", &sram::TensorLifetime::consumers)
        .def_readwrite("alive_from", &sram::TensorLifetime::alive_from)
        .def_readwrite("alive_to", &sram::TensorLifetime::alive_to)
        .def("__repr__", [](const sram::TensorLifetime& t) {
            return "<TensorLifetime '" + t.name + "' size=" +
                   std::to_string(t.size) +
                   " lifetime=[" + std::to_string(t.alive_from) +
                   "," + std::to_string(t.alive_to) + "]>";
        });

    py::class_<sram::OpNode>(m, "OpNode")
        .def(py::init<>())
        .def_readwrite("name", &sram::OpNode::name)
        .def_readwrite("op_type", &sram::OpNode::op_type)
        .def_readwrite("output_size", &sram::OpNode::output_size)
        .def_readwrite("inputs", &sram::OpNode::inputs)
        .def_readwrite("output", &sram::OpNode::output)
        .def_readwrite("depends_on", &sram::OpNode::depends_on)
        .def("__repr__", [](const sram::OpNode& n) {
            return "<OpNode '" + n.name + "' type=" + n.op_type +
                   " output=" + n.output + ">";
        });

    py::class_<sram::DAG>(m, "DAG")
        .def(py::init<>())
        .def_readwrite("name", &sram::DAG::name)
        .def_readwrite("ops", &sram::DAG::ops)
        .def("__repr__", [](const sram::DAG& d) {
            return "<DAG '" + d.name + "' ops=" +
                   std::to_string(d.ops.size()) + ">";
        });

    py::class_<sram::TimelineSnapshot>(m, "TimelineSnapshot")
        .def(py::init<>())
        .def_readwrite("time", &sram::TimelineSnapshot::time)
        .def_readwrite("event", &sram::TimelineSnapshot::event)
        .def_readwrite("tensor_name", &sram::TimelineSnapshot::tensor_name)
        .def_readwrite("peak_memory", &sram::TimelineSnapshot::peak_memory)
        .def_readwrite("active_memory", &sram::TimelineSnapshot::active_memory)
        .def_readwrite("live_tensors", &sram::TimelineSnapshot::live_tensors);

    // ===== AllocConfig (needed for Python-defined strategies) =====
    py::class_<sram::AllocConfig>(m, "AllocConfig")
        .def(py::init<>())
        .def_readwrite("alignment", &sram::AllocConfig::alignment)
        .def_readwrite("sram_size", &sram::AllocConfig::sram_size)
        .def_readwrite("max_memory", &sram::AllocConfig::max_memory)
        .def_readwrite("verbose", &sram::AllocConfig::verbose);

    // ===== Overflow types =====
    py::class_<sram::OverflowEvent>(m, "OverflowEvent")
        .def(py::init<>())
        .def_readwrite("time", &sram::OverflowEvent::time)
        .def_readwrite("active_memory", &sram::OverflowEvent::active_memory)
        .def_readwrite("limit", &sram::OverflowEvent::limit)
        .def_readwrite("excess", &sram::OverflowEvent::excess)
        .def_readwrite("live_tensors", &sram::OverflowEvent::live_tensors);

    py::class_<sram::OverflowReport>(m, "OverflowReport")
        .def(py::init<>())
        .def_readwrite("max_memory_limit", &sram::OverflowReport::max_memory_limit)
        .def_readwrite("peak_memory", &sram::OverflowReport::peak_memory)
        .def_readwrite("peak_excess", &sram::OverflowReport::peak_excess)
        .def_readwrite("violations", &sram::OverflowReport::violations);

    py::class_<sram::AllocationResult>(m, "AllocationResult")
        .def(py::init<>())
        .def_readwrite("success", &sram::AllocationResult::success)
        .def_readwrite("error_msg", &sram::AllocationResult::error_msg)
        .def_readwrite("tensor_offsets", &sram::AllocationResult::tensor_offsets)
        .def_readwrite("tensor_sizes", &sram::AllocationResult::tensor_sizes)
        .def_readwrite("tensor_lifetimes", &sram::AllocationResult::tensor_lifetimes)
        .def_readwrite("peak_memory", &sram::AllocationResult::peak_memory)
        .def_readwrite("total_tensor_bytes", &sram::AllocationResult::total_tensor_bytes)
        .def_readwrite("reuse_ratio", &sram::AllocationResult::reuse_ratio)
        .def_readwrite("num_reuses", &sram::AllocationResult::num_reuses)
        .def_readwrite("timeline", &sram::AllocationResult::timeline)
        .def_readwrite("sched_order", &sram::AllocationResult::sched_order)
        .def_readwrite("overflow", &sram::AllocationResult::overflow)
        .def_readwrite("overflow_report", &sram::AllocationResult::overflow_report)
        .def("__repr__", [](const sram::AllocationResult& r) {
            if (!r.success) return "<AllocationResult FAILED: " + r.error_msg + ">";
            std::string tag = r.overflow ? " [OVERFLOW]" : "";
            return "<AllocationResult peak=" + std::to_string(r.peak_memory) +
                   " reuse=" + std::to_string(r.num_reuses) +
                   " ratio=" + std::to_string(r.reuse_ratio) +
                   tag + ">";
        });

    // ===== Strategy Classes =====
    py::class_<sram::AllocationStrategy, PyAllocationStrategy, std::shared_ptr<sram::AllocationStrategy>>(
        m, "AllocationStrategy")
        .def(py::init<>())
        .def("name", &sram::AllocationStrategy::name)
        .def("allocate", &sram::AllocationStrategy::allocate,
             py::arg("tensors"), py::arg("config"));

    py::class_<sram::LinearScanStrategy, sram::AllocationStrategy,
               std::shared_ptr<sram::LinearScanStrategy>>(m, "LinearScanStrategy")
        .def(py::init<>())
        .def("name", &sram::LinearScanStrategy::name)
        .def("allocate", &sram::LinearScanStrategy::allocate,
             py::arg("tensors"), py::arg("config"));

    py::class_<sram::BestFitStrategy, sram::AllocationStrategy,
               std::shared_ptr<sram::BestFitStrategy>>(m, "BestFitStrategy")
        .def(py::init<>())
        .def("name", &sram::BestFitStrategy::name)
        .def("allocate", &sram::BestFitStrategy::allocate,
             py::arg("tensors"), py::arg("config"));

    py::class_<sram::LargestFirstStrategy, sram::AllocationStrategy,
               std::shared_ptr<sram::LargestFirstStrategy>>(m, "LargestFirstStrategy")
        .def(py::init<>())
        .def("name", &sram::LargestFirstStrategy::name)
        .def("allocate", &sram::LargestFirstStrategy::allocate,
             py::arg("tensors"), py::arg("config"));

    // ===== PerfettoTracer =====
    py::class_<sram::PerfettoTracer>(m, "PerfettoTracer")
        .def(py::init<>())
        .def("to_json", &sram::PerfettoTracer::to_json)
        .def("save", &sram::PerfettoTracer::save, py::arg("path"));

    // ===== Context: SRAMAllocator =====
    py::class_<sram::SRAMAllocator>(m, "SRAMAllocator")
        .def(py::init<>())
        .def("set_strategy", &sram::SRAMAllocator::set_strategy,
             py::arg("strategy"))
        .def("get_strategy", &sram::SRAMAllocator::get_strategy)
        .def("set_alignment", &sram::SRAMAllocator::set_alignment,
             py::arg("align"))
        .def("set_sram_size", &sram::SRAMAllocator::set_sram_size,
             py::arg("size"))
        .def("set_max_memory", &sram::SRAMAllocator::set_max_memory,
             py::arg("max"))
        .def("set_verbose", &sram::SRAMAllocator::set_verbose,
             py::arg("v"))
        .def("set_tracer", &sram::SRAMAllocator::set_tracer,
             py::arg("tracer"))
        .def("allocate", &sram::SRAMAllocator::allocate,
             py::arg("dag"))
        .def("allocate_from_tensors",
             &sram::SRAMAllocator::allocate_from_tensors,
             py::arg("tensors"));

    // ===== Registry API =====
    m.def("register_strategy", &sram::SRAMAllocator::register_strategy,
          py::arg("name"), py::arg("factory"));
    m.def("create_strategy", &sram::SRAMAllocator::create_strategy,
          py::arg("name"));
}
