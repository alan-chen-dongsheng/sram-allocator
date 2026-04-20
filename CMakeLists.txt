cmake_minimum_required(VERSION 3.26)
project(sram_allocator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED CONFIG
    PATHS /usr/local/lib/python3.12/dist-packages/pybind11/share/cmake/pybind11
)

# Core library
add_library(sram_alloc SHARED
    src/cpp/linear_scan.cpp
)

# pybind11 bindings
pybind11_add_module(_core MODULE
    src/cpp/bindings.cpp
)

target_link_libraries(_core PRIVATE sram_alloc ${Python3_LIBRARIES})

# Copy to Python package
set(PYTHON_PKG "${CMAKE_SOURCE_DIR}/src/python/sram_allocator")
add_custom_command(TARGET sram_alloc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:sram_alloc> ${PYTHON_PKG}/
)
add_custom_command(TARGET _core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_core> ${PYTHON_PKG}/
)
