cmake_minimum_required(VERSION 3.26)
project(sram_allocator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Let .so find libsram_alloc.so in the same directory at runtime
set(CMAKE_BUILD_RPATH "\$ORIGIN")
set(CMAKE_INSTALL_RPATH "\$ORIGIN")

find_package(pybind11 CONFIG REQUIRED)

# Core library
add_library(sram_alloc SHARED src/cpp/linear_scan.cpp)
target_compile_options(sram_alloc PRIVATE -O3 -Wall)

# pybind11 Python extension
pybind11_add_module(_core MODULE src/cpp/bindings.cpp)
target_link_libraries(_core PRIVATE sram_alloc)

# Install targets — scikit-build-core packs them into the wheel
install(TARGETS sram_alloc _core LIBRARY DESTINATION .)
