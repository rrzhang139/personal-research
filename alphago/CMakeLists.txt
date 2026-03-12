cmake_minimum_required(VERSION 3.18)
project(mcts_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native -ffast-math)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Threads REQUIRED)

# nanobind via scikit-build-core or manual
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_mcts_cpp
    csrc/go_game.cpp
    csrc/mcts_node.cpp
    csrc/self_play.cpp
    csrc/bindings.cpp
)

target_include_directories(_mcts_cpp PRIVATE csrc)
target_link_libraries(_mcts_cpp PRIVATE Threads::Threads)

install(TARGETS _mcts_cpp LIBRARY DESTINATION mcts_cpp)
