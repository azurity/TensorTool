cmake_minimum_required(VERSION 3.10.0)
project(TensorTool VERSION 0.1.0 LANGUAGES C CXX)

find_package(Eigen3 REQUIRED)

add_library(TensorTool INTERFACE)
target_compile_features(TensorTool INTERFACE cxx_std_17)
target_link_libraries(TensorTool INTERFACE Eigen3::Eigen)
target_include_directories(TensorTool INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

include(CTest)
enable_testing()

find_package(Catch2 REQUIRED)

file(GLOB_RECURSE test_src tests/*.cpp)
add_executable(TensorToolTests ${test_src})
target_link_libraries(TensorToolTests PRIVATE TensorTool Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(TensorToolTests)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
