cmake_minimum_required(VERSION 3.10)
project(MatrixMul)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Найти OpenCL
find_package(OpenCL REQUIRED)

# Найти GTest
find_package(GTest REQUIRED)

# Общие include директории
include_directories(${OpenCL_INCLUDE_DIRS})

# Тесты
enable_testing()

add_executable(test_matrix tests/test_matrix.cpp matrix.cpp)

target_link_libraries(test_matrix
    ${OpenCL_LIBRARIES}
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME MatrixTest COMMAND test_matrix)
set_tests_properties(MatrixTest PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
