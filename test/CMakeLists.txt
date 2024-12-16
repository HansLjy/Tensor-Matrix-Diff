cmake_minimum_required(VERSION 3.22)

set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
cmake_path(APPEND TEST_OUTPUT_DIR "outputs")
configure_file(TestConfig.hpp.in TestConfig.hpp)

find_package(GTest REQUIRED)

add_executable(test test.cc)
target_link_libraries(test PRIVATE TMD)

enable_testing()
include(GoogleTest)
gtest_discover_tests(test)
target_link_libraries(test PRIVATE GTest::gtest_main)
target_link_libraries(test PRIVATE Numerics)
target_include_directories(test PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_executable(example example.cc)
target_link_libraries(example PRIVATE TMD Numerics)
target_include_directories(example PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
