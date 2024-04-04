cmake_minimum_required(VERSION 3.22)

find_package(GTest REQUIRED)

add_executable(test test.cc)
target_link_libraries(test PRIVATE TMD)

enable_testing()
include(GoogleTest)
gtest_discover_tests(test)
target_link_libraries(test PRIVATE GTest::gtest_main)
target_link_libraries(test PRIVATE Numerics)