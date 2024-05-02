cmake_minimum_required(VERSION 3.22)

find_package(Eigen3 3.4.0 REQUIRED EXACT)

configure_file(TMDConfig.hpp.in TMDConfig.hpp)

file(GLOB TMD_SRC *.cc)

add_library(TMD ${TMD_SRC})
target_include_directories(TMD PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(TMD PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
message(STATUS ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(TMD PUBLIC Eigen3::Eigen)

if (IMPLEMENT_SLOW_EVALUATION)
	target_link_libraries(TMD PRIVATE Numerics)
endif()