cmake_minimum_required(VERSION 3.22)

if (IMPLEMENT_SLOW_EVALUATION)
	add_subdirectory(Numerics)
endif()