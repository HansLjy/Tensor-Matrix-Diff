cmake_minimum_required(VERSION 3.22)

if (IMPLEMENT_SLOW_EVALUATION)
	if (NOT TARGET Numeric)
		add_subdirectory(Numerics)
	endif()
endif()