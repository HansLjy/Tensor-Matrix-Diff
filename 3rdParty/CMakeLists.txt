cmake_minimum_required(VERSION 3.22)

if (IMPLEMENT_SLOW_EVALUATION)
	if (NOT TARGET Numerics)
		add_subdirectory(Numerics)
	endif()
endif()