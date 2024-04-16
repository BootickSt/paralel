cmake_minimum_required(VERSION 3.14)
project(1lab)

set(CMAKE_CXX_STANDART 11)

option(float "set_float" OFF)

if(float)
	add_compile_definitions(FLOAT)
	add_executable(paralel1_float paralel1.cpp)
else()
	add_executable(paralel1_double paralel1.cpp)
endif()
