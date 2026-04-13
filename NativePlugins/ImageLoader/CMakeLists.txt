cmake_minimum_required(VERSION 3.15)
project(ImageLoader C)

set(CMAKE_C_STANDARD 11)

add_library(ImageLoader SHARED src/imageloader.c)
target_include_directories(ImageLoader PRIVATE src)

if(UNIX AND NOT APPLE)
	target_link_libraries(ImageLoader m)
endif()

# Output name without the 'lib' prefix on all platforms
set_target_properties(ImageLoader PROPERTIES PREFIX "")
