cmake_minimum_required(VERSION 3.15)
project(ImageSaver C)

set(CMAKE_C_STANDARD 11)

add_library(ImageSaver SHARED src/imagesaver.c)
target_include_directories(ImageSaver PRIVATE src)

if(UNIX AND NOT APPLE)
	target_link_libraries(ImageSaver m)
endif()

# Output name without the 'lib' prefix on all platforms
set_target_properties(ImageSaver PROPERTIES PREFIX "")
