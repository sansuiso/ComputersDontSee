project(ComputersDontSee)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)

IF(APPLE)
	set( COCOA_LIBS ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/Cocoa.framework )
	set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -F/usr/local/lib")
ENDIF(APPLE)

set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUGS} -D__MY_DEBUG__" )

#----------------
# Core functions
#----------------
file(GLOB CDS_LIB_SOURCES
	src/lib/*.cpp
	src/lib/*.hpp
)

add_library(
	ComputersDontSee 
	${CDS_LIB_SOURCES}
)

# GUI

#-------------
# Executables 
#-------------
add_executable(
	Inpainting
	src/apps/main_Inpainting.cpp
)

target_link_libraries(
	Inpainting
	ComputersDontSee
	${OpenCV_LIBS}
)
