project(ComputersDontSee)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)

# Will be used later for Grand Central Dispatch (GCD/libdispatch) parallel loops
IF(APPLE)
	set( WITH_DISPATCH true )
	set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -F/usr/local/lib")
ENDIF(APPLE)

#---------------------
# libComputersDontSee
#---------------------
include_directories("./include")
file( GLOB_RECURSE LIB_CDS_SOURCES
      src/lib/*.cpp 
)

add_library( ComputersDontSee
	     ${LIB_CDS_SOURCES} 
)

target_link_libraries(
	ComputersDontSee
	${OpenCV_LIBS}
)
      
#-------------
# Executables 
#-------------

# --- TV-Inpainting ---
file( GLOB TV_INPAINTING_APP_SOURCES src/tv_inpainting_app/*.cpp )

add_executable(
	tv_inpainting
	${TV_INPAINTING_APP_SOURCES}
)

target_link_libraries(
	tv_inpainting
	ComputersDontSee
	${OpenCV_LIBS}
)

# --- BG-Subtraction ---
file( GLOB BG_SUBTRACT_APP_SOURCES src/bg_subtract_app/*.cpp )

add_executable(
	bg_subtract
	${BG_SUBTRACT_APP_SOURCES}
)

target_link_libraries(
	bg_subtract
	ComputersDontSee
	${OpenCV_LIBS}
)
