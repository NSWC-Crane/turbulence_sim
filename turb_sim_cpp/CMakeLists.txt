# ----------------------------------------------------------------------------
#  Root CMake file for the playground code
# ----------------------------------------------------------------------------
#
# To perform an out of source build do the following on the command line:
# 1. navigate to the root project directory (in this case DfD2)
# 2. mkdir build
# 3. cd build
# 4. depending on your system (windows/linux) type the following
# Example CMake build line for windows & VS2015
# windows: cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_TOOLCHAIN_FILE="D:/Projects/vcpkg/scripts/buildsystems/vcpkg.cmake" ..
# linux: cmake -DUSE_AVX_INSTRUCTIONS=ON ..
#
# 5. To compile:
# windows: cmake --build . --config Release
# linux: cmake --build . --config Release -- -j4
#


cmake_minimum_required(VERSION 2.8.12)

PROJECT(tubulence_sim_project)
set(exe_name "tb")

set(PLATFORM "$ENV{PLATFORM}" CACHE INTERNAL "Copied from environment variable")
if("${PLATFORM}" STREQUAL "")
    set(PLATFORM "NOT_SET")
    message("The PLATFORM Environment variable is not set on this system.  Setting it to:")
endif()
message("Platform: " ${PLATFORM})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32)

	if(${PLATFORM} STREQUAL "Laptop")
		set(COMMON_ROOT "D:/Common")
        set(VCPKG_ROOT "D:/Projects/vcpkg")
	elseif(${PLATFORM} STREQUAL "LaptopN")
		set(COMMON_ROOT "C:/Projects/Common")
        set(VCPKG_ROOT "C:/Projects/vcpkg-master")
    elseif(${PLATFORM} STREQUAL "LaptopLenovo")
        set(COMMON_ROOT "C:/Common")
        set(VCPKG_ROOT "C:/Projects/vcpkg-master")
    else()
        set(COMMON_ROOT "C:/Projects/Common")
        set(VCPKG_ROOT "C:/Projects/vcpkg-master")
    endif()
    
    #set(GSL_DIR "D:/Projects/vcpkg/installed/x64-windows/")
    
    include(cmake_utils/find_GSL.cmake)
    if(GSL_FOUND)
        set(project_includes ${project_includes} ${GSL_INCLUDE_DIRS})
        set(project_libs ${project_libs} ${GSL_LIBRARIES})
    endif()
    
    
    
    #set(CMAKE_TOOLCHAIN_FILE "D:/Projects/vcpkg/scripts/buildsystems/vcpkg.cmake"
    #    CACHE STRING "Vcpkg toolchain file")
    
    #find_package(GSL REQUIRED
        #HINTS "D:/Projects/vcpkg/scripts/buildsystems/vcpkg.cmake"
    #    )
    
else()

    if(${PLATFORM} STREQUAL "MainGear")
        set(_ECLIPSE_VERSION "3.8")
        set(COMMON_ROOT "~/Projects/Common")
    elseif(${PLATFORM} STREQUAL "HPC")
        set(COMMON_ROOT "~/Projects/Common")
    else()
        set(COMMON_ROOT "~/Projects/Common")
    endif()

endif()




#include(${COMMON_ROOT}/cmake_utils/use_cpp_11.cmake)
include(${COMMON_ROOT}/cmake_utils/use_sse_avx.cmake)

# look for any *.c and *.cpp files
file(GLOB SOURCES "src/*.c*"
    "${COMMON_ROOT}/src/*.cpp"
    )

set(project_includes
    ${project_includes}
    "include"
    "${COMMON_ROOT}/include"
    )

# find OpenCV
message(STATUS "--------------------------------------------------------------------------------")
message(STATUS "Looking for OpenCV installation...")
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
	set(project_includes ${project_includes} ${OpenCV_INCLUDE_DIRS})
	set(project_libs ${project_libs} ${OpenCV_LIBS})
	message(STATUS "Found OpenCV: " ${OpenCV_DIR})
else()
	message("OpenCV not found...")
endif()
message(STATUS "--------------------------------------------------------------------------------")

message(" ")
message("--------------------------------------------------------------------------------")
message("Source Files: ")
foreach(S ${SOURCES})
    message("  " ${S})
endforeach()
message("--------------------------------------------------------------------------------")

message(" ")

message("--------------------------------------------------------------------------------")
message("Project Includes: ")
foreach(S ${project_includes})
    message("  " ${S})
endforeach()
message("--------------------------------------------------------------------------------")

message(" ")

message("--------------------------------------------------------------------------------")
message("Project Libraries: ")
foreach(S ${project_libs})
    message("  " ${S})
endforeach()
message("--------------------------------------------------------------------------------")

message(" ")

include_directories(${project_includes})
add_executable(${exe_name} ${SOURCES})

if(WIN32)

    target_link_libraries(${exe_name} ${project_libs} gsl gslcblas)
   
    message("MSVC Version: " ${MSVC_VERSION})
    # uncomment for MSVC and cmake greater than 3.6.0
    set(default_build_type "Release")
    if(MSVC AND (${CMAKE_VERSION} VERSION_GREATER "3.6.0"))
        target_compile_options(${exe_name} PRIVATE "/MP")
        set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${exe_name})
    endif()
    
elseif(UNIX)
#    target_link_libraries(${exe_name} ${project_libs} ${CMAKE_DL_LIBS})    
    target_link_libraries(${exe_name} ${project_libs})    
    target_link_libraries(${exe_name} PRIVATE GSL::gsl GSL::gslcblas)

endif()

message(" ")
message(STATUS "Build Complete!")
message(" ")

