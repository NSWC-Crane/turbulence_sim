# ----------------------------------------------------------------------------
#  Root CMake file for the turbulence simulator library code
# ----------------------------------------------------------------------------
#
# To perform an out of source build do the following on the command line:
# 1. navigate to the root project directory
# 2. mkdir build
# 3. cd build
# 4. depending on your system (windows/linux) type the following
# Example CMake build line for windows & VS2019
# windows: cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_TOOLCHAIN_FILE="D:/Projects/vcpkg/scripts/buildsystems/vcpkg.cmake" ..
# linux: cmake -DUSE_AVX_INSTRUCTIONS=ON ..
#
# 5. To compile:
# windows: cmake --build . --config Release
# linux: cmake --build . --config Release -- -j4
#

cmake_minimum_required(VERSION 3.6)

PROJECT(tubulence_sim_library VERSION 1.0 DESCRIPTION "Shared library for generating turbelence in images")
set(LIB_NAME "turb_sim")

#set(default_build_type "Release")

set(PLATFORM "$ENV{PLATFORM}" CACHE INTERNAL "Copied from environment variable")
if("${PLATFORM}" STREQUAL "")
    set(PLATFORM "NOT_SET")
    message("The PLATFORM Environment variable is not set on this system.  Setting it to:")
endif()
message("Platform: " ${PLATFORM})

# find the root of the main project
string(FIND ${PROJECT_SOURCE_DIR} "/" INDEX REVERSE)
string(SUBSTRING ${PROJECT_SOURCE_DIR} 0 ${INDEX} PR_ROOT)

set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32)

	if(${PLATFORM} STREQUAL "Laptop")
		set(COMMON_ROOT "D:/Common")
        #set(VCPKG_ROOT "D:/Projects/vcpkg")
	elseif(${PLATFORM} STREQUAL "LaptopN")
		set(COMMON_ROOT "C:/Projects/Common")
        #set(VCPKG_ROOT "C:/Projects/vcpkg-master")
    elseif(${PLATFORM} STREQUAL "LaptopLenovo")
        set(COMMON_ROOT "C:/Common")
        #set(VCPKG_ROOT "C:/Projects/vcpkg-master")
    else()
        set(COMMON_ROOT "C:/Projects/Common")
        #set(VCPKG_ROOT "C:/Projects/vcpkg-master")
    endif()

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

include(${COMMON_ROOT}/cmake_utils/use_sse_avx.cmake)

# look for any *.c and *.cpp files
file(GLOB SOURCES "src/*.c*"
    )

set(project_includes
    "include"
    "${PR_ROOT}/common/include"
    "${COMMON_ROOT}/include"
    )
    
include("${PR_ROOT}/common/cmake_utils/find_GSL.cmake")
if(GSL_FOUND)
    set(project_includes ${project_includes} ${GSL_INCLUDE_DIRS})
    set(project_libs ${project_libs} ${GSL_LIBRARIES})
endif()    
 

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
message(STATUS "--------------------------------------------------------------------------------")
message(STATUS "Source Files: ")
foreach(S ${SOURCES})
    message(STATUS "  " ${S})
endforeach()
message(STATUS "--------------------------------------------------------------------------------")

message(" ")

message(STATUS "--------------------------------------------------------------------------------")
message(STATUS "Project Includes: ")
foreach(S ${project_includes})
    message(STATUS "  " ${S})
endforeach()
message(STATUS "--------------------------------------------------------------------------------")

message(" ")

message(STATUS "--------------------------------------------------------------------------------")
message(STATUS "Project Libraries: ")
foreach(S ${project_libs})
    message(STATUS "  " ${S})
endforeach()
message(STATUS "--------------------------------------------------------------------------------")

message(" ")

#----------------------------------------------------------------------------
message(STATUS "Building Library...")

add_definitions(-DBUILD_LIB)

include_directories(${project_includes})

if(WIN32)

    add_definitions(-DLIB_EXPORTS)

    add_library(${LIB_NAME} SHARED ${SOURCES})
    set_target_properties(${LIB_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
    target_link_libraries(${LIB_NAME} ${project_libs})

    if(MSVC AND (${CMAKE_VERSION} VERSION_GREATER "3.6.0"))
        message(STATUS "MSVC Version: " ${MSVC_VERSION})
        target_compile_options(${LIB_NAME} PUBLIC "/MP" "/bigobj")
        set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${LIB_NAME})
    endif() 

elseif(UNIX)

    # Change these to point to the right location for a particular machine
    #add_definitions("-fPIC")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    #find_package(X11 REQUIRED)
    add_library(${LIB_NAME} SHARED ${SOURCES})
    set_target_properties(${LIB_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
    #target_link_libraries(${LIB_NAME} ${X11_LIBRARIES})
    #target_link_libraries(${LIB_NAME} ${X11_LIBRARIES})
    target_link_libraries(${LIB_NAME} ${project_libs})

elseif(APPLE)
    message(FATAL_ERROR "Apple -> Not Even!")

endif()

#----------------------------------------------------------------------------
message(" ")
message(STATUS "Build Complete!")
message(" ")

