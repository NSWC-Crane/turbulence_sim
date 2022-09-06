message(STATUS "--------------------------------------------------------------------------------")
message(STATUS "Looking for GSL Library...")

find_path(GSL_INCLUDE_DIRS gsl_cblas.h
    PATHS /usr/local "D:/Projects/vcpkg/installed/x64-windows" "C:/Projects/vcpkg/installed/x64-windows" ENV CPATH
    PATH_SUFFIXES include/gsl
    )

find_library(GSL_LIBRARIES gsl
    HINTS ${GSL_INCLUDE_DIRS}
    PATHS /usr/local "D:/Projects/vcpkg/installed/x64-windows" "C:/Projects/vcpkg/installed/x64-windows"
    PATH_SUFFIXES lib amd64 lib64 x64 
    )
    
find_library(GSL_CBLAS_LIB gslcblas
    HINTS ${GSL_LIBRARIES}
    PATHS /usr/local "D:/Projects/vcpkg/installed/x64-windows" "C:/Projects/vcpkg/installed/x64-windows"
    PATH_SUFFIXES lib amd64 lib64 x64 
    )    
    
set(GSL_LIBRARIES ${GSL_LIBRARIES} ${GSL_CBLAS_LIB})

   
mark_as_advanced(GSL_INCLUDE_DIRS GSL_LIBRARIES)

if (GSL_LIBRARIES AND GSL_INCLUDE_DIRS)
    set(GSL_FOUND TRUE)
    message(STATUS "Found GSL Library: " ${GSL_LIBRARIES})
    message(STATUS "Found GSL Includes: " ${GSL_INCLUDE_DIRS})

else()
    message("--- GSL library was not found! ---")
    set(GSL_FOUND FALSE)
endif()

message(STATUS "--------------------------------------------------------------------------------")
message(STATUS " ")
