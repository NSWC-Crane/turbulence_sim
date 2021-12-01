message(STATUS "--------------------------------------------------------------------------------")
message(STATUS "Looking for the MZA SHaRE Library...")

find_path(SHaRE_INCLUDE_DIRS MzaShare.h
    PATHS /usr/local /opt/MZA /home/$ENV{HOME}/MZA "C:/MZA/ScalingCode/2021a/ScalingCodeAPI" "D:/MZA/ScalingCode/2021a/ScalingCodeAPI" "E:/MZA/ScalingCode/2021a/ScalingCodeAPI" ENV CPATH
    PATH_SUFFIXES include
    )

find_library(SHaRE_LIBS MzaShare
    HINTS ${FTDI_INCLUDE_DIRS}
    PATHS /usr/local /opt/MZA /home/$ENV{HOME}/MZA "C:/MZA/ScalingCode/2021a/ScalingCodeAPI" "D:/MZA/ScalingCode/2021a/ScalingCodeAPI" "E:/MZA/ScalingCode/2021a/ScalingCodeAPI"
    PATH_SUFFIXES lib lib64 bin/windows/x64
    )
    
mark_as_advanced(SHaRE_LIBS SHaRE_INCLUDE_DIRS)

if (SHaRE_LIBS AND SHaRE_INCLUDE_DIRS)
    set(SHaRE_FOUND TRUE)

    message(STATUS "Found the MZA SHaRE Library: " ${SHaRE_LIBS})
else()
    message("--- MZA SHaRE Library was not found! ---")
    message("--- Provide search path to CMake ---")
    set(SHaRE_FOUND FALSE)
endif()

message(STATUS "--------------------------------------------------------------------------------")
message(STATUS " ")
