cmake_minimum_required(VERSION 3.16)
project(teleop)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Booster SDK is installed by /opt/booster/sdk/install.sh
# Headers are under /opt/booster/sdk/include
# Libraries are under /opt/booster/sdk/lib (or system paths after install)
set(BOOSTER_SDK_DIR "/opt/booster/sdk" CACHE PATH "Booster Robotics SDK root")

foreach(LIB booster_robotics_sdk fastcdr fastrtps)
    find_library(LIB_${LIB}
        NAMES ${LIB}
        HINTS "${BOOSTER_SDK_DIR}/lib" /usr/local/lib /usr/lib
    )
    if(NOT LIB_${LIB})
        message(FATAL_ERROR "Library '${LIB}' not found.")
    endif()
endforeach()

add_executable(teleop teleop.cpp)

target_include_directories(teleop PRIVATE
    "${BOOSTER_SDK_DIR}/include"
)

target_link_libraries(teleop PRIVATE
    "${LIB_booster_robotics_sdk}"
    "${LIB_fastcdr}"
    "${LIB_fastrtps}"
    dl pthread ssl crypto
)

install(TARGETS teleop RUNTIME DESTINATION bin)
