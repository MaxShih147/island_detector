cmake_minimum_required(VERSION 3.16)
project(island_detector VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# Fetch Clipper2
include(FetchContent)
FetchContent_Declare(
  clipper2
  GIT_REPOSITORY https://github.com/AngusJohnson/Clipper2.git
  GIT_TAG        Clipper2_1.4.0
  SOURCE_SUBDIR  CPP
)
FetchContent_MakeAvailable(clipper2)

# --- Library ---
add_library(island_detector
  src/island_detector.cpp
)

target_include_directories(island_detector PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(island_detector PUBLIC
  ${OpenCV_LIBS}
  Clipper2
)

# --- CLI tool ---
add_executable(island_detect_cli src/main.cpp)
target_link_libraries(island_detect_cli PRIVATE island_detector)
