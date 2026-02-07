cmake_minimum_required(VERSION 3.16)
project(island_detector VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

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
)

# --- CLI tool ---
add_executable(island_detect_cli src/main.cpp)
target_link_libraries(island_detect_cli PRIVATE island_detector)
