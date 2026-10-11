cmake_minimum_required(VERSION 3.16)
project(photokeeper CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn objdetect)
find_package(Threads REQUIRED)

add_library(pipeline pipeline_c/pipeline.cpp pipeline_c/edges.cpp pipeline_c/faces.cpp
                     pipeline_c/palette.cpp pipeline_c/quality.cpp pipeline_c/text.cpp)
target_link_libraries(pipeline PUBLIC ${OpenCV_LIBS} Threads::Threads)
target_include_directories(pipeline PUBLIC ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
# shortcut: model dir baked in at build time, swap for an init(models_dir) when the core ships inside an app
target_compile_definitions(pipeline PRIVATE PK_MODELS="${CMAKE_SOURCE_DIR}/models")

add_executable(photokeeper main.cpp)
target_link_libraries(photokeeper pipeline)

add_executable(test_pipeline tests/test_pipeline.cpp)
target_link_libraries(test_pipeline pipeline)
