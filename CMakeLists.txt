cmake_minimum_required(VERSION 3.24)
project(photokeeper CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # clangd (Zed, VS Code) reads build/compile_commands.json
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)  # cache, not plain: OpenCV sets properties on it (fails on MSVC)
endif()

# OpenCV is built from source into this tree: static, pthreads, only the modules used, bundled codecs.
set(OPENCV_MODULES core imgproc imgcodecs dnn objdetect)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)  # OpenCV 4.10's bundled 3rdparty CMake predates CMake 4
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
string(REPLACE ";" "," BUILD_LIST "${OPENCV_MODULES}")
set(BUILD_LIST "${BUILD_LIST}" CACHE STRING "" FORCE)
foreach(opt BUILD_JPEG BUILD_PNG BUILD_TIFF BUILD_ZLIB BUILD_PROTOBUF)
  set(${opt} ON CACHE BOOL "" FORCE)
endforeach()
foreach(opt WITH_TBB WITH_OPENMP WITH_IPP WITH_OPENCL WITH_ITT WITH_EIGEN WITH_LAPACK WITH_FFMPEG WITH_GTK WITH_V4L
            WITH_GSTREAMER WITH_1394 WITH_VA WITH_VA_INTEL WITH_ADE WITH_QUIRC WITH_VTK WITH_GDAL WITH_GPHOTO2
            WITH_OPENEXR WITH_JASPER WITH_OPENJPEG WITH_WEBP WITH_IMGCODEC_HDR WITH_IMGCODEC_SUNRASTER
            WITH_IMGCODEC_PXM WITH_IMGCODEC_PFM BUILD_TESTS BUILD_PERF_TESTS BUILD_EXAMPLES BUILD_opencv_apps
            BUILD_opencv_python3 BUILD_JAVA BUILD_DOCS OPENCV_GENERATE_PKGCONFIG BUILD_ANDROID_PROJECTS BUILD_ANDROID_EXAMPLES
            BUILD_WITH_STATIC_CRT)  # MSVC: link the same dynamic CRT (/MD) as everything else, Flutter included
  set(${opt} OFF CACHE BOOL "" FORCE)
endforeach()
include(FetchContent)
FetchContent_Declare(opencv
  URL https://github.com/opencv/opencv/archive/refs/tags/4.10.0.tar.gz
  URL_HASH SHA256=b2171af5be6b26f7a06b1229948bbb2bdaa74fcf5cd097e0af6378fce50a6eb9
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(opencv)
find_package(Threads REQUIRED)

add_library(pipeline pipeline_c/pipeline.cpp pipeline_c/load.cpp pipeline_c/edges.cpp pipeline_c/faces.cpp
                     pipeline_c/palette.cpp pipeline_c/quality.cpp pipeline_c/text.cpp)
# pipeline_c/pk.cpp (the C ABI) is deliberately not in `pipeline`: an app compiles it into its own shared library,
# otherwise the linker drops the unreferenced exports.
target_include_directories(pipeline PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR})  # build dir: opencv_modules.hpp
foreach(m ${OPENCV_MODULES})
  target_link_libraries(pipeline PUBLIC opencv_${m})
  target_include_directories(pipeline PUBLIC ${opencv_SOURCE_DIR}/modules/${m}/include)  # OpenCV keeps these PRIVATE
endforeach()
target_link_libraries(pipeline PUBLIC Threads::Threads)
# default model dir for the CLI; apps override it with pk_init()
target_compile_definitions(pipeline PRIVATE PK_MODELS="${CMAKE_CURRENT_SOURCE_DIR}/models")

if(NOT PROJECT_IS_TOP_LEVEL)
  return()  # an app embedding the library (add_subdirectory / FetchContent) only wants `pipeline`
endif()

add_executable(photokeeper main.cpp)
target_link_libraries(photokeeper pipeline)

add_executable(test_pipeline tests/test_pipeline.cpp pipeline_c/pk.cpp)
target_link_libraries(test_pipeline pipeline)
# OpenCV's CMake redirects executables to bin/; keep ours at build/photokeeper
set_target_properties(photokeeper test_pipeline PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
