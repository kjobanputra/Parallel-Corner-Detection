cmake_minimum_required(VERSION 3.16)
project(Parallel_Corner_Detection)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/local/opt/opencv@2/include -L/usr/local/opt/opencv@2/lib -lopencv_highgui -lopencv_features2d -lopencv_imgproc -lopencv_core")

add_executable(serialCornerDetection
        serialCornerDetection.cpp)

add_executable(cpuParallelCornerDetection
        cpuParallelCornerDetection.cpp)
