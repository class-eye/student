project(student)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
include(mini-caffe/mini-caffe.cmake)
include(student/student.cmake)

file(GLOB SRC main/*.h* main/*.c*)
add_executable(main ${SRC})
target_link_libraries(main caffe ${OpenCV_LIBS} ${STUDENT_LIBRARY})
