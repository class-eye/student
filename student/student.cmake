include_directories(${CMAKE_CURRENT_LIST_DIR})
set(SRC ${CMAKE_CURRENT_LIST_DIR}/student.cpp
		${CMAKE_CURRENT_LIST_DIR}/pose.cpp  
		${CMAKE_CURRENT_LIST_DIR}/fs.cpp  
		${CMAKE_CURRENT_LIST_DIR}/behavior.cpp  
		${CMAKE_CURRENT_LIST_DIR}/functions.cpp  
		${CMAKE_CURRENT_LIST_DIR}/student/functions.hpp
		${CMAKE_CURRENT_LIST_DIR}/student/behavior.hpp
		${CMAKE_CURRENT_LIST_DIR}/student/student.hpp
		${CMAKE_CURRENT_LIST_DIR}/student/fs.hpp
		${CMAKE_CURRENT_LIST_DIR}/student/Timer.hpp
		${CMAKE_CURRENT_LIST_DIR}/student/pose.hpp)	
set(STUDENT_INCLUDE ${CMAKE_CURRENT_LIST_DIR})
set(STUDENT_LIBRARY student)
add_library(student STATIC ${SRC})
target_link_libraries(student ${OpenCV_LIBS})