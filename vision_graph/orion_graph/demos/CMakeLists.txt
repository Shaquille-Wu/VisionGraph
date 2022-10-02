cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)
set(CMAKE_VERBOSE_MAKEFILE  ON)

set (GRAPH_DEMO_SRC
     face_graph.cpp)
add_executable(face_graph ${GRAPH_DEMO_SRC})
target_link_libraries(face_graph ${TBB_LIBS} vision_graph dl ${ANDROID_LOG_LIB})

set (GRAPH_VIDEO_DEMO_SRC
     face_video_graph.cpp)
add_executable(face_video_graph ${GRAPH_VIDEO_DEMO_SRC})
target_link_libraries(face_video_graph ${TBB_LIBS} vision_graph dl ${ANDROID_LOG_LIB})

set (MINI_GRAPH_SRC
     mini_graph.cpp)
add_executable(mini_graph ${MINI_GRAPH_SRC})
target_link_libraries(mini_graph ${TBB_LIBS} vision_graph dl ${ANDROID_LOG_LIB})

include_directories(${OpenCV_INCLUDE_DIRS})
set (DLCV_SNPE_TEST_SRC
     dlcv_snpe_test.cpp)
add_executable(dlcv_snpe_test ${DLCV_SNPE_TEST_SRC})
target_link_libraries(dlcv_snpe_test ${DLCVLIB} ${OpenCV_LIBRARIES} ${TBB_LIBS} dl ${ANDROID_LOG_LIB})