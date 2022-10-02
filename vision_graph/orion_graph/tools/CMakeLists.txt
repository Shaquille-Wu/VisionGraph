cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)
set(CMAKE_VERBOSE_MAKEFILE  ON)

set (DRAW_GRAPH_SRC
     draw_graph.cpp
     string_func.cpp
     ../vision_graph/vision_graph.cpp
     ../vision_graph/sub_graph.cpp)

add_executable(draw_graph ${DRAW_GRAPH_SRC})
target_link_libraries(draw_graph ${TBB_LIBS} ${DLCVLIB} dl)

install (TARGETS draw_graph DESTINATION tools)