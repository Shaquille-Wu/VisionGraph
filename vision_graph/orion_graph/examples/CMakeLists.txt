cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)
set(CMAKE_VERBOSE_MAKEFILE  ON)

add_executable(sub_string_finder sub_string_finder.cpp)
target_link_libraries(sub_string_finder ${TBB_LIBS} dl ${ANDROID_LOG_LIB})

#add_executable(dining_philosophers dining_philosophers.cpp)
#target_link_libraries(dining_philosophers ${TBB_LIBS} dl ${ANDROID_LOG_LIB})

add_executable(hello_graph hello_graph.cpp)
target_link_libraries(hello_graph ${TBB_LIBS} dl ${ANDROID_LOG_LIB} )

add_executable(orion_simple_flow orion_simple_flow.cpp)
target_link_libraries(orion_simple_flow ${TBB_LIBS} dl ${ANDROID_LOG_LIB})

add_executable(condition_multifunction_node condition_multifunction_node.cpp)
target_link_libraries(condition_multifunction_node ${TBB_LIBS} dl ${ANDROID_LOG_LIB})

add_executable(vision_flow_graph vision_flow_graph.cpp)
target_link_libraries(vision_flow_graph ${TBB_LIBS} dl)

add_executable(tbb_parallel_for tbb_parallel_for.cpp)
target_link_libraries(tbb_parallel_for ${TBB_LIBS} dl)

add_executable(parallel_task parallel_task.cpp)
target_link_libraries(parallel_task ${TBB_LIBS} dl)