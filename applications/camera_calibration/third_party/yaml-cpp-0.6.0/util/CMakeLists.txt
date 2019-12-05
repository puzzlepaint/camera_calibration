add_sources(parse.cpp)
add_executable(parse2 parse.cpp)
target_link_libraries(parse2 yaml-cpp)
set_target_properties(parse2 PROPERTIES COMPILE_FLAGS "-std=c++11")

add_sources(sandbox.cpp)
add_executable(sandbox2 sandbox.cpp)
target_link_libraries(sandbox2 yaml-cpp)
set_target_properties(sandbox2 PROPERTIES COMPILE_FLAGS "-std=c++11")

add_sources(read.cpp)
add_executable(read2 read.cpp)
target_link_libraries(read2 yaml-cpp)
set_target_properties(read2 PROPERTIES COMPILE_FLAGS "-std=c++11")
