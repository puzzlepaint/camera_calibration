cmake_minimum_required(VERSION 3.6...3.14)
project(generic_models LANGUAGES C CXX)

# Use a release build (with debug info) by default, since it runs slowly with a debug build
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not specified, using RelWithDebInfo")
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# Enable C++11 globally
set(CMAKE_CXX_STANDARD 11)

# Require external library: Eigen3
find_package(Eigen3 REQUIRED)

# Define generic_models executable
add_executable(generic_models
  src/central_generic.h
  src/central_generic_unprojection_jacobian.h
  src/main.cc
  src/noncentral_generic.h
  src/noncentral_generic_unprojection_jacobian.h
  src/util.h
)
set_target_properties(generic_models PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories(generic_models PUBLIC
  src
  ${EIGEN3_INCLUDE_DIR}
)
