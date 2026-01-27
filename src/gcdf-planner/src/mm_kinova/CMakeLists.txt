cmake_minimum_required(VERSION 3.0.2)
project(mm_kinova)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nav_msgs
  visualization_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp std_msgs nav_msgs visualization_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)
