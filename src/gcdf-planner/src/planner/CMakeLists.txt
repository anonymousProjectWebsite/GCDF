cmake_minimum_required(VERSION 3.10)
project(planner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
  message_generation
  mm_kinova
  pcl_conversions
)

find_package(PCL REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(CPPAD REQUIRED cppad)
find_package(casadi CONFIG REQUIRED)
find_package(CRISP REQUIRED)
# print if find crisp
message(STATUS "Solver include dir: ${CRISP_INCLUDE_DIR}")
message(STATUS "Solver library dir: ${CRISP_LIBRARIES}")
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS
system
filesystem
log_setup
log
)

add_message_files(FILES
  State.msg
  SolverInput.msg
  SolverInputList.msg
)

generate_messages(DEPENDENCIES
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp roslib std_msgs nav_msgs geometry_msgs sensor_msgs visualization_msgs message_runtime mm_kinova pcl_conversions
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${CRISP_INCLUDE_DIRS}                 
  ${EIGEN3_INCLUDE_DIR}     
  ${CPPAD_INCLUDE_DIRS}
)

add_executable(planner_vis_node
  src/planner_vis_node.cpp
  src/planner_vis.cpp
)

add_dependencies(planner_vis_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(planner_vis_node
  ${catkin_LIBRARIES}
)

add_executable(pcd_map_node
  src/pcd_map_node.cpp
)

add_dependencies(pcd_map_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(pcd_map_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)


add_executable(GenerateProblemMobileManipulator src/GenerateProblemMobileManipulator.cpp)

target_link_libraries(GenerateProblemMobileManipulator
${CRISP_LIBRARIES}
${Boost_LIBRARIES}
${EIGEN3_LIBRARIES}
${CPPAD_LIBRARIES}
-ldl
casadi::casadi
)

catkin_install_python(PROGRAMS
  script/SolveMobileManipulatorExp.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


