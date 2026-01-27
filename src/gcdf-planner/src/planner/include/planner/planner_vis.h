#pragma once

#include <ros/ros.h>
#include <ros/package.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PoseStamped.h>

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <atomic>

#include "planner/SolverInputList.h"
#include "mm_kinova/moma_param.h"

class PlannerVis
{
public:
  PlannerVis(ros::NodeHandle& nh);
  void solverOutputCb(const planner::SolverInputListConstPtr& msg);
  void goalTriggerCb(const geometry_msgs::PoseStamped::ConstPtr& msg);

private:
  void loadGoals();
  void visPathMesh(const std::vector<Eigen::VectorXd>& path,
                   ros::Publisher& pub,
                   const std::vector<float>& rgba,
                   int id);
  void visEeTraj(const std::vector<Eigen::VectorXd>& path,
                 ros::Publisher& pub,
                 const std::vector<float>& rgba);
  void visCollisionSpheres(const std::vector<Eigen::VectorXd>& path,
                           ros::Publisher& pub,
                           const std::vector<float>& rgba,
                           int id);

  ros::NodeHandle nh_;
  ros::Subscriber solver_sub_;
  ros::Subscriber goal_sub_;
  ros::Publisher tracking_traj_pub_;
  ros::Publisher ee_traj_pub_;
  ros::Publisher colli_spheres_pub_;
  ros::Publisher solver_input_pub_;

  std::string goal_list_path_;
  std::string map_name_;
  int num_goal_ = 50;
  bool show_goal_set_ = true;
  bool show_collision_spheres_ = false;
  int state_dim_ = 9;
  int horizon_len_ = 40;
  std::vector<double> init_state_;

  bool goals_loaded_ = false;
  std::vector<Eigen::VectorXd> goal_list_;

  MomaParam moma_param_;
  mutable std::atomic<int> traj_vis_counter_{0};
};
