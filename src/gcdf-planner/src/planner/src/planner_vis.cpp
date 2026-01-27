#include "planner/planner_vis.h"

#include <ros/package.h>
#include <geometry_msgs/PoseStamped.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

PlannerVis::PlannerVis(ros::NodeHandle& nh) : nh_(nh)
{
  nh_.param("goal_list_path", goal_list_path_, std::string(""));
  nh_.param("map_name", map_name_, std::string(""));
  nh_.param("num_goal", num_goal_, 10);
  nh_.param("show_goal_set", show_goal_set_, true);
  nh_.param("show_collision_spheres", show_collision_spheres_, false);
  nh_.param("state_dim", state_dim_, 9);
  nh_.param("horizon_len", horizon_len_, 40); // optimization horizon

  if (goal_list_path_.empty() && !map_name_.empty()) {
    goal_list_path_ = ros::package::getPath("planner") + "/env/" + map_name_ + "/goal_list.txt";
  }

  init_state_.assign(state_dim_, 0.0);
  std::vector<double> init_state_param;
  if (nh_.getParam("init_state", init_state_param) && !init_state_param.empty()) {
    for (size_t i = 0; i < init_state_param.size() && i < init_state_.size(); ++i) {
      init_state_[i] = init_state_param[i];
    }
  }

  tracking_traj_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("tracking_traj", 1, true);
  ee_traj_pub_ = nh_.advertise<visualization_msgs::Marker>("ee_traj", 1, true);
  colli_spheres_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("collision_spheres", 1, true);
  solver_input_pub_ = nh_.advertise<planner::SolverInputList>("solver_input_list", 1, true);

  solver_sub_ = nh_.subscribe("solver_output_list", 1, &PlannerVis::solverOutputCb, this);
  goal_sub_ = nh_.subscribe("/move_base_simple/goal", 1, &PlannerVis::goalTriggerCb, this);

  // Show the robot at initial pose on startup.
  Eigen::VectorXd init_vec(state_dim_);
  for (int i = 0; i < state_dim_; ++i) {
    init_vec(i) = init_state_[i];
  }
  visPathMesh({init_vec}, tracking_traj_pub_, {0.6f, 0.6f, 0.6f, 0.6f}, 100);
  // visualize collision ball at initial pose
  if (show_collision_spheres_) {
    visCollisionSpheres({init_vec}, colli_spheres_pub_, {0.93f, 0.35f, 0.35f, 0.25f}, 500);
  }
}

void PlannerVis::loadGoals()
{
  if (goals_loaded_) {
    return;
  }

  if (goal_list_path_.empty()) {
    ROS_WARN("[PlannerVis] goal_list_path is empty, skip loading goals.");
    goals_loaded_ = true;
    return;
  }

  std::ifstream goal_file(goal_list_path_);
  if (!goal_file.is_open()) {
    ROS_ERROR("[PlannerVis] Cannot open goal file: %s", goal_list_path_.c_str());
    goals_loaded_ = true;
    return;
  }

  std::vector<Eigen::VectorXd> raw_list;
  std::string line;
  int count = 0;
  while (std::getline(goal_file, line) && raw_list.size() < static_cast<size_t>(num_goal_)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream iss(line);
    Eigen::VectorXd goal_state(3 + moma_param_.dof_num);
    for (int i = 0; i < 3 + static_cast<int>(moma_param_.dof_num); i++) {
      iss >> goal_state(i);
    }
    raw_list.push_back(goal_state);
    count++;
  }
  goal_file.close();

  goal_list_ = raw_list;

  if (show_goal_set_ && !goal_list_.empty()) {
    visPathMesh(goal_list_, tracking_traj_pub_, {0.72f, 0.78f, 0.91f, 1.0f}, 2020);
  }

  goals_loaded_ = true;
}

void PlannerVis::solverOutputCb(const planner::SolverInputListConstPtr& msg)
{
  loadGoals();

  const int traj_number = static_cast<int>(msg->solver_input_list.size());
  int vis_id = 0;

  for (int i = 0; i < traj_number; i++) {
    const planner::SolverInput& solver_output = msg->solver_input_list[i];
    std::vector<Eigen::VectorXd> traj_states_truncated, traj_states;
    const int step_num = static_cast<int>(solver_output.stateTrajectory.size());

    for (int j = 0; j < step_num; j++) {
      const planner::State& state_msg = solver_output.stateTrajectory[j];
      Eigen::VectorXd state_vec(state_msg.data.size());
      for (size_t k = 0; k < state_msg.data.size(); k++) {
        state_vec(k) = state_msg.data[k];
      }
      traj_states.push_back(state_vec);
      if (j % 4 != 0) {
        continue;
      }
      traj_states_truncated.push_back(state_vec);
    }

    if (!goal_list_.empty() && i < static_cast<int>(goal_list_.size())) {
      traj_states.push_back(goal_list_[i]);
    } else if (goal_list_.empty()) {
      ROS_WARN_THROTTLE(1.0, "[PlannerVis] goal_list is empty, skip appending goal state.");
    } else {
      ROS_WARN_THROTTLE(1.0, "[PlannerVis] goal_list index %d out of range (size=%zu).", i, goal_list_.size());
    }

    visPathMesh(traj_states_truncated, tracking_traj_pub_,
                {0.97f, 0.92f, 0.72f, 0.1f}, (vis_id + 3000) * 1000);
    vis_id++;
    visEeTraj(traj_states, ee_traj_pub_, {0.72f, 0.78f, 0.91f, 0.5f});
    if (show_collision_spheres_) {
      visCollisionSpheres(traj_states_truncated, colli_spheres_pub_,
                          {0.93f, 0.35f, 0.35f, 0.25f}, (vis_id + 6000) * 1000);
    }
  }
}

void PlannerVis::goalTriggerCb(const geometry_msgs::PoseStamped::ConstPtr& /*msg*/)
{
  loadGoals();
  if (goal_list_.empty()) {
    ROS_WARN("[PlannerVis] goal_list is empty, skip publishing solver input.");
    return;
  }

  planner::SolverInputList solver_input_list_msg;
  solver_input_list_msg.solver_input_list.resize(goal_list_.size());

  for (size_t i = 0; i < goal_list_.size(); ++i) {
    planner::SolverInput solver_input_msg;
    // check is the size of x0 and xf match state_dim_
    if (static_cast<int>(solver_input_msg.x0.data.size()) != state_dim_) {
      ROS_ERROR("[PlannerVis] state_dim (%d) does not match SolverInput x0 size (%zu).",
                state_dim_, solver_input_msg.x0.data.size());
    }

    for (int j = 0; j < state_dim_; ++j) {
      solver_input_msg.x0.data[j] = init_state_[j];
      solver_input_msg.xf.data[j] = goal_list_[i](j);
    }

    solver_input_msg.stateTrajectory.resize(horizon_len_);
    for (int k = 0; k < horizon_len_; ++k) {
      double alpha = (horizon_len_ <= 1) ? 1.0 : static_cast<double>(k) / (horizon_len_ - 1);
      for (int j = 0; j < state_dim_; ++j) {
        solver_input_msg.stateTrajectory[k].data[j] =
            (1.0 - alpha) * init_state_[j] + alpha * goal_list_[i](j); // you can use whatever initialization strategy via this interface, I put a simple linear interpolation here
      }
    }

    solver_input_list_msg.solver_input_list[i] = solver_input_msg;
  }

  solver_input_pub_.publish(solver_input_list_msg);
  ROS_INFO("[PlannerVis] Published SolverInputList with %zu goals.", goal_list_.size());
}

void PlannerVis::visPathMesh(const std::vector<Eigen::VectorXd>& path,
                             ros::Publisher& pub,
                             const std::vector<float>& rgba,
                             int id)
{
  if (path.empty()) {
    std::cout << "[PlannerVis] visPathMesh: empty path, skip visualization." << std::endl;
    return;
  }

  visualization_msgs::MarkerArray moma_marker;
  for (const auto& moma_pos : path) {
    std::vector<Eigen::Vector3d> joint_pos, joint_axis;
    std::vector<Eigen::Vector4d> sphere_pos_radius;
    std::vector<Eigen::Vector3d> p_all_link;
    std::vector<Eigen::Matrix3d> R_all_link;
    moma_param_.computeKinematics(moma_pos, joint_pos, joint_axis, sphere_pos_radius, &p_all_link, &R_all_link);

    if (p_all_link.empty() || R_all_link.empty()) {
      continue;
    }

    const Eigen::Vector3d& p_base = p_all_link[0];
    const Eigen::Matrix3d& R_base = R_all_link[0];
    Eigen::Quaterniond q_base(R_base);

    visualization_msgs::Marker base_marker;
    base_marker.header.frame_id = "world";
    base_marker.id = id++;
    base_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
    base_marker.action = visualization_msgs::Marker::ADD;
    base_marker.mesh_resource = moma_param_.chassis_mesh;

    base_marker.pose.position.x = p_base.x();
    base_marker.pose.position.y = p_base.y();
    base_marker.pose.position.z = p_base.z();
    base_marker.pose.orientation.w = q_base.w();
    base_marker.pose.orientation.x = q_base.x();
    base_marker.pose.orientation.y = q_base.y();
    base_marker.pose.orientation.z = q_base.z();

    base_marker.color.a = rgba[3];
    base_marker.color.r = rgba[0];
    base_marker.color.g = rgba[1];
    base_marker.color.b = rgba[2];

    base_marker.scale.x = 1.0;
    base_marker.scale.y = 1.0;
    base_marker.scale.z = 1.0;

    moma_marker.markers.push_back(base_marker);

    const size_t n_frames = p_all_link.size();
    for (size_t frame_id = 1; frame_id < n_frames; ++frame_id) {
      const Eigen::Vector3d& pL = p_all_link[frame_id];
      const Eigen::Matrix3d& RL = R_all_link[frame_id];
      Eigen::Quaterniond qL(RL);

      visualization_msgs::Marker link_marker;
      link_marker.header.frame_id = "world";
      link_marker.id = id++;
      link_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
      link_marker.action = visualization_msgs::Marker::ADD;
      int link_idx = static_cast<int>(frame_id);
      if (link_idx - 1 < static_cast<int>(moma_param_.link_meshes.size())) {
        link_marker.mesh_resource = moma_param_.link_meshes[link_idx - 1];
      }

      link_marker.pose.position.x = pL.x();
      link_marker.pose.position.y = pL.y();
      link_marker.pose.position.z = pL.z();
      link_marker.pose.orientation.w = qL.w();
      link_marker.pose.orientation.x = qL.x();
      link_marker.pose.orientation.y = qL.y();
      link_marker.pose.orientation.z = qL.z();

      link_marker.color.a = rgba[3];
      link_marker.color.r = rgba[0];
      link_marker.color.g = rgba[1];
      link_marker.color.b = rgba[2];

      link_marker.scale.x = 1.0;
      link_marker.scale.y = 1.0;
      link_marker.scale.z = 1.0;

      moma_marker.markers.push_back(link_marker);
    }
  }

  if (!moma_marker.markers.empty()) {
    std::cout << "[PlannerVis] visPathMesh: publishing " << moma_marker.markers.size() << " markers." << std::endl;
    pub.publish(moma_marker);
  }
}

void PlannerVis::visEeTraj(const std::vector<Eigen::VectorXd>& path,
                           ros::Publisher& pub,
                           const std::vector<float>& rgba)
{
  if (path.empty()) {
    return;
  }

  visualization_msgs::Marker line_strip;
  line_strip.header.frame_id = "world";
  line_strip.header.stamp = ros::Time::now();
  line_strip.ns = "ee_trajectory";
  line_strip.action = visualization_msgs::Marker::ADD;
  line_strip.pose.orientation.w = 1.0;
  line_strip.id = traj_vis_counter_.fetch_add(1) + 2077;
  line_strip.type = visualization_msgs::Marker::LINE_STRIP;
  line_strip.scale.x = 0.10;
  line_strip.scale.y = 0.10;
  line_strip.scale.z = 0.10;

  Eigen::Vector4d gripper_prev = moma_param_.getColliPts(path.front()).back();
  Eigen::Vector4d rgba_random = {
  1.0 * (rand() % 1000) / 1000.0, 
  1.0 * (rand() % 1000) / 1000.0, 
  1.0 * (rand() % 1000) / 1000.0, 
  1.0};
  for (size_t i = 0; i < path.size(); i++) {
    Eigen::Vector4d gripper = moma_param_.getColliPts(path[i]).back();

    geometry_msgs::Point pt;
    pt.x = gripper(0);
    pt.y = gripper(1);
    pt.z = gripper(2);
    line_strip.points.push_back(pt);

    gripper_prev = gripper;
    // use random color
    std_msgs::ColorRGBA color;
    color.r = rgba_random[0];
    color.g = rgba_random[1];
    color.b = rgba_random[2];
    color.a = rgba_random[3];
    line_strip.colors.push_back(color);
  }

  pub.publish(line_strip);
}

void PlannerVis::visCollisionSpheres(const std::vector<Eigen::VectorXd>& path,
                                     ros::Publisher& pub,
                                     const std::vector<float>& rgba,
                                     int id)
{
  if (path.empty()) {
    return;
  }
  visualization_msgs::MarkerArray marker_array;
  marker_array = moma_param_.getColliMarkerArray(path.front());
  if (!marker_array.markers.empty()) {
    pub.publish(marker_array);
  }
}
