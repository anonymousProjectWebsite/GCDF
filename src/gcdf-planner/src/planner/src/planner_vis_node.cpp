#include "planner/planner_vis.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "planner_vis_node");
  ros::NodeHandle nh("~");

  PlannerVis vis(nh);
  ros::spin();
  return 0;
}
