#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pcd_map_node");
  ros::NodeHandle nh("~");

  std::string frame_id = "world";
  std::string map_name;
  double resolution = 0.1;
  double vis_rate = 10.0;
  double z_lift = 0.0;

  nh.param("frame_id", frame_id, frame_id);
  nh.param("map_name", map_name, std::string(""));
  nh.param("resolution", resolution, resolution);
  nh.param("vis_rate", vis_rate, vis_rate);

  if (map_name.empty()) {
    ROS_ERROR("[pcd_map_node] map_name must be set.");
    return 1;
  }
  std::string pcd_path = ros::package::getPath("planner") + "/env/" + map_name + "/map.pcd";

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path, *cloud_in) == -1) {
    ROS_ERROR("[pcd_map_node] Failed to read PCD file: %s", pcd_path.c_str());
    return 1;
  }

  pcl::PointCloud<pcl::PointXYZ> cloud_map;
  if (resolution > 1e-6) {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud_in);
    vg.setLeafSize(resolution, resolution, resolution);
    vg.filter(cloud_map);
  } else {
    cloud_map = *cloud_in;
  }

  cloud_map.width = cloud_map.points.size();
  cloud_map.height = 1;
  cloud_map.is_dense = true;

  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud_map, cloud_msg);
  cloud_msg.header.frame_id = frame_id;

  ros::Publisher map_pub = nh.advertise<sensor_msgs::PointCloud2>("global_cloud", 1, true);

  ros::Rate rate(vis_rate);
  while (ros::ok()) {
    cloud_msg.header.stamp = ros::Time::now();
    map_pub.publish(cloud_msg);
    ros::spinOnce();
    rate.sleep();
  }

  return 0;
}
