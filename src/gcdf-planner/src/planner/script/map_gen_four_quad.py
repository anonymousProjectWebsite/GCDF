#!/usr/bin/env python3
import rospy
from gazebo_msgs.srv import SpawnModel
import random
import math
from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.srv import GetWorldProperties, DeleteModel
import os
from datetime import datetime
EXP_WORLD_DIR = "PATH-TO-YOUR-FOLDER"
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def model_sdf_snippet(name, size, pose_xyz, rpy, color="Gazebo/White"):
    x, y, z = pose_xyz
    rr, pp, yy = rpy
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} {z} {rr} {pp} {yy}</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>{size[0]} {size[1]} {size[2]}</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>{size[0]} {size[1]} {size[2]}</size></box>
          </geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <name>{color}</name>
            </script>
          </material>
        </visual>
      </link>
    </model>
    """


def save_world_file(models, x_min, x_max, y_min, y_max, obs_num):
    """
    models: list of dicts: {name, size, xyz, rpy, color}
    """
    ensure_dir(EXP_WORLD_DIR)

    map_size = f"x[{x_min:.2f},{x_max:.2f}]_y[{y_min:.2f},{y_max:.2f}]"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"world_{map_size}_obs{obs_num}_{ts}.world"
    out_path = os.path.join(EXP_WORLD_DIR, fname)

    body = "\n".join([
        model_sdf_snippet(m["name"], m["size"], m["xyz"], m["rpy"], m.get("color", "Gazebo/White"))
        for m in models
    ])

    world = f"""<sdf version="1.6">
  <world name="default">
    {body}
  </world>
</sdf>
"""
    with open(out_path, "w") as f:
        f.write(world)

    print(f"[OK] World saved to: {out_path}")
    return out_path


def get_model_list():
    rospy.wait_for_service('/gazebo/get_world_properties')
    try:
        get_world_properties = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        world_properties = get_world_properties()
        return world_properties.model_names
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
        return []

def delete_model(model_name):
    rospy.wait_for_service('/gazebo/delete_model')
    try:
        delete_model_service = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        delete_model_service(model_name)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
        return
    
def clear_all_models():
    model_list = get_model_list()  
    for model_name in model_list:
        delete_model(model_name) 

def generate_box_sdf(size=[1.0, 1.0, 1.0], pose=[0.0, 0.0, 0.0], rpy=[0.0, 0.0, 0.0], color="Gazebo/Red"):
    sdf_template = f"""
    <sdf version="1.6">
      <model name="box">
        <static>true</static>
        <link name="link">
          <pose>{pose[0]} {pose[1]} {pose[2]} {rpy[0]} {rpy[1]} {rpy[2]}</pose>
          <collision name="collision">
            <geometry>
              <box>
                <size>{size[0]} {size[1]} {size[2]}</size>
              </box>
            </geometry>
          </collision>
          <visual name="visual">
            <geometry>
              <box>
                <size>{size[0]} {size[1]} {size[2]}</size>
              </box>
            </geometry>
            <material>
              <script>
                <uri>file://media/materials/scripts/gazebo.material</uri>
                <name>{color}</name>
              </script>
            </material>
          </visual>
        </link>
      </model>
    </sdf>
    """
    return sdf_template

def check_valid_obs(obstacles, new_obs, min_dist=1.0):
    if new_obs[0] < 1.0 and new_obs[0] > -1.0:
       if new_obs[1] < 1.0 and new_obs[1] > -1.0:
          return False
    for obs in obstacles:
        dist = ((obs[0]-new_obs[0])**2 + (obs[1]-new_obs[1])**2)**0.5
        if dist < min_dist:
            return False
    return True


def quadrant_of(x, y):
  # 0: x>=0,y>=0 ; 1: x<0,y>=0 ; 2: x<0,y<0 ; 3: x>=0,y<0
  if x >= 0 and y >= 0:
    return 0
  if x < 0 and y >= 0:
    return 1
  if x < 0 and y < 0:
    return 2
  return 3


def sample_in_quadrant(q, x_min, x_max, y_min, y_max):
  # Return (x,y) sampled in quadrant q while avoiding the central square
  if q == 0:
    x = random.uniform(max(0.01, 0.0), x_max)
    y = random.uniform(max(0.01, 0.0), y_max)
  elif q == 1:
    x = random.uniform(x_min, min(-0.01, 0.0))
    y = random.uniform(max(0.01, 0.0), y_max)
  elif q == 2:
    x = random.uniform(x_min, min(-0.01, 0.0))
    y = random.uniform(y_min, min(-0.01, 0.0))
  else:
    x = random.uniform(max(0.01, 0.0), x_max)
    y = random.uniform(y_min, min(-0.01, 0.0))
  return x, y

# rospy.wait_for_service('/gazebo/spawn_sdf_model')
clear_all_models()
rospy.init_node('map_gen', anonymous=True)
rospy.wait_for_service('/gazebo/spawn_sdf_model')
spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
num_blocks_ = [80,100,120]
x_min, x_max = -6.0, 6.0
y_min, y_max = -6.0, 6.0
size = [0.3, 0.3, 0.3]
num_worlds = 5
max_attempts = 500
obstacles_all_floating = []
obstacles_all_grouding = []
quadrant_counts = [0, 0, 0, 0]
models_meta = []
for num_blocks in num_blocks_:
  for _ in range(num_worlds):
    clear_all_models()
    models_meta.clear()
    for i in range(num_blocks):
      print("Spawning block {}".format(i))

      # choose min distance based on block type
      if i < num_blocks / 2:
        min_dist = 1.0
        size = [random.uniform(0.5, 1.5), 0.3, 0.3]
        rpy = [0, 0, random.uniform(0, 3.14 / 2)]
        color = "Gazebo/White"
        # use floating flag to choose height: floating -> random higher z, else sit on ground
        z = random.uniform(0.9, 1.4)
          # try to sample in the least populated quadrant first
        count = 0
        while True:
          q = quadrant_counts.index(min(quadrant_counts))
          sample = sample_in_quadrant(q, x_min, x_max, y_min, y_max)
          if sample is None:
              continue
          x, y = sample
          if check_valid_obs(obstacles_all_floating, [x, y], min_dist=min_dist):
            obstacles_all_floating.append([x, y])
            quadrant_counts[quadrant_of(x, y)] += 1
            break
          count += 1
          if count > max_attempts:
            obstacles_all_floating.append([x, y])
            quadrant_counts[quadrant_of(x, y)] += 1
            break
      else:
        min_dist = 1.0
        size = [random.uniform(0.2, 0.3), random.uniform(0.2, 0.3), random.uniform(0.5, 1.0)]
        z = size[2] / 2.0
        rpy = [0, 0, 0]
        color = "Gazebo/White"
        count = 0
        while True:
          q = quadrant_counts.index(min(quadrant_counts))
          sample = sample_in_quadrant(q, x_min, x_max, y_min, y_max)
          if sample is None:
              continue
          x, y = sample
          if check_valid_obs(obstacles_all_grouding, [x, y], min_dist=min_dist):
            obstacles_all_grouding.append([x, y])
            quadrant_counts[quadrant_of(x, y)] += 1
            break
          count += 1
          if count > max_attempts:
            obstacles_all_floating.append([x, y])
            quadrant_counts[quadrant_of(x, y)] += 1
            break
      sdf = generate_box_sdf(size, pose=[x, y, z], rpy=rpy, color=color)
      # spawn at sampled pose (use same x,y,z)
      spawn_model(model_name="block{}".format(i), model_xml=sdf,
        robot_namespace="", initial_pose=Pose(position=Point(0, 0, 0)), reference_frame="world")
      models_meta.append({
        "name": f"block{i}",
        "size": size,
        "xyz": [x, y, z],
        "rpy": rpy,
        "color": color
      })

    save_world_file(models_meta, x_min, x_max, y_min, y_max, obs_num=num_blocks)
    rospy.sleep(1)  # wait a bit before next world generation



