# REMANI Fixed-Goal Reproduction

This folder is a cleaned, minimal copy used to reproduce the REMANI baseline in the comparative evaluation. It keeps one example map and ten fixed goals.

## What Is Included

- Workspace root: this directory is a ROS catkin workspace.
- Planner package: `src/REMANI-Planner/remani_planner/plan_manage`
- Example map: `src/REMANI-Planner/remani_planner/plan_env/env/world_x[-6.00,6.00]_y[-6.00,6.00]_obs120_20251219_061529/map.pcd`
- Fixed goals: `src/REMANI-Planner/remani_planner/plan_env/env/world_x[-6.00,6.00]_y[-6.00,6.00]_obs120_20251219_061529/goal_list.txt`
- Main launch file: `src/REMANI-Planner/remani_planner/plan_manage/launch/exp0.launch`

## Build

The original experiments used ROS Noetic on Ubuntu 20.04.

```bash
cd /path/to/comparative_repro/remani
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

## Run REMANI

Start the simulator, map loader, planner, and RViz:

```bash
cd /path/to/comparative_repro/remani
source devel/setup.bash
roslaunch remani_planner exp0.launch
```

In a second terminal, trigger the fixed-goal batch once:

```bash
cd /path/to/comparative_repro/remani
source devel/setup.bash
rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped \
  '{header: {frame_id: "world"}, pose: {orientation: {w: 1.0}}}'
```

The planner ignores the clicked target position in this batch mode and reads the fixed goals from `goal_list.txt`.

## Output

Results are written to:

```text
src/REMANI-Planner/remani_planner/plan_env/env/world_x[-6.00,6.00]_y[-6.00,6.00]_obs120_20251219_061529/remani_results.txt
```

## Key Parameters

- Launch chain: `src/REMANI-Planner/remani_planner/plan_manage/launch/exp0.launch`
  - Includes `run_in_sim.launch`, which starts the planner, simulator, map generator, and RViz.
- Map and goals: `src/REMANI-Planner/remani_planner/plan_manage/launch/run_in_sim.launch`
  - `map/pcd_folder` selects the example map folder.
  - `map_resolution`, `map_size_x`, `map_size_y`, and `map_size_z` set the map dimensions.
- Robot model and limits: `src/REMANI-Planner/remani_planner/plan_manage/config/mm_param.yaml`
  - Mobile base geometry, wheel limits, manipulator degrees of freedom, and joint limits are set here.
- Planner/search/optimization: `src/REMANI-Planner/remani_planner/plan_manage/config/remani_planner_param.yaml`
  - Hybrid A* search, whole-body RRT fallback, optimization weights, safety margins, and global planning mode are set here.
- Batch goal count and result writer: `src/REMANI-Planner/remani_planner/plan_manage/src/remani_replan_fsm.cpp`
  - This copy uses ten goals to keep the example lightweight.

## Notes

- `fsm.global_plan: true` in `remani_planner_param.yaml` makes the planner use the global map published by the map generator.
- The map generator reads the same fixed map folder used by the TopAY copy, so both baselines run on the same point cloud and goal list.
