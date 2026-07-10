# TopAY Fixed-Goal Reproduction

This folder is a cleaned, minimal copy used to reproduce the TopAY baseline in the comparative evaluation. It keeps one example map and ten fixed goals.

## What Is Included

- Workspace root: this directory is a ROS catkin workspace.
- Example map: `src/simulator/random_map_generator/env/exp_pcd/world_x[-6.00,6.00]_y[-6.00,6.00]_obs120_20251219_061529/map.pcd`
- Fixed goals: `src/simulator/random_map_generator/env/exp_pcd/world_x[-6.00,6.00]_y[-6.00,6.00]_obs120_20251219_061529/goal_list.txt`
- Main launch file: `src/planner/launch/run_all.launch`

## Build

The original experiments used ROS Noetic on Ubuntu 20.04. The original `Dockerfile` is kept for dependency reference.

```bash
cd /path/to/comparative_repro/topay
catkin_make
source devel/setup.bash
```

## Run TopAY

Start the simulator, map loader, planner, and RViz:

```bash
cd /path/to/comparative_repro/topay
source devel/setup.bash
roslaunch planner run_all.launch rviz:=true
```

In a second terminal, trigger the fixed-goal batch once:

```bash
cd /path/to/comparative_repro/topay
source devel/setup.bash
rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped \
  '{header: {frame_id: "world"}, pose: {orientation: {w: 1.0}}}'
```

The planner ignores the clicked target position in this batch mode and reads the fixed goals from `goal_list.txt`.

## Output

Results are written to:

```text
src/simulator/random_map_generator/env/exp_pcd/world_x[-6.00,6.00]_y[-6.00,6.00]_obs120_20251219_061529/topay_results.txt
```

## Key Parameters

- Method switch: `src/planner/params/agent.yaml`
  - `planner_node.agent.mode: "planner_tro_exp"` enables fixed-goal batch evaluation.
  - `planner_node.agent.planner: "moma"` selects the TopAY baseline.
  - `planner_node.agent.planning_budget` controls per-goal planning time budget.
- Map and goals: `src/simulator/random_map_generator/params/map.yaml`
  - `map.case_id: 2` loads the fixed example PCD.
  - `map.pcd_folder` selects the example map folder.
  - `map.has_goal_set: true` reads `goal_list.txt`.
  - `map.num_goals: 10` limits the batch to ten goals.
- Robot and kinematic limits: `src/simulator/fake_moma/include/fake_moma/moma_param.h`
- Topological path search: `src/planner/params/topo_prm.yaml`
- Whole-body sampling: `src/planner/params/mcrrts.yaml`
- Trajectory optimization: `src/planner/params/optimizer.yaml`
- Controller settings: `src/planner/params/mpc.yaml`

## Optional GCDF-CRISP Mode

This copy can also drive the external CRISP solver used in the comparison. Set `planner_node.agent.planner: "crisp"` in `src/planner/params/agent.yaml`, then run the solver side separately. The planner publishes solver inputs on `solver_input_list` and expects outputs on `/swerve_base/solver_output_list`.
