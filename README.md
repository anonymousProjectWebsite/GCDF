# Comparative Evaluation Reproduction Package

This directory contains cleaned, lightweight copies of the two baseline workspaces used for fixed-goal comparative evaluation.

## Contents

- `topay/`: TopAY baseline workspace with one example map and ten fixed goals.
- `remani/`: REMANI baseline workspace with the same example map and ten fixed goals.

Both copies are configured to load:

```text
world_x[-6.00,6.00]_y[-6.00,6.00]_obs120_20251219_061529
```

## How To Run

Use the method-specific README files:

- `topay/README.md`
- `remani/README.md`

Each README lists the launch command, trigger command, output path, and key parameter files.


## Configuration Summary

| Method | Workspace | Main Launch | Map/Goal Config | Core Parameters | Output |
| --- | --- | --- | --- | --- | --- |
| TopAY | `topay/` | `src/planner/launch/run_all.launch` | `src/simulator/random_map_generator/params/map.yaml` | `src/planner/params/agent.yaml`, `src/planner/params/topo_prm.yaml`, `src/planner/params/mcrrts.yaml`, `src/planner/params/optimizer.yaml` | `src/simulator/random_map_generator/env/exp_pcd/<map>/topay_results.txt` |
| REMANI | `remani/` | `src/REMANI-Planner/remani_planner/plan_manage/launch/exp0.launch` | `src/REMANI-Planner/remani_planner/plan_manage/launch/run_in_sim.launch` | `src/REMANI-Planner/remani_planner/plan_manage/config/mm_param.yaml`, `src/REMANI-Planner/remani_planner/plan_manage/config/remani_planner_param.yaml` | `src/REMANI-Planner/remani_planner/plan_env/env/<map>/remani_results.txt` |


