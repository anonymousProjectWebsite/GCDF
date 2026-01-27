# Fast and Safe Trajectory Optimization for Mobile Manipulators With Neural Configuration Space Distance Field

This repository provides an open-source implementation of a fast and robust trajectory optimization framework for mobile manipulators, built upon neural Generalized Configuration Space Distance Fields (GCDF).

### Features

- **Generalized Configuration Space Distance Field (GCDF)**  
  Extend CDF from fixed-base arms to mobile manipulators with translational + rotational joints, while preserving core optimization-friendly properties.

- **Neural GCDF Representation**  
  Scalable data generation and training for neural GCDFs in unbounded workspaces, enabling accurate values/gradients and fast GPU-parallel queries.

- **High-Performance Optimization Solver**  
  Open-source C++ sequential convex solver with neural implicit GCDF constraints, supporting online constraint updates, batched GPU evaluation, and sparsity-aware scaling to thousands of constraints.

- **Fast Whole-Body Planning**  
  Rapid, smooth, collision-free whole-body trajectories from naive initial guesses, validated in high-density randomized clutter and real-world experiments.



## Table of Contents

- [Features](#features)
- [1. Download Repository](#1-download-repository)
- [2. Train the GCDF Model](#2-train-the-gcdf-model)
  - [Training Preparation](#training-preparation)
  - [Contents](#contents)
  - [Quick Start](#quick-start)
  - [Notes](#notes)
  - [Outputs](#outputs)
- [3. Solver & Planner Installations](#3-solver--planner-installations)
  - [Third Party Libraries](#third-party-libraries)
    - [Casadi](#casadi)
    - [PIQP](#piqp)
    - [L4Casadi](#l4casadi)
- [4. Build the Planner](#4-build-the-planner)
- [5. Usage](#5-usage)

## 1. Download Repository


```bash
git clone https://github.com/anonymousProjectWebsite/GCDF.git
cd GCDF
```

## 2. Train the GCDF model
Generalized Configuration Space Distance Fields (GCDF) training and evaluation for your mobile manipulators.
This process goes seperately the the subsequent solving and planning. We already prepare the trained model used in our simulations in the rep.

### Training Preparation
In host or your own conda env:
```
cd GCDF/src/gcdf-training
```
Install dependencies from `requiements.txt` (note the filename spelling):

```
pip install -r requiements.txt
```

### Contents

- `nn_gcdf.py`: GCDF training, inference, and evaluation.
- `bf_sdf.py`: BP-SDF model utilities and SDF queries.
- `data_generator.py`: Offline data generation for CDF training.
- `resource/`: Robot meshes, URDF, and pretrained BP-SDF models.

### Quick Start

1. Prepare data (optional if `data.pt` already exists):
   - Run `data_generator.py` to generate `data.npy`.
   - Run `nn_gcdf.py` with `self.process_data(...)` enabled to create `data.pt`.

2. Train GCDF:
   - Run `nn_gcdf.py` and call `train_nn(...)`.
   - Multi-GPU (optional): edit `self.gpulist` in `nn_gcdf.py` to list GPU IDs (e.g., `[0, 1, 2]`). The training decode step will be split across these GPUs; 

3. Evaluate:
   - Use `eval_nn(...)`, `eval_nn_noise(...)`, or `check_model(...)` in `nn_gcdf.py`.

### Notes

- `bf_sdf.py` can be used  load SDF models for robot links.
- Pretrained SDF models are stored under `resource/models/`. If you want to train your own robot sdf model, we refer to the [RDF](https://github.com/yimingli1998/RDF) repo; our implementation of the SDF training pipeline is based on the methodology described there.
- If you use Weights & Biases, make sure you are logged in before training.

### Outputs

- `data.npy`, `data.pt`: processed datasets.
- `model_dict.pt` and `model_gcdf.pth`: training checkpoints and final model. 

  Your trained pth file should finally be put inside ``GCDF\src\gcdf-solver\scripts``, and then you are done with the training process.

## 3. Solver & Planner Installtions
We highly suggest starting from the docker image we provided from now.
```sh
cd GCDF
docker build -t gcdf-image .
docker run --gpus all -it \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/home/robot/workspace \
  --name gcdf_container \
  gcdf-image
```

### Third Party Libraries
In your container:
#### Casadi
Install Casadi from source with python package to avoid conflict for L4casadi and the loading part in our solver
```sh
    git clone https://github.com/casadi/casadi.git
    cd casadi
    git checkout 3.7.2
    mkdir build
    cd build
    sudo apt install swig --install-recommends
    cmake -DWITH_PYTHON=ON -DWITH_PYTHON3=ON ..
    make
    sudo make install
```


#### PIQP
Install PIQP 
```sh
    git clone https://github.com/PREDICT-EPFL/piqp.git
    cd piqp
    git checkout v0.5.0 
    (the latest version has unknown incompatiablity with Eigen)
    mkdir build
    cd build
    cmake .. -DCMAKE_CXX_FLAGS="-march=native" -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF
    cmake --build . --config Release
    sudo cmake --install . --config Release
``` 

#### L4Casadi
Install from source to avoid potential API incompatiable. (if you have two casadi lib in your system, there will be conflict when you want load the casadi lib generated by l4casadi)
```sh
pip install torch
git clone https://github.com/Tim-Salzmann/l4casadi.git
cd l4casadi
pip install -r requirements_build.txt
pip install -U importlib-metadata
pip install -U importlib-resources
# this docker image already has CUDA-toolkit installed, if not use docker, you need to configure your GPU environment.
# !!!open pyproject.toml and delete the dependence on casadi.
CUDACXX=/usr/local/cuda/bin/nvcc pip install . --no-build-isolation
```

```sh
cd ~/workspace/src/gcdf-solver/scripts
# run the script to generate the function handle with the trained gcdf model for our simulation experiments
python3 nn_gcdf_l4casadi.py
# Now you will have two generated folders ./_l4c_generated and ../core, in ../core, there are two files: gcdf_func.c, gcdf_func.h

# Then go to the source code of the solver
cd ~/workspace/src/gcdf-solver/src
# modify the top-level Cmakelist to fit your paths for the two generated folders.
# build the project, generate the dynamic library of the implicit casadi functions and install the solver.
# first time would be relatively slow for generating the cdf_func.so
# namespace for the solver is called CRISP since we use the source code in https://computationalrobotics.seas.harvard.edu/CRISP/ as the backbone for our large scale neural implicit GCDF constraints.
```sh
mkdir build
cd build
cmake ..
make
sudo make install
```
```sh
# test if the implicit function can be loaded by casadi
export LD_LIBRARY_PATH=/home/robot/.local/lib/python3.8/site-packages/l4casadi/lib/:/home/robot/.local/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH

python3 ~/workspace/src/gcdf-solver/scripts/test.py
```


## 4. Build the planner
 package for solving and visualization of our trajectory optimization algorithm with the implicit GCDF function.
```sh
cd workspace/src/gcdf-planner
catkin init
catkin build
```
You are all set.


## 5. Usage
- Train the neural network for GCDF representation.
- Generate the dynamic library of the GCDF function handle.
- Build the planner.
- Define your optimizaiton problem in `GCDF\src\gcdf-planner\src\planner\src\GenerateProblemMobileManipulator.cpp` and run this node to generate cppad style dynamic library function handles for your optimization problems. This follows [`CRISP`](https://computationalrobotics.seas.harvard.edu/CRISP/) style.
- Launch the planner.
  ```sh
  cd ~/workspace/src/gcdf_planner
  source devel/setup.bash
  roslaunch planner planner.launch
  ```
    Then give a 2D navigation goal in rviz, the planner will read the goal and pcd files in `GCDF\src\gcdf-planner\src\planner\env` and send them to the solver node. The workflow provided is simplified and is straightforward for you to customize your own robot and environments and planning goals. 

    Add GCDF constraints and solve the optimizaition problem in the python solver node `GCDF\src\gcdf-planner\src\planner\script\SolveMobileManipulatorExp.py`

- We also provide a util function for you to batchly generate your own gazebo random world in ``GCDF\src\gcdf-planner\src\planner\script\map_gen_four_quad.py``. Then you can obtain the pcd file with [this](https://github.com/arshadlab/gazebo_map_creator) repo. 


