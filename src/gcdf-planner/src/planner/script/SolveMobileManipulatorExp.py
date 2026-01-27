#!/usr/bin/env python
import sys
import numpy as np
from sympy import atan2
import casadi as cs
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from mm_msg.msg import SolverInput, State, SolverInputList
from threading import Lock
sys.path.append("/home/robot/workspace/src/gcdf-solver/lib")
casadi_lib_path = "/home/robot/workspace/src/gcdf-solver/lib/libcdf_func.so"
import pyCRISP
import open3d as o3d

# Global variables
cloudMutex = Lock()
GlobalMap = PointCloud2()

# ROS node initialization
rospy.init_node('gcdf_solver_node')

# Publisher for solver output
solver_output_pub = rospy.Publisher("solver_output_list", SolverInputList, queue_size=10)

# CasADi configuration
gcdf_value_function = cs.external("y", casadi_lib_path)
gcdf_gradient_function = cs.external("df_test_truncated", casadi_lib_path)

# CasADi function warmup
warmup_x = cs.DM.zeros(*cdf_value_function.sparsity_in(0).shape)
for _ in range(2):
    cdf_value_function(warmup_x)
    cdf_gradient_function(warmup_x)

# Optimization problem parameters
N = 40
num_state = 9
num_control = 9
variableNum = N * (num_state + num_control)
problemName = "MobileManipulator"
folderName = "model"

cdf_params = None

# Create optimization problem
problem = pyCRISP.OptimizationProblem(variableNum, problemName)

# load the generated objective function and constraints
objective = pyCRISP.ObjectiveFunction(
    variableNum, (num_state + num_control)*N, problemName, folderName, "mobileManipulatorObjective"
)
dynamics = pyCRISP.ConstraintFunction(
    variableNum, problemName, folderName, "mobileManipulatorDynamicConstraints"
)
initial = pyCRISP.ConstraintFunction(
    variableNum, num_state + num_control, problemName, folderName, "mobileManipulatorInitialConstraints"
)
jointLimitsConstraints = pyCRISP.ConstraintFunction(
    variableNum, 12, problemName, folderName, "mobileManipulatorJointLimits"
)
velocityLimitsConstraints = pyCRISP.ConstraintFunction(
    variableNum, problemName, folderName, "mobileManipulatorVelocityLimits"
)

# !add gcdf constraints using the cutomized interface, cdf params is the designed structure to maintain the obstacle list and mapping and the gcdf constraints that you can updated online
gcdf_constraint = pyCRISP.ConstraintFunction(
    variableNum, gcdf_value_function.serialize(), gcdf_gradient_function.serialize(), cdf_params, "mobileManipulatorCDFConstraints"
)

# Add objective function and constraints to the problem
problem.add_objective(objective)
problem.add_equality_constraint(dynamics)
problem.add_equality_constraint(initial)
problem.add_inequality_constraint(jointLimitsConstraints)
problem.add_inequality_constraint(gcdf_constraint)
problem.add_inequality_constraint(velocityLimitsConstraints)

# Solver interface setup
params = pyCRISP.SolverParameters()
solver = pyCRISP.SolverInterface(problem, params)

# Initial values setup
xf = np.zeros(num_state + num_control)
x0 = np.zeros(num_state + num_control)
# joint limit for kinova 6DOF arm
jointLimits = np.array([
    -3.2, 3.2, -2.26, 2.26, -2.56, 2.56,
    -3.2, 3.2, -2.1, 2.1, -3.2, 3.2
]) 

# Initialize the solver
xInitialGuess = np.zeros(variableNum)
solver.set_problem_parameters("mobileManipulatorInitialConstraints", x0)
solver.set_problem_parameters("mobileManipulatorObjective", xInitialGuess)
solver.set_problem_parameters("mobileManipulatorJointLimits", jointLimits)
# solver.set_hyper_parameters("verbose", np.ones(1))
solver.set_hyper_parameters("constraintTol", np.array([1e-1]))
solver.set_hyper_parameters("trustRegionTol", np.array([1e-2]))
solver.initialize(xInitialGuess)

# ROS callback functions
def rosCloudCallback(msg):
    """Callback function to process point cloud data."""
    global GlobalMap
    with cloudMutex:
        GlobalMap = msg
    rospy.loginfo(f"Updated global map with {int(msg.width) * int(msg.height)} points")


# Solver command callback
def solve_command_callback(msg):
    """Callback function to handle solver commands."""
    rospy.loginfo("Solve command received")
    
    solver_input_list = msg.solver_input_list
    if len(solver_input_list) == 0:
        rospy.logwarn("Empty solver input list received")
        return
    
    M = len(solver_input_list)
    rospy.loginfo(f"Processing {M} solve requests")
    
    # Obtain initial and target states and initial guess
    all_x0 = np.zeros((M, num_state + num_control))
    all_xf = np.zeros((M, num_state + num_control))
    all_initial_guesses = np.zeros((M, variableNum))
    
    for m, solver_input in enumerate(solver_input_list):
        all_x0[m, :num_state] = np.array(solver_input.x0.data)
        all_xf[m, :num_state] = np.array(solver_input.xf.data)
        
        # Use initial guess from input if available, else linear interpolation
        for i in range(N):
            idx = i * (num_state + num_control)
            if solver_input.stateTrajectory and len(solver_input.stateTrajectory) > i:
                all_initial_guesses[m, idx:idx + num_state] = np.array(solver_input.stateTrajectory[i].data)
            else:
                alpha = float(i) / (N - 1)
                all_initial_guesses[m, idx:idx + num_state] = (1 - alpha) * all_x0[m, :num_state] + alpha * all_xf[m, :num_state]
    
    # Get point cloud for obstacle information
    with cloudMutex:
        if GlobalMap is None or (int(GlobalMap.width) * int(GlobalMap.height) == 0):
            rospy.logwarn("Global map is empty, skipping this solve command")
            return
        pts = []
        try:
            for p in point_cloud2.read_points(GlobalMap, field_names=("x", "y", "z"), skip_nans=True):
                pts.append([p[0], p[1], p[2]])
        except Exception as e:
            rospy.logerr(f"Failed to read PointCloud2: {e}")
            return
        
        if len(pts) == 0:
            rospy.logwarn("No valid points in global map")
            return
        
        points = np.asarray(pts)
        # Filter points
        filtered_points = points[(points[:, 2] > 0) & (points[:, 0] > -6.5) & 
                                 (points[:, 0] < 6.5) & (points[:, 1] > -6.5) & (points[:, 1] < 6.5)]
        o3d_points = o3d.geometry.PointCloud()
        o3d_points.points = o3d.utility.Vector3dVector(filtered_points)
    
    # Downsample point cloud for optimization
    DownSampled_points = o3d_points.voxel_down_sample(voxel_size=0.25)
    
    # Solve for each input without reconstructing the problem.
    solver_output_list_msg = SolverInputList()
    solver_output_list_msg.solver_input_list = []
    
    for m in range(M):
        rospy.loginfo(f"Solving for input {m+1}/{M}")
        x0 = all_x0[m, :]
        xf = all_xf[m, :]
        xInitialGuess = all_initial_guesses[m, :]
        
        # Set problem parameters
        solver.set_problem_parameters("mobileManipulatorInitialConstraints", x0)
        
        # Update obstacle list and mapping
        obs_lists = []
        lower_bound = np.array([-1.0, -1.0, 0.0])
        upper_bound = np.array([1.0, 1.0, 2.0])
        max_batch_size = 8000 # Max number of obstacles processed per step, should be same with your generated function handle.
        num_configuration_variables_per_step = num_state
        
        for point in DownSampled_points.points:
            obs = np.array([point[0], point[1], point[2]])
            obs_lists.append(obs)
        
        # if you are use a local bounding box for each step, it would be better to solve iteratively (twice is enough) to refine the obstacle mapping.
        # if your map is not that large, you can remove the bounding box and consider all obstacles for all steps directly in one step.
        for _ in range(2):
            obs_index_mapping = []
            for i in range(N - 1):
                center = np.array([xInitialGuess[(i + 1) * (num_state + num_control)],
                                  xInitialGuess[(i + 1) * (num_state + num_control) + 1],
                                  0.0])
                min_bound = center + lower_bound
                max_bound = center + upper_bound
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                indices = bbox.get_point_indices_within_bounding_box(DownSampled_points.points)
                obs_index_mapping.append(indices)
            
            # Update CDF parameters and solve
            cdf_params = pyCRISP.CDFParameters(obs_lists, obs_index_mapping, max_batch_size, num_configuration_variables_per_step)
            solver.set_problem_parameters("mobileManipulatorCDFConstraints", cdf_params)
            solver.set_problem_parameters("mobileManipulatorObjective", xInitialGuess)
            solver.reset_problem(xInitialGuess)
            solver.solve()
            
            xInitialGuess = solver.get_solution()
            solution = solver.get_solution()
        
        # Check success and prepare output
        is_success = solver.isSuccessful()
        if is_success:
            rospy.loginfo(f"Solver successful for input {m+1}")
            solver_output = SolverInput()
            for i in range(N):
                idx = i * (num_state + num_control)
                state_now = State()
                state_now.data = solution[idx:idx + num_state]
                solver_output.stateTrajectory.append(state_now)
            solver_output_list_msg.solver_input_list.append(solver_output)
        else:
            rospy.logwarn(f"Solver failed for input {m+1}")
    
    # Publish results
    solver_output_pub.publish(solver_output_list_msg)
    rospy.loginfo(f"Published {len(solver_output_list_msg.solver_input_list)} solutions")

# Subscribe to input commands and point cloud
state_sub = rospy.Subscriber("solver_input_list", SolverInputList, solve_command_callback, queue_size=10)
cloud_sub = rospy.Subscriber("global_cloud", PointCloud2, rosCloudCallback)

# ROS spin loop
rospy.spin()