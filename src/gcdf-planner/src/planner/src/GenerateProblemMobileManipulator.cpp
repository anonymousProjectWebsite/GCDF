// Define the optimization problem for a mobile manipulator using CRISP style interface
// leave GCDF constraints constructed online using python interface.
// This is an example code showing how to define a mobile manipulator problem using the CRISP solver interface.
// In this example, 40 timesteps are used, with a time step of 0.1s.
#include <filesystem>
#include "solver_core/SolverInterface.h"
#include <chrono>
#include "math.h"

namespace fs = std::filesystem;
using namespace CRISP;

// velocity control for mobile manipulator: a mobile base with a 6 DOF manipulator.
// no cdf collision constraints for testing the pipeline.
// all states: [qx, qy, theta_z, q0, q1, q2, q3, q4, q5, vx,vy, omega, v0,v1,v2,v3,v4,v5] 
const size_t N  = 40;
const size_t num_state = 9;
const size_t num_control = 9;
const scalar_t dt = 0.1;

ad_function_t mobileManipulatorDynamicConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
    y.resize((N - 1) * num_state);
    for (size_t i = 0; i < N - 1; ++i) {
        size_t idx = i * (num_state + num_control);
        // Extract state and control for current and next time steps
        ad_scalar_t qx_i = x[idx + 0];
        ad_scalar_t qy_i = x[idx + 1];
        ad_scalar_t theta_z_i = x[idx + 2];
        ad_scalar_t q0_i = x[idx + 3];
        ad_scalar_t q1_i = x[idx + 4];
        ad_scalar_t q2_i = x[idx + 5];
        ad_scalar_t q3_i = x[idx + 6];
        ad_scalar_t q4_i = x[idx + 7];
        ad_scalar_t q5_i = x[idx + 8];
        ad_scalar_t vx_i = x[idx + 9];
        ad_scalar_t vy_i = x[idx + 10];
        ad_scalar_t omega_i = x[idx + 11];
        ad_scalar_t v0_i = x[idx + 12];
        ad_scalar_t v1_i = x[idx + 13];
        ad_scalar_t v2_i = x[idx + 14];
        ad_scalar_t v3_i = x[idx + 15];
        ad_scalar_t v4_i = x[idx + 16];
        ad_scalar_t v5_i = x[idx + 17];

        ad_scalar_t qx_next = x[idx + (num_state + num_control) + 0];
        ad_scalar_t qy_next = x[idx + (num_state + num_control) + 1];
        ad_scalar_t theta_z_next = x[idx + (num_state + num_control) + 2];
        ad_scalar_t q0_next = x[idx + (num_state + num_control) + 3];
        ad_scalar_t q1_next = x[idx + (num_state + num_control) + 4];
        ad_scalar_t q2_next = x[idx + (num_state + num_control) + 5];
        ad_scalar_t q3_next = x[idx + (num_state + num_control) + 6];
        ad_scalar_t q4_next = x[idx + (num_state + num_control) + 7];
        ad_scalar_t q5_next = x[idx + (num_state + num_control) + 8];

        // explicit state update
        y.segment(i * num_state, num_state) << qx_next - qx_i - (vx_i * cos(theta_z_i) - vy_i * sin(theta_z_i)) * dt,
                                                qy_next - qy_i - (vx_i * sin(theta_z_i) + vy_i * cos(theta_z_i)) * dt,
                                                theta_z_next - theta_z_i - omega_i * dt,
                                                q0_next - q0_i - v0_i * dt,
                                                q1_next - q1_i - v1_i * dt,
                                                q2_next - q2_i - v2_i * dt,
                                                q3_next - q3_i - v3_i * dt,
                                                q4_next - q4_i - v4_i * dt,
                                                q5_next - q5_i - v5_i * dt;
    }
    std::cout << "Dynamics constraints constructed." << std::endl;
};

// initial constraints
ad_function_with_param_t mobileManipulatorInitialConstraints = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(num_state);
    y.segment(0, num_state) << x[0] - p[0],
                    x[1] - p[1],
                    x[2] - p[2],
                    x[3] - p[3],
                    x[4] - p[4],
                    x[5] - p[5],
                    x[6] - p[6],
                    x[7] - p[7],
                    x[8] - p[8];
    std::cout << "Initial constraints constructed." << std::endl;
};

ad_function_with_param_t mobileManipulatorJointLimits = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
    size_t total_constraints = (N * 6 * 2);
    y.resize(total_constraints);
    for (size_t i = 0; i < N; ++i) {
        size_t idx = i * (num_state + num_control);
        y[i * 12 + 0] = x[idx + 3] - p[0]; // q0 >= lower limit
        y[i * 12 + 1] = p[1] - x[idx + 3]; // q0 <= upper limit
        y[i * 12 + 2] = x[idx + 4] - p[2]; // q1 >= lower limit
        y[i * 12 + 3] = p[3] - x[idx + 4]; // q1 <= upper limit
        y[i * 12 + 4] = x[idx + 5] - p[4]; // q2 >= lower limit
        y[i * 12 + 5] = p[5] - x[idx + 5]; // q2 <= upper limit
        y[i * 12 + 6] = x[idx + 6] - p[6]; // q3 >= lower limit
        y[i * 12 + 7] = p[7] - x[idx + 6]; // q3 <= upper limit
        y[i * 12 + 8] = x[idx + 7] - p[8]; // q4 >= lower limit
        y[i * 12 + 9] = p[9] - x[idx + 7]; // q4 <= upper limit
        y[i * 12 + 10] = x[idx + 8] - p[10]; // q5 >= lower limit
        y[i * 12 + 11] = p[11] - x[idx + 8]; // q5 <= upper limit
    }
    std::cout << "Joint limits constraints constructed." << std::endl;
};


ad_function_t mobileManipulatorVelocityLimits = [](const ad_vector_t& x, ad_vector_t& y) {
    size_t total_constraints = ((N-1) * num_control * 2);
    y.resize(total_constraints);
    for (size_t i = 0; i < N - 1; ++i) {
        size_t idx = i * (num_state + num_control);
        // Velocity limits for vx, vy, omega, v0, v1, v2, v3, v4, v5
        y[i * 18 + 0] = 2.0 - x[idx + 9];   // vx <= 1.0
        y[i * 18 + 1] = x[idx + 9] + 2.0;  // vx >= -1.0
        y[i * 18 + 2] = 2.0 - x[idx + 10];  // vy <= 1.0
        y[i * 18 + 3] = x[idx + 10] + 2.0; // vy >= -1.0
        y[i * 18 + 4] = M_PI - x[idx + 11];  // omega <= pi/9
        y[i * 18 + 5] = x[idx + 11] + M_PI; // omega >= -pi/9
        y[i * 18 + 6] = M_PI - x[idx + 12];   // v0 <= pi/6
        y[i * 18 + 7] = x[idx + 12] + M_PI;  // v0 >= -pi/6
        y[i * 18 + 8] = M_PI - x[idx + 13];  // v1 <= pi/6
        y[i * 18 + 9] = x[idx + 13] + M_PI; // v1 >= -pi/6
        y[i * 18 + 10] = M_PI - x[idx + 14];   // v2 <= pi/6
        y[i * 18 + 11] = x[idx + 14] + M_PI;  // v2 >= -pi/6
        y[i * 18 + 12] = M_PI - x[idx + 15];  // v3 <= pi/6
        y[i * 18 + 13] = x[idx + 15] + M_PI; // v3 >= -pi/6
        y[i * 18 + 14] = M_PI - x[idx + 16];   // v4 <= pi/6
        y[i * 18 + 15] = x[idx + 16] + M_PI;  // v4 >= -pi/6
        y[i * 18 + 16] = M_PI - x[idx + 17];  // v5 <= pi/6
        y[i * 18 + 17] = x[idx + 17] + M_PI; // v5 >= -pi/6
    }
    std::cout << "Velocity limits constraints constructed." << std::endl;
};

// cost function for mobile manipulator
ad_function_with_param_t mobileManipulatorObjective = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(1);
    y[0] = 0.0;
    ad_scalar_t tracking_cost(0.0);
    ad_scalar_t control_cost(0.0);
    ad_matrix_t Q_intermediate(num_state + 1 + num_control, num_state + 1 + num_control);
    ad_matrix_t Q_final(num_state + 1 + num_control, num_state + 1 + num_control);
    Q_intermediate.setZero();
    Q_intermediate(0, 0) = 0.1;
    // Q_intermediate(1, 1) = 0;
    // Q_intermediate(2, 2) = 0.01;
    // Q_intermediate(3, 3) = 0.01;
    // Q_intermediate(4, 4) = 0.01;
    // Q_intermediate(5, 5) = 0.01;
    // Q_intermediate(6, 6) = 0.01;
    // Q_intermediate(7, 7) = 0.01;
    // Q_intermediate(8, 8) = 0.01;
    // Q_intermediate(9, 9) = 0.01;
    Q_final.setZero();
    Q_final(0, 0) = 10;
    Q_final(1, 1) = 10;
    Q_final(2, 2) = 0.1;
    Q_final(3, 3) = 0.1;
    Q_final(4, 4) = 0.1;
    Q_final(5, 5) = 0.1;
    Q_final(6, 6) = 0.1;
    Q_final(7, 7) = 0.1;
    Q_final(8, 8) = 0.1;
    Q_final(9, 9) = 0.1; // to penalize the orientation error properly
    Q_final(10, 10) = 0.1; // to penalize the orientation error properly
    Q_final(11, 11) = 0.1; // to penalize the orientation
    Q_final(12, 12) = 0.1; // to penalize the joint velocities
    Q_final(13, 13) = 0.1; // to penalize the joint
    Q_final(14, 14) = 0.1; // to penalize the joint velocities
    Q_final(15, 15) = 0.1; // to penalize the joint
    Q_final(16, 16) = 0.1; // to penalize the joint
    Q_final(17, 17) = 0.1; // to penalize the joint
    Q_final(18, 18) = 0.1; // to penalize the joint velocities
    ad_matrix_t R(num_control, num_control);
    R.setZero();
    R(0, 0) = 0.01;
    R(1, 1) = 0.01;
    R(2, 2) = 0.001;
    R(3, 3) = 0.001;
    R(4, 4) = 0.001;
    R(5, 5) = 0.001;
    R(6, 6) = 0.001;
    R(7, 7) = 0.001;
    R(8, 8) = 0.001;

    // currently, lets not add intermediate tracking references.
    for (size_t i = 0; i < N; i++){
        size_t idx = i * (num_state + num_control);
        // Extract state and control for current and next time steps
        ad_scalar_t qx_i = x[idx + 0];
        ad_scalar_t qy_i = x[idx + 1];
        ad_scalar_t theta_z_i = x[idx + 2];
        ad_scalar_t q0_i = x[idx + 3];
        ad_scalar_t q1_i = x[idx + 4];
        ad_scalar_t q2_i = x[idx + 5];
        ad_scalar_t q3_i = x[idx + 6];
        ad_scalar_t q4_i = x[idx + 7];
        ad_scalar_t q5_i = x[idx + 8];
        ad_scalar_t vx_i = x[idx + 9];
        ad_scalar_t vy_i = x[idx + 10];
        ad_scalar_t omega_i = x[idx + 11];
        ad_scalar_t v0_i = x[idx + 12];
        ad_scalar_t v1_i = x[idx + 13];
        ad_scalar_t v2_i = x[idx + 14];
        ad_scalar_t v3_i = x[idx + 15];
        ad_scalar_t v4_i = x[idx + 16];
        ad_scalar_t v5_i = x[idx + 17];


        // add intermediate tracking cost
        if (i < N - 1)
        {
            ad_vector_t tracking_error(num_state + 1 + num_control);
            tracking_error << qx_i - p[i* 18 + 0],
                              qy_i - p[i*18 + 1],
                              cos(theta_z_i) - cos(p[i*18 + 2]),
                              sin(theta_z_i) - sin(p[i*18 + 2]),
                              q0_i - p[i*18 + 3],
                              q1_i - p[i*18 + 4],
                              q2_i - p[i*18 + 5],
                              q3_i - p[i*18 + 6],
                              q4_i - p[i*18 + 7],
                              q5_i - p[i*18 + 8],
                              vx_i - p[i*18 + 9],
                              vy_i - p[i*18 + 10],
                              omega_i - p[i*18 + 11],
                              v0_i - p[i*18 + 12],
                              v1_i - p[i*18 + 13],
                              v2_i - p[i*18 + 14],
                              v3_i - p[i*18 + 15],
                              v4_i - p[i*18 + 16],
                              v5_i - p[i*18 + 17];
            tracking_cost += tracking_error.transpose() * Q_intermediate * tracking_error;
        }

        if (i == N - 1)
        {
            ad_vector_t tracking_error(num_state + 1 + num_control);
            tracking_error << qx_i - p[i* 18 + 0],
                              qy_i - p[i*18 + 1],
                              cos(theta_z_i) - cos(p[i*18 + 2]),
                              sin(theta_z_i) - sin(p[i*18 + 2]),
                              q0_i - p[i*18 + 3],
                              q1_i - p[i*18 + 4],
                              q2_i - p[i*18 + 5],
                              q3_i - p[i*18 + 6],
                              q4_i - p[i*18 + 7],
                              q5_i - p[i*18 + 8],
                              vx_i - p[i*18 + 9],
                              vy_i - p[i*18 + 10],
                              omega_i - p[i*18 + 11],
                              v0_i - p[i*18 + 12],
                              v1_i - p[i*18 + 13],
                              v2_i - p[i*18 + 14],
                              v3_i - p[i*18 + 15],
                              v4_i - p[i*18 + 16],
                              v5_i - p[i*18 + 17];
                            
            tracking_cost += tracking_error.transpose() * Q_final * tracking_error;
        }

        if (i < N -1)
        {
            ad_vector_t control_error(num_control);
            control_error << x[idx + 9], // vx
                             x[idx + 10], // vy
                             x[idx + 11], // omega
                             x[idx + 12], // v0
                             x[idx + 13], // v1
                             x[idx + 14], // v2
                             x[idx + 15], // v3
                             x[idx + 16], // v4
                             x[idx + 17]; // v5

            control_cost += control_error.transpose() * R * control_error;
        }
    }
    y[0] = tracking_cost + control_cost;
    std::cout << "Objective constructed." << std::endl;
};

int main(){
    size_t variableNum = N * (num_state + num_control);
    std::string problemName = "MobileManipulator";
    std::string folderName = "model";
    OptimizationProblem mobileManipulatorProblem(variableNum, problemName);

    auto obj = std::make_shared<ObjectiveFunction>(variableNum, (num_state + num_control)*N, problemName, folderName, "mobileManipulatorObjective", mobileManipulatorObjective);
    auto dynamics = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "mobileManipulatorDynamicConstraints", mobileManipulatorDynamicConstraints);
    auto initial = std::make_shared<ConstraintFunction>(variableNum, num_state + num_control, problemName, folderName, "mobileManipulatorInitialConstraints", mobileManipulatorInitialConstraints);
    auto jointLimitsConstraints = std::make_shared<ConstraintFunction>(variableNum, 12, problemName, folderName, "mobileManipulatorJointLimits", mobileManipulatorJointLimits);
    auto velocityLimitsConstraints = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "mobileManipulatorVelocityLimits", mobileManipulatorVelocityLimits);
}


                    
                
