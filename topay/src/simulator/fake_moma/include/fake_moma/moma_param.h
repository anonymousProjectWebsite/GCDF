#pragma once

#include <Eigen/Dense>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <ros/ros.h>
#include <vector>
#include <string>
#include <iostream>
#include <unordered_map>
#include <array>
#include <cmath>

#define PRINTF_WHITE(STRING) std::cout<<STRING
#define PRINT_GREEN(STRING) std::cout<<"\033[92m"<<STRING<<"\033[m\n"
#define PRINT_RED(STRING) std::cout<<"\033[31m"<<STRING<<"\033[m\n"
#define PRINTF_RED(STRING) std::cout<<"\033[31m"<<STRING<<"\033[m"
#define PRINT_YELLOW(STRING) std::cout<<"\033[33m"<<STRING<<"\033[m\n"
#define PRINTF_YELLOW(STRING) std::cout<<"\033[33m"<<STRING<<"\033[m"

// Same as the original NodeHandle::getParam, but throws exception if the parameter is not found
#define GET_PARAM_OR_THROW(nh, param_name, param_var) \
    do { \
        if (!(nh).getParam((param_name), (param_var))) { \
            std::string error_msg = "Failed to get required parameter: " + std::string(param_name); \
            throw std::runtime_error(error_msg); \
        } \
    } while (0)

using namespace Eigen;
using namespace std;
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/**
 * MomaParam for Kinova Gen3 (6-DoF) mobile manipulator
 * ----------------------------------------------------
 *
 *  - Base: a fake differential-drive chassis
 *  - Arm:  Kinova Gen3 6-DoF
 *
 * Kinematics model:
 *  - We follow the URDF convention: each revolute joint i is represented as
 *      T_parent_child(q_i) = T_origin_i * Rz(q_i)
 *    where T_origin_i is given by the URDF origin (xyz, rpy) from parent link
 *    to child link when q_i = 0.
 *  - All joints rotate about the local Z axis.
 *
 * Collision model:
 *  - Chassis: cylinder around base_link in the x-y plane.
 *  - Arm: for each link (including the tool segment) we place up to two
 *    collision spheres along the local +Z axis of that link. The offsets along
 *    the link (colli_points) are derived from link lengths in the Kinova
 *    documentation. Radii are set to `cylinder_radius`.
 */

struct MomaParam
{
    // chassis parameters
    double chassis_length = 0.525;   // from URDF base_link collision
    double chassis_width  = 0.4;
    double chassis_height = 0.2745;  // approximate
    double chassis_colli_radius = 0.45; // conservative radius in x-y plane
    double max_v  = 1.2;
    double max_a  = 0.8;
    double max_w  = 1.25; // angular vel
    double max_dw = 1.0;  // angular accel

    // arm parameters (Kinova Gen3 6-DoF)
    static constexpr size_t kDofNum = 6;  // compile-time constant
    size_t dof_num = kDofNum;             // runtime copy, moma_param.dof_num


    double cylinder_radius = 0.055;  // radius used for cylinder collision model of the robot arm
    // Geometric / collision parameters
    VectorXd colli_length;         // [d0 ... d6] 7 segments along z
    VectorXd colli_points;         // 2 points per segment (distance along local z)
    VectorXd colli_point_radius;   // radius for each sphere
    VectorXd default_colli_point_radius;
    VectorXi colli_link_map;       // mapping sphere index -> link index
    VectorXd link_length;          // approximate physical link lengths

    // joint limits etc
    VectorXd joint_pos_limit_min;
    VectorXd joint_pos_limit_max;
    VectorXd joint_vel_limit;
    VectorXd joint_acc_limit;
    VectorXd joint_eff_limit;

    // mesh abs paths
    const std::string chassis_mesh  = "file:///home/topay/src/simulator/moma_ours/swerve_description/meshes/holonomic/base_link.STL";

    
    
    // const std::string gripper_mesh  = "/ABS/PATH/to/gripper.dae";
    // arm links path vector
    const std::vector<std::string> link_meshes = {
        "file:///home/topay/src/simulator/moma_ours/swerve_description/meshes/jaco/base_link.STL",
        "file:///home/topay/src/simulator/moma_ours/swerve_description/meshes/jaco/shoulder_link.STL",
        "file:///home/topay/src/simulator/moma_ours/swerve_description/meshes/jaco/bicep_link.STL",
        "file:///home/topay/src/simulator/moma_ours/swerve_description/meshes/jaco/forearm_link.STL",
        "file:///home/topay/src/simulator/moma_ours/swerve_description/meshes/jaco/spherical_wrist_1_link.STL",
        "file:///home/topay/src/simulator/moma_ours/swerve_description/meshes/jaco/spherical_wrist_2_link.STL",
        "file:///home/topay/src/simulator/moma_ours/swerve_description/meshes/jaco/bracelet_no_vision_link.STL"
    };

    // not heavily used but kept for compatibility
    MatrixX3d joint_offset;        // optional static offsets (unused in FK)
    MatrixX3d joint_dof_axis;      // joint axes in local frames (always z)

    // collision points local axis
    RowMatrixXd colli_local_axis;

    // Self‑collision masking matrix 
    MatrixXi collision_matrix;

    // Transform from chassis base_link frame to arm base (armbase_link)
    Matrix3d relative_R;   // rotation
    Vector3d relative_t;   // translation

    // URDF joint origins: parent link -> child link (q = 0)
    //   T_parent_child = [ R_origin  p_origin; 0 0 0 1 ]
    std::array<Vector3d, kDofNum> joint_origin_pos;  // xyz
    std::array<Matrix3d, kDofNum> joint_origin_rot;  // rotation matrix from rpy

    // ---------------------------------------------------------------------
    // Helper: convert URDF rpy (roll, pitch, yaw) to rotation matrix
    // URDF uses fixed-axis extrinsic rotations X-Y-Z, so
    //   R = Rz(yaw) * Ry(pitch) * Rx(roll)
    // ---------------------------------------------------------------------
    static Matrix3d rpyToRot(double roll, double pitch, double yaw)
    {
        Matrix3d R;
        R = AngleAxisd(yaw,   Vector3d::UnitZ()).toRotationMatrix() *
            AngleAxisd(pitch, Vector3d::UnitY()).toRotationMatrix() *
            AngleAxisd(roll,  Vector3d::UnitX()).toRotationMatrix();
        return R;
    }

    MomaParam()
    {
        // ----------------------
        // Allocate Eigen members
        // ----------------------
        link_length         = VectorXd::Zero(dof_num);
        joint_pos_limit_min = VectorXd::Zero(dof_num);
        joint_pos_limit_max = VectorXd::Zero(dof_num);
        joint_vel_limit     = VectorXd::Zero(dof_num);
        joint_acc_limit     = VectorXd::Zero(dof_num);
        joint_eff_limit     = VectorXd::Zero(dof_num);
        joint_offset        = MatrixX3d::Zero(dof_num, 3);
        joint_dof_axis      = MatrixX3d::Zero(dof_num, 3);

        // All arm joints rotate about local z
        for (size_t i = 0; i < dof_num; ++i)
        {
            joint_dof_axis.row(i) = Vector3d(0.0, 0.0, 1.0);
        }

        // ------------------------------------------------------
        // Relative transform: base_link -> armbase_link (URDF)
        //   <joint name="arm_to_chassis" type="fixed">
        //      origin xyz="0.03385 -0.00764 0.306" rpy="0 0 0"
        // ------------------------------------------------------
        relative_R = Matrix3d::Identity();
        relative_t << 0.2625, 0.0, 0.2745;

        // ------------------------------------------------------
        // Kinova Gen3 approximate link lengths (meters)
        // from user manual (Figure 85, vertical distances):
        //   base->act1:  0.1564
        //   act1->act2:  0.1284
        //   act2->act3:  0.4100
        //   act3->act4:  0.2084
        //   act4->act5:  0.1059
        //   act5->act6:  0.1059
        //   act6->tool:  0.0615
        // link_length is only used for visualization / debugging.
        // ------------------------------------------------------
        link_length << 0.1564, 0.1284, 0.4100, 0.2084, 0.1059, 0.1059;

        // Segment lengths used to place collision spheres
        colli_length = VectorXd::Zero(dof_num + 1);
        colli_length << 0.1564, 0.1284, 0.4100, 0.2084, 0.1059, 0.1059, 0.0615;

        // Two collision spheres per segment (distance along local +Z)
        colli_points       = VectorXd::Zero((dof_num + 1) * 2);
        colli_point_radius = VectorXd::Zero((dof_num + 1) * 2);

        for (size_t i = 0; i < dof_num + 1; ++i)
        {
            double L = colli_length(static_cast<int>(i));
            // set one collision ball at center of that link
            colli_points(2 * i)     = 0.5 * L;
            colli_points(2 * i + 1) = L;

            // colli_point_radius(2 * i)     = 0.5 * L;
            // colli_point_radius(2 * i + 1) = 0.065;
        }
        // link hand-craft 
        colli_points(4) = 0.25 * 0.4100;colli_points(5) = 0.75 * 0.4100;
        colli_point_radius << colli_length(0)/2,0.065,
                              colli_length(1)/2,0.065,
                              colli_length(2)/4+0.03,colli_length(2)/4+0.03,
                              colli_length(3)/2,0.065,
                              colli_length(4)/2,0.065,
                              colli_length(5)/2,0.065,
                              colli_length(6)/2,0.065;
        // colli_point_radius.setConstant(0.05);
        default_colli_point_radius = colli_point_radius;

        // Joint limits
        joint_pos_limit_min << -3.2, -2.26, -2.56, -3.2, -2.1, -3.2;
        joint_pos_limit_max <<  3.2,  2.26,  2.56,  3.2,  2.1,  3.2;

        // Velocity / acceleration / effort limits (rough, can be updated)
        joint_vel_limit.setConstant(2.0);    // rad/s
        // joint_acc_limit.setConstant(6.28);   // rad/s^2
        joint_acc_limit << 5.2,5.2,5.2,10.0,10.0,10.0;
        joint_eff_limit << 40.0, 40.0, 40.0, 9.0, 9.0, 9.0;

        // ------------------------------------------------------
        // URDF joint origins (parent -> child when q = 0)
        //   From the provided moma.xacro-expanded URDF
        // ------------------------------------------------------
        // armjoint_1: armbase_link -> armshoulder_link
        joint_origin_pos[0] = Vector3d(0.0, 0.0, 0.15643);
        joint_origin_rot[0] = rpyToRot(-M_PI, 0.0, 0.0);  // rpy="-3.1416 0 0"

        // armjoint_2: armshoulder_link -> armbicep_link
        joint_origin_pos[1] = Vector3d(0.0, 0.005375, -0.12838);
        joint_origin_rot[1] = rpyToRot(M_PI/2, 0.0, 0.0); // rpy="1.5708 0 0"

        // armjoint_3: armbicep_link -> armforearm_link
        joint_origin_pos[2] = Vector3d(0.0, -0.41, 0.0);
        joint_origin_rot[2] = rpyToRot(M_PI, 0.0, 0.0);    // rpy="3.1416 0 0"

        // armjoint_4: armforearm_link -> armspherical_wrist_1_link
        joint_origin_pos[3] = Vector3d(0.0, 0.20843, -0.006375);
        joint_origin_rot[3] = rpyToRot(M_PI/2, 0.0, 0.0); // rpy="1.5708 0 0"

        // armjoint_5: armspherical_wrist_1_link -> armspherical_wrist_2_link
        joint_origin_pos[4] = Vector3d(0.0, -0.00017505, -0.10593);
        joint_origin_rot[4] = rpyToRot(-M_PI/2, 0.0, 0.0); // rpy="-1.5708 0 0"

        // armjoint_6: armspherical_wrist_2_link -> armbracelet_link
        joint_origin_pos[5] = Vector3d(0.0, 0.10593, -0.00017505);
        joint_origin_rot[5] = rpyToRot(M_PI/2, 0.0, 0.0);  // rpy="1.5708 0 0"


        // collision point axis in each local frame for kinova 6dof robot
        colli_local_axis = RowMatrixXd::Zero((dof_num + 1)*2, 3);
        // from base link to joint 1
        colli_local_axis.row(0) << 0.0,0.0,1.0;
        colli_local_axis.row(1) << 0.0,0.0,1.0;
        // from joint 1 to joint 2
        colli_local_axis.row(2) << 0.0,0.0,-1.0;
        colli_local_axis.row(3) << 0.0,0.0,-1.0;
        // from joint 2 to joint 3
        colli_local_axis.row(4) << 0.0,-1.0,0.0;
        colli_local_axis.row(5) << 0.0,-1.0,0.0;
        // from joint 3 to joint 4
        colli_local_axis.row(6) << 0.0,1.0,0.0;
        colli_local_axis.row(7) << 0.0,1.0,0.0;
        // from joint 4 to joint 5
        colli_local_axis.row(8) << 0.0,0.0,-1.0;
        colli_local_axis.row(9) << 0.0,0.0,-1.0;
        // from joint 5 to joint 6
        colli_local_axis.row(10) << 0.0,1.0,0.0;
        colli_local_axis.row(11) << 0.0,1.0,0.0;
        // from joint 6 to interface module frame
        colli_local_axis.row(12) << 0.0,0.0,-1.0;
        colli_local_axis.row(13) << 0.0,0.0,-1.0;

        // ------------------------------------------------------
        // Build colli_link_map and collision_matrix using the current
        // collision spheres at zero configuration.
        // ------------------------------------------------------
        VectorXd zero_conf = VectorXd::Zero(3 + dof_num);
        std::vector<Vector4d> cpts = getColliPts(zero_conf);

        colli_link_map.resize(static_cast<int>(cpts.size()));
        int idx = 0;
        for (size_t i = 0; i < dof_num + 1; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                double d = colli_points(2 * i + j);
                double r = colli_point_radius(2 * i + j);
                if (d <= 0.0 || r <= 0.0)
                    continue;
                if (idx < colli_link_map.size())
                    colli_link_map(idx) = static_cast<int>(i);
                ++idx;
            }
        }

        collision_matrix.resize(static_cast<int>(cpts.size()), static_cast<int>(cpts.size()));
        collision_matrix.setConstant(-1);
        for (size_t i = 0; i < cpts.size(); ++i)
        {
            for (size_t j = i; j < cpts.size(); ++j)
            {
                if (i == j)
                {
                    collision_matrix(static_cast<int>(i), static_cast<int>(j)) = 1;
                    continue;
                }
                double dist = (cpts[i].head<3>() - cpts[j].head<3>()).norm();
                if (dist <= cpts[i][3] + cpts[j][3] + 1e-1)
                {
                    collision_matrix(static_cast<int>(i), static_cast<int>(j)) = 1;
                    collision_matrix(static_cast<int>(j), static_cast<int>(i)) = 1;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Collision radius utilities
    // ------------------------------------------------------------------
    // void setColliRs(const Eigen::VectorXd &colli_rs)
    // {
    //     // colli_rs: 2*(dof_num+1) sphere radii + 1 chassis radius (length 2*(dof+1)+1)
    //     colli_point_radius.setZero();
    //     int expected = static_cast<int>((dof_num + 1) * 2 + 1);
    //     if (colli_rs.size() != expected)
    //     {
    //         PRINT_YELLOW("[MomaParam] setColliRs: unexpected size, expected " << expected << ", got " << colli_rs.size());
    //     }

    //     int n_spheres = std::min<int>(colli_point_radius.size(), colli_rs.size());
    //     for (int i = 0; i < n_spheres; ++i)
    //     {
    //         colli_point_radius(i) = colli_rs(i);
    //     }

    //     if (colli_rs.size() > colli_point_radius.size())
    //     {
    //         chassis_colli_radius = colli_rs(colli_point_radius.size());
    //     }
    // }

    void resetColliRs()
    {
        colli_point_radius   = default_colli_point_radius;
        chassis_colli_radius = 0.45;
    }

    // ------------------------------------------------------------------
    // Forward kinematics helper used by both getColliPts and gradient
    // computations. It computes:
    //   - joint world positions and axes (for Jacobians)
    //   - collision sphere world positions
    // ------------------------------------------------------------------
    void computeKinematics(const Eigen::VectorXd &moma_pos,
                           std::vector<Vector3d> &joint_pos,
                           std::vector<Vector3d> &joint_axis,
                           std::vector<Vector4d> &sphere_pos_radius,
                           std::vector<Vector3d> *p_all_link = nullptr,
                           std::vector<Matrix3d> *R_all_link = nullptr) const
    {
        joint_pos.clear();
        joint_axis.clear();
        sphere_pos_radius.clear();
        // Base pose
        double bx   = moma_pos[0];
        double by   = moma_pos[1];
        double yaw  = moma_pos[2];

        if (p_all_link) p_all_link->clear();
        if (R_all_link) R_all_link->clear();

        Vector3d p_base(bx, by, 0.0);
        Matrix3d R_base;
        R_base << cos(yaw), -sin(yaw), 0.0,
                  sin(yaw),  cos(yaw), 0.0,
                  0.0,       0.0,      1.0;
        // Start at arm base (armbase_link) frame
        Vector3d p = p_base + R_base * relative_t;
        Matrix3d R = R_base * relative_R;
        if (p_all_link) p_all_link->push_back(p_base);
        if (R_all_link) R_all_link->push_back(R_base);
        // For base yaw axis
        // (used when computing gradients w.r.t. moma_pos[2])
        // origin at chassis top center
        // Vector3d o_base = p_base; -- computed where needed

        // Iterate over segments/links
        for (size_t i = 0; i < dof_num + 1; ++i)
        {   
            if (p_all_link) p_all_link->push_back(p);
            if (R_all_link) R_all_link->push_back(R);
            // Place collision spheres for the current link i
            for (int j = 0; j < 2; ++j)
            {
                int index = i*2 + j;
                double d = colli_points(index);
                double r = colli_point_radius(index);
                if (d <= 0.0 || r <= 1e-6)
                    continue;
                Vector3d sp = p + R * colli_local_axis.row(index).transpose() * d;
                sphere_pos_radius.push_back(Vector4d(sp(0), sp(1), sp(2), r));
            }

            if (i == dof_num)
                break;  // last segment has no further joint

            // Transform to joint i frame (parent -> child) at q=0
            p = p + R * joint_origin_pos[i];
            R = R * joint_origin_rot[i];

            // Joint i world origin and axis (about local Z)
            joint_pos.push_back(p);
            joint_axis.push_back(R.col(2));

            // Apply rotation about local Z by q_i
            double qi = moma_pos[3 + static_cast<int>(i)];
            Matrix3d Rz;
            Rz << cos(qi), -sin(qi), 0.0,
                  sin(qi),  cos(qi), 0.0,
                  0.0,      0.0,     1.0;
            R = R * Rz;
            // p stays the same (rotation about joint origin)
        }
    }

    // ------------------------------------------------------------------
    // Collision points (world coordinates + radius)
    // ------------------------------------------------------------------
    std::vector<Vector4d> getColliPts(const Eigen::VectorXd &moma_pos) const
    {
        std::vector<Vector3d> joint_pos, joint_axis;
        std::vector<Vector4d> sphere_pos_radius;
        computeKinematics(moma_pos, joint_pos, joint_axis, sphere_pos_radius);
        return sphere_pos_radius;
    }

    // ------------------------------------------------------------------
    // Self-collision detection using sphere model
    // ------------------------------------------------------------------
    bool isSelfCollision(const Eigen::VectorXd &moma_pos, Eigen::VectorXi &collision_link)
    {
        collision_link.resize(1 + 1 + dof_num);
        collision_link.setZero();

        bool is_collision = false;
        std::vector<Vector4d> cpts = getColliPts(moma_pos);

        // Chassis vs. arm spheres (simple z + radius check, same as original)
        for (size_t i = 0; i < cpts.size(); ++i)
        {
            if (i != 0 && cpts[i](2) < chassis_height + cpts[i](3))
            {
                collision_link(colli_link_map(static_cast<int>(i)) + 1) = 1;
                collision_link(0) = 1;
                is_collision = true;
            }
        }

        // Arm self-collision between collision spheres
        for (size_t i = 0; i < cpts.size(); ++i)
        {
            for (size_t j = i + 1; j < cpts.size(); ++j)
            {
                if (collision_matrix(static_cast<int>(i), static_cast<int>(j)) == 1)
                    continue; // always intersecting, ignore

                double dist = (cpts[i].head<3>() - cpts[j].head<3>()).norm();
                if (dist < cpts[i][3] + cpts[j][3] - 1e-2)
                {
                    collision_link(colli_link_map(static_cast<int>(i)) + 1) = 1;
                    collision_link(colli_link_map(static_cast<int>(j)) + 1) = 1;
                    is_collision = true;
                }
            }
        }

        return is_collision;
    }

    // ------------------------------------------------------------------
    // Gradients of a scalar cost w.r.t. moma_pos given gradients w.r.t.
    // each collision sphere position.
    // simple deduction by using chain rule and geometric jacobian
    Eigen::VectorXd getColliGrads(const Eigen::VectorXd &moma_pos,
                                const std::vector<Eigen::Vector3d> &pos_grads)
    {
        int n_dofs = static_cast<int>(dof_num);
        Eigen::VectorXd grads = Eigen::VectorXd::Zero(3 + n_dofs);

        std::vector<Vector3d> joint_pos, joint_axis;
        std::vector<Vector4d> sphere_pos_radius;
        computeKinematics(moma_pos, joint_pos, joint_axis, sphere_pos_radius);

        if (sphere_pos_radius.size() != pos_grads.size())
        {
            PRINT_YELLOW("[MomaParam] getColliGrads: sphere_pos_radius.size() != pos_grads.size(), using min size");
        }
        size_t N = std::min(sphere_pos_radius.size(), pos_grads.size());


        Vector3d o_base(moma_pos[0], moma_pos[1], 0.0);
        Vector3d z0(0.0, 0.0, 1.0);

        for (size_t k = 0; k < N; ++k)
        {
            const Vector3d &p = sphere_pos_radius[k].head<3>();
            const Vector3d &g = pos_grads[k];                 
            // -------- base translation --------
            grads(0) += g(0);  // ∂p/∂x = [1,0,0]^T
            grads(1) += g(1);  // ∂p/∂y = [0,1,0]^T

            // -------- base yaw part --------
            Vector3d dptheta = z0.cross(p - o_base);  // dp/dθ
            grads(2) += g.dot(dptheta);


            // -------- only for upstream joints --------
            // which link the pts be
            int link_id = colli_link_map(static_cast<int>(k));
            int max_joint = std::min(link_id, n_dofs);

            for (int j = 0; j < max_joint; ++j)
            {
                const Vector3d &oj = joint_pos[j];    // joint j origin in world frame
                const Vector3d &zj = joint_axis[j];   // joint j axis direction in world frame

                Vector3d dpdq = zj.cross(p - oj);     // dp/dq_j
                grads(3 + j) += g.dot(dpdq);          // dC/dq_j += (dC/dp)^T * dp/dq_j
            }
        }

        return grads;
    }





    // ------------------------------------------------------------------
    // Forward kinematics for end-effector pose.
    // no use in the moma_traj_opt cause the final states are given
    // ------------------------------------------------------------------
    Eigen::VectorXd getFKPose(const Eigen::VectorXd &moma_pos)
    {
        Eigen::VectorXd fk_pose(9);
        fk_pose.setZero();
        return fk_pose;
    }

    // ------------------------------------------------------------------
    // Gradient of a scalar cost w.r.t. moma_pos given gradient w.r.t.
    // end-effector pose. For simplicity we currently only propagate the
    // position part (first 3 entries of ee_grad). Orientation gradients
    // can be added later if needed.
    // ------------------------------------------------------------------
    Eigen::VectorXd getEEGrads(const Eigen::VectorXd &moma_pos,
                               const Eigen::VectorXd &ee_grad)
    {
        Eigen::VectorXd grads = Eigen::VectorXd::Zero(3 + dof_num);
        return grads;
    }

    // ------------------------------------------------------------------
    // Visualization helpers for collision model.
    // ------------------------------------------------------------------
    visualization_msgs::MarkerArray getColliMarkerArray(Eigen::VectorXd moma_pos)
    {
        visualization_msgs::MarkerArray colli_marker_array;

        // chassis cylinder
        visualization_msgs::Marker chassis_marker;
        chassis_marker.header.frame_id = "world";
        chassis_marker.id   = 0;
        chassis_marker.type = visualization_msgs::Marker::CYLINDER;
        chassis_marker.action = visualization_msgs::Marker::ADD;
        chassis_marker.scale.x = chassis_colli_radius * 2.0;
        chassis_marker.scale.y = chassis_colli_radius * 2.0;
        chassis_marker.scale.z = chassis_height;
        chassis_marker.pose.position.x = moma_pos[0];
        chassis_marker.pose.position.y = moma_pos[1];
        chassis_marker.pose.position.z = chassis_height / 2.0;
        chassis_marker.pose.orientation.w = 1.0;
        chassis_marker.color.a = 0.5;
        chassis_marker.color.r = 0.0;
        chassis_marker.color.g = 0.0;
        chassis_marker.color.b = 1.0;
        colli_marker_array.markers.push_back(chassis_marker);

        // manipulator ball
        std::vector<Vector4d> cpts = getColliPts(moma_pos);
        for (size_t i = 0; i < cpts.size(); ++i)
        {
            visualization_msgs::Marker m;
            m.header.frame_id = "world";
            m.id   = static_cast<int>(i) + 1;
            m.type = visualization_msgs::Marker::SPHERE;
            m.action = visualization_msgs::Marker::ADD;
            m.scale.x = cpts[i][3] * 2.0;
            m.scale.y = cpts[i][3] * 2.0;
            m.scale.z = cpts[i][3] * 2.0;
            m.pose.position.x = cpts[i][0];
            m.pose.position.y = cpts[i][1];
            m.pose.position.z = cpts[i][2];
            m.pose.orientation.w = 1.0;
            m.color.a = 0.5;
            m.color.r = 0.0;
            m.color.g = 0.0;
            m.color.b = 1.0;
            colli_marker_array.markers.push_back(m);
        }

        return colli_marker_array;
    }

    visualization_msgs::MarkerArray getColliCylinderArray(Eigen::VectorXd moma_pos)
    {
        visualization_msgs::MarkerArray colli_marker_array;

        // chassis
        visualization_msgs::Marker chassis_marker;
        chassis_marker.header.frame_id = "world";
        chassis_marker.header.stamp    = ros::Time::now();
        chassis_marker.id   = 0;
        chassis_marker.type = visualization_msgs::Marker::CYLINDER;
        chassis_marker.action = visualization_msgs::Marker::ADD;
        chassis_marker.scale.x = chassis_colli_radius * 2.0;
        chassis_marker.scale.y = chassis_colli_radius * 2.0;
        chassis_marker.scale.z = chassis_height;
        chassis_marker.pose.position.x = moma_pos[0];
        chassis_marker.pose.position.y = moma_pos[1];
        chassis_marker.pose.position.z = chassis_height / 2.0;
        chassis_marker.pose.orientation.w = 1.0;
        chassis_marker.color.a = 0.5;
        chassis_marker.color.r = 0.0;
        chassis_marker.color.g = 0.0;
        chassis_marker.color.b = 1.0;
        colli_marker_array.markers.push_back(chassis_marker);

        // ----------------- get all link positions and orientations -----------------
        std::vector<Vector3d> joint_pos, joint_axis;
        std::vector<Vector4d> sphere_pos_radius;
        std::vector<Vector3d> p_all_link;
        std::vector<Matrix3d> R_all_link;

        computeKinematics(moma_pos,
                        joint_pos,
                        joint_axis,
                        sphere_pos_radius,
                        &p_all_link,
                        &R_all_link);

        int marker_id = 1;
        int n_links   = static_cast<int>(dof_num) + 1;

        for (int i = 0; i < n_links; ++i)
        {
            if (i + 1 >= static_cast<int>(p_all_link.size()) ||
                i + 1 >= static_cast<int>(R_all_link.size()))
                break;

            Vector3d p_link = p_all_link[i + 1];
            Matrix3d R_link = R_all_link[i + 1];

            double L = colli_length(i);
            if (L <= 1e-6)
                continue;

            Vector3d axis_local = colli_local_axis.row(2 * i).transpose();
            Vector3d axis_world = R_link * axis_local;
            Vector3d center = p_link + axis_world * (0.5 * L);
            // calculate q to rotate the cylinder such that its z-axis towards axis_world
            Vector3d z_axis(0.0, 0.0, 1.0);
            Eigen::Quaterniond q;
            double dot = z_axis.dot(axis_world);

            if (dot > 1.0 - 1e-6)
            {
                q = Eigen::Quaterniond::Identity();
            }
            else if (dot < -1.0 + 1e-6)
            {
                q = Eigen::AngleAxisd(M_PI, Vector3d(1.0, 0.0, 0.0));
            }
            else
            {
                Vector3d rot_axis = z_axis.cross(axis_world);
                rot_axis.normalize();
                double angle = std::acos(dot);
                q = Eigen::AngleAxisd(angle, rot_axis);
            }
            visualization_msgs::Marker m;
            m.header.frame_id = "world";
            m.header.stamp    = ros::Time::now();
            m.id   = marker_id++;
            m.type = visualization_msgs::Marker::CYLINDER;
            m.action = visualization_msgs::Marker::ADD;
            m.scale.x = cylinder_radius * 2.0;  
            m.scale.y = cylinder_radius * 2.0;
            m.scale.z = L;

            m.pose.position.x = center[0];
            m.pose.position.y = center[1];
            m.pose.position.z = center[2];
            m.pose.orientation.w = q.w();
            m.pose.orientation.x = q.x();
            m.pose.orientation.y = q.y();
            m.pose.orientation.z = q.z();

            m.color.a = 0.5;
            m.color.r = 0.0;
            m.color.g = 0.0;
            m.color.b = 1.0;

            colli_marker_array.markers.push_back(m);
        }

        return colli_marker_array;
    }

    // -------------------- code below may be unused ------------------ //
    // Simple accessors
    Eigen::VectorXd getJointLimitsMin() { return joint_pos_limit_min; }
    Eigen::VectorXd getJointLimitsMax() { return joint_pos_limit_max; }
    Eigen::VectorXd getJointVelLimits() { return joint_vel_limit; }

    // Utilities for voxelized environment (unchanged)
    int toAddress(const Eigen::Vector3i &voxel_num, const Eigen::Vector3i &id)
    {
        return id(0) * voxel_num(1) * voxel_num(2) + id(1) * voxel_num(2) + id(2);
    }

    int toAddress(const Eigen::Vector3i &voxel_num, int &x, int &y, int &z)
    {
        return x * voxel_num(1) * voxel_num(2) + y * voxel_num(2) + z;
    }

    void getBallsGrids(std::unordered_map<std::string, double> params,
                       const Eigen::VectorXd &obstacle_data,
                       Eigen::Ref<Eigen::VectorXi> result)
    {
        int             buffer_size;
        double          resolution, resolution_inv;
        Eigen::Vector3d map_origin;
        Eigen::Vector3d map_size;
        Eigen::Vector3i min_idx;
        Eigen::Vector3i max_idx;
        Eigen::Vector3i voxel_num;

        map_size[0] = params["map_size_x"];
        map_size[1] = params["map_size_y"];
        map_size[2] = params["map_size_z"];
        resolution  = params["resolution"];
        map_origin[0] = params["origin_x"];
        map_origin[1] = params["origin_y"];
        map_origin[2] = params["origin_z"];

        resolution_inv = 1.0 / resolution;

        voxel_num(0) = static_cast<int>(ceil(map_size(0) / resolution));
        voxel_num(1) = static_cast<int>(ceil(map_size(1) / resolution));
        voxel_num(2) = static_cast<int>(ceil(map_size(2) / resolution));

        min_idx = Eigen::Vector3i::Zero();
        max_idx = voxel_num - Eigen::Vector3i::Ones();

        buffer_size = voxel_num(0) * voxel_num(1) * voxel_num(2);
        result.resize(buffer_size);
        result.setZero();

        int osize = static_cast<int>(obstacle_data.size() / 4);
        for (int i = 0; i < osize; ++i)
        {
            Eigen::Vector3i id;
            Eigen::Vector3d pos(obstacle_data(i * 4 + 0),
                                obstacle_data(i * 4 + 1),
                                obstacle_data(i * 4 + 2));
            double r = obstacle_data(i * 4 + 3);
            int p = static_cast<int>(ceil(r / resolution));

            for (int x = -p; x <= p; ++x)
            {
                for (int y = -p; y <= p; ++y)
                {
                    for (int z = -p; z <= p; ++z)
                    {
                        Eigen::Vector3d ps = pos + Eigen::Vector3d(x, y, z) * resolution;
                        if ((ps - pos).norm() <= r)
                        {
                            id(0) = std::max(min_idx(0), std::min(max_idx(0), static_cast<int>(floor((ps(0) - map_origin(0)) * resolution_inv))));
                            id(1) = std::max(min_idx(1), std::min(max_idx(1), static_cast<int>(floor((ps(1) - map_origin(1)) * resolution_inv))));
                            id(2) = std::max(min_idx(2), std::min(max_idx(2), static_cast<int>(floor((ps(2) - map_origin(2)) * resolution_inv))));

                            if (id(0) >= min_idx(0) && id(0) <= max_idx(0) &&
                                id(1) >= min_idx(1) && id(1) <= max_idx(1) &&
                                id(2) >= min_idx(2) && id(2) <= max_idx(2))
                            {
                                int address = toAddress(voxel_num, id);
                                result(address) = 1;
                            }
                        }
                    }
                }
            }
        }
    }
};
