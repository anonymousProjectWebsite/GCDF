#include "planner/planner.h"
#include <sys/select.h>
#include <unistd.h>


namespace nmoma_planner
{   

    static bool wait_enter_or_key(double timeout_sec)
    {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(STDIN_FILENO, &fds);

        timeval tv;
        tv.tv_sec  = (int)timeout_sec;
        tv.tv_usec = (int)((timeout_sec - tv.tv_sec) * 1e6);

        int ret = select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv);
        if (ret > 0 && FD_ISSET(STDIN_FILENO, &fds))
        {
            //  '\n'
            char buf[256];
            (void)read(STDIN_FILENO, buf, sizeof(buf));
            return true;
        }
        return false; // timeout
    }


    static inline double wrapToPi(double a) {
        return std::atan2(std::sin(a), std::cos(a));
    }

    static Eigen::VectorXd interpState(const Eigen::VectorXd& a,
                                    const Eigen::VectorXd& b,
                                    double t)
    {
        Eigen::VectorXd out = (1.0 - t) * a + t * b;

        // yaw  index=2
        double dyaw = wrapToPi(b(2) - a(2));
        out(2) = wrapToPi(a(2) + t * dyaw);
        return out;
    }

    static std::vector<Eigen::VectorXd> resamplePathToN(const std::vector<Eigen::VectorXd>& path, int N)
    {
        std::vector<Eigen::VectorXd> out;
        if (N <= 0) return out;
        if (path.empty()) return out;

        const int M = static_cast<int>(path.size());
        if (M == 1) {
            out.assign(N, path.front());
            return out;
        }
        if (N == 1) {
            out.push_back(path.front());
            return out;
        }

        // cumulative arc length in configuration space
        std::vector<double> s(M, 0.0);
        for (int i = 1; i < M; ++i) {
            s[i] = s[i-1] + (path[i] - path[i-1]).norm();
        }
        const double total = s.back();

        out.reserve(N);
        out.push_back(path.front());

        if (total < 1e-12) {
            // 
            for (int k = 1; k < N; ++k) out.push_back(path.front());
            return out;
        }

        int seg = 0;
        for (int k = 1; k < N - 1; ++k) {
            double target = total * (static_cast<double>(k) / (N - 1));

            while (seg < M - 2 && s[seg + 1] < target) seg++;

            double s0 = s[seg];
            double s1 = s[seg + 1];
            double t = (target - s0) / std::max(1e-12, (s1 - s0));

            out.push_back(interpState(path[seg], path[seg + 1], t));
        }

        out.push_back(path.back());
        return out;
    }



    
    static double pathCostConfigSpace(const std::vector<Eigen::VectorXd>& path)
    {
        if (path.size() < 2) return 1e100;
        double cost = 0.0;
        for (size_t k = 1; k < path.size(); ++k)
            cost += (path[k] - path[k-1]).norm();
        return cost;
    }


    void Planner::init(ros::NodeHandle& nh)
    {   
        GET_PARAM_OR_THROW(nh, "agent/local_mode", local_mode);
        GET_PARAM_OR_THROW(nh, "agent/replan_interval", replan_interval);
        GET_PARAM_OR_THROW(nh, "agent/planning_horizon", planning_horizon);
        GET_PARAM_OR_THROW(nh, "agent/mode", mode);
        GET_PARAM_OR_THROW(nh, "agent/stat_num", stat_num);
        GET_PARAM_OR_THROW(nh, "agent/only_front", only_front);
        GET_PARAM_OR_THROW(nh, "agent/fixed_startgoal", fixed_startgoal);
        
        GET_PARAM_OR_THROW(nh, "agent/planner", planner_type);
        GET_PARAM_OR_THROW(nh, "agent/fixed_sequence", fixed_sequence);

        GET_PARAM_OR_THROW(nh, "agent/startgoal_dist_range", startgoal_dist_range);

        GET_PARAM_OR_THROW(nh, "agent/scene", scene);

        GET_PARAM_OR_THROW(nh, "map/pcd_folder", map_name);
        GET_PARAM_OR_THROW(nh, "map/has_goal_set", has_goal_set);
        GET_PARAM_OR_THROW(nh, "map/num_goals", num_goal);
        GET_PARAM_OR_THROW(nh, "map/video_flag", video_flag);


        int number;
        bool random_ee = true;
        nh.getParam("agent/random_ee", random_ee);
        if (!random_ee)
        {
            std::vector<double> pick, mid, place;
            nh.param<std::vector<double>>("agent/pick_state", pick, std::vector<double>());
            nh.param<std::vector<double>>("agent/mid_state", mid, std::vector<double>());
            nh.param<std::vector<double>>("agent/place_state", place, std::vector<double>());
            pick_vec = Eigen::Map<Eigen::VectorXd>(pick.data(), pick.size());
            Eigen::VectorXd mid_vec = Eigen::Map<Eigen::VectorXd>(mid.data(), mid.size());
            place_vec = Eigen::Map<Eigen::VectorXd>(place.data(), place.size());
            wps_list.push_back(pick_vec);
            wps_list.push_back(mid_vec);
            wps_list.push_back(place_vec);
            wps_list.push_back(mid_vec);
            wps_list.push_back(Eigen::VectorXd::Zero(pick_vec.size()));
        }

        grid_map.reset(new GridMap);
        grid_map->init(nh);
        
        graph_search = std::make_shared<JPS::GraphSearch>(grid_map, moma_param.chassis_colli_radius);
        birrts = std::make_shared<BiRRTs>(grid_map);
        birrts->init(nh);
        topo_prm.reset(new TopologyPRM);
        topo_prm->setEnv(grid_map);
        topo_prm->init(nh);
        mcrrts = std::make_shared<MCRRTs>(grid_map);
        mcrrts->init(nh);
        ompl_planner = std::make_shared<OMPLPlanner>(grid_map);
        ompl_planner->init(nh);
        traj_opter = std::make_shared<MomaTrajOpt>(grid_map);
        traj_opter->init(nh);
        mpc.reset(new OMPC);
        // mpc.reset(new MPC);
        mpc->init(nh);

        traj_opters.resize(8);
        mc_rrtsers.resize(8);
        for (int i = 0; i < 8; i++)
        {
            traj_opters[i] = std::make_unique<MomaTrajOpt>(grid_map);
            traj_opters[i]->init(nh);
            mc_rrtsers[i].reset(new MCRRTs(grid_map));
            mc_rrtsers[i]->init(nh);
            opt_traj_pub_list.push_back(
                nh.advertise<visualization_msgs::MarkerArray>("/opt_traj_" + std::to_string(i + 1), 1)
            );
            front_traj_pub_list.push_back(
                nh.advertise<visualization_msgs::MarkerArray>("/front_traj_" + std::to_string(i + 1), 1)
            );

            vis_isAvailable.push_back(false);
        }
        vis_front_paths.resize(8);
        vis_opt_paths.resize(8);
        vis_timer = nh.createTimer(ros::Duration(0.1), &Planner::timerCallback, this);

        plot_traj_ee = nh.advertise<visualization_msgs::Marker>("plot_traj_ee", 1);
        plot_traj_ee = nh.advertise<visualization_msgs::Marker>("plot_traj_ee_2", 1);


        front_pub = nh.advertise<visualization_msgs::MarkerArray>("/front_path", 1);
        ompl_pub = nh.advertise<visualization_msgs::MarkerArray>("/ompl_path", 1);
        end_pub = nh.advertise<visualization_msgs::MarkerArray>("/end_path", 1);
        car_traj_pub = nh.advertise<nav_msgs::Path>("/car_traj", 1);
        car_target_pub = nh.advertise<visualization_msgs::Marker>("/car_target", 1);

        bk_front_pub = nh.advertise<visualization_msgs::MarkerArray>("/bk_front_path", 1);
        bk_end_pub = nh.advertise<visualization_msgs::MarkerArray>("/bk_end_path", 1);

        init_end_pub = nh.advertise<visualization_msgs::MarkerArray>("/init_end_path", 1);
        afirst_end_pub = nh.advertise<visualization_msgs::MarkerArray>("/afirst_end_path", 1);

        prm_pub = nh.advertise<visualization_msgs::MarkerArray>("/prm_path", 1);
        vis_prm_pub = nh.advertise<visualization_msgs::MarkerArray>("/vis_prm_path", 1);

        tracking_traj = nh.advertise<visualization_msgs::MarkerArray>("/tracking_traj", 1);
        time_txt = nh.advertise<visualization_msgs::MarkerArray>("/time_txt", 1);

        moma_cmd_pub = nh.advertise<fake_moma::MomaCmd>("cmd", 1);

        // solver input pub
        solver_input_pub = nh.advertise<planner::SolverInputList>("solver_input_list", 1);   
        // solver output sub
        solver_output_sub = nh.subscribe("/swerve_base/solver_output_list", 1, &Planner::rcvSolverOutputCallBack, this);
        // wps_sub = nh.subscribe<geometry_msgs::Pose>("/manual_target", 1, &Planner::rcvWpsCallBack, this);
        state_sub = nh.subscribe("state", 1, &Planner::rcvStateCallBack, this);


        if (mode.compare("planner") == 0) 
        {
            if (random_ee)
                statistics_sub = nh.subscribe("/move_base_simple/goal", 1, &Planner::planCallBack, this);
            else
                statistics_sub = nh.subscribe("/manual_target", 1, &Planner::planCallBack, this);
        } else if (mode.compare("benchmark") == 0) {
            statistics_sub = nh.subscribe("/move_base_simple/goal", 1, &Planner::benchmarkCallback, this);
        } else if (mode.compare("ablation") == 0) {
            statistics_sub = nh.subscribe("/move_base_simple/goal", 1, &Planner::ablationCallback, this);
        } else if (mode.compare("planner_tro_exp") == 0) {
            statistics_sub = nh.subscribe("/move_base_simple/goal", 1, &Planner::planNewCallBack, this);
        } else {
            throw std::runtime_error("Unidentified mode");
        }


        if (local_mode)
        {
            std::thread cmd_thread(Planner::cmdCallback, this);
            cmd_thread.detach();
            std::thread replan_thread(Planner::replanCallback, this);
            replan_thread.detach();
            std::thread safe_thread(Planner::safeCallback, this);
            safe_thread.detach();
        }
        
        if (fixed_sequence) {
            eng = default_random_engine(42);
        } else {
            random_device rd;
            eng = default_random_engine(rd());

        }

        se2_set.setZero();
        front_path.clear();
        now_state.resize(3+moma_param.dof_num);
        now_state.setZero();
        now_dstate = now_state;
        begin_time = ros::Time::now();

        car_pts_pub =  nh.advertise<visualization_msgs::MarkerArray>("car_pts", 1);
        mani_pts_pub=  nh.advertise<visualization_msgs::Marker>("mani_pts", 100, true);
        
        plot_traj_ee = nh.advertise<visualization_msgs::Marker>("ee_traj", 1, true);
        plot_traj_ee_2=nh.advertise<visualization_msgs::Marker>("ee_traj_2", 1, true);

        mesh_traj_pub= nh.advertise<planner::MeshTraj>("mesh_traj", 1, true);


        return;
    }


    void Planner::rcvSolverOutputCallBack(const planner::SolverInputListPtr msg)
    {
        // std::vector<int> count_list = {
        //     1,2,3,4,5,6,7,8,9,10,11,12,13,14,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49
        // };
        // std::vector<int> count_list = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49
        // };
        std::vector<int> count_list = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49};
        std::vector<int> truncated_count_list;
        if(video_flag == 0){
            for(int i = 0; i < 10; i++){
                truncated_count_list.push_back(count_list[i]);
            }
        }else if(video_flag == 1){
            for(int i = 10; i < 20; i++){
                truncated_count_list.push_back(count_list[i]);
            }
        }else if(video_flag == 2){
            for(int i = 20; i < 30; i++){
                truncated_count_list.push_back(count_list[i]);
            }
        }else if(video_flag == 3){
            for(int i = 30; i < 40; i++){
                truncated_count_list.push_back(count_list[i]);
            }
        }
        planner::SolverInputList solver_output_list = *msg;
        std::cout << "[Planner] Received solver output list with " << solver_output_list.solver_input_list.size() << " trajectories." << std::endl;
        std::string pcd_path = ros::package::getPath("random_map_generator")+"/env/" + map_name;
        std::vector<Eigen::VectorXd> goal_list;
        goal_list.clear();
        std::ifstream goal_file(pcd_path + "/goal_list.txt");
        if (goal_file.is_open())
        {
            std::string line;
            int count = 0;
            while (std::getline(goal_file, line) && goal_list.size() < (size_t)num_goal)
            {
                std::istringstream iss(line);
                Eigen::VectorXd goal_state(3 + moma_param.dof_num);
                for (int i = 0; i < 3 + moma_param.dof_num; i++)
                    iss >> goal_state(i);
                // if [0, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 20, 22, 23, 25, 26, 28, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 46, 47, 49]
                // if (count == 0 || count == 2 || count == 4 || count == 5 || count == 6 || count == 7 || count == 8 || count == 9 ||
                //     count == 11 || count == 12 || count == 13 || count == 15 || count == 16 || count == 17 || count == 18 || count == 19 ||
                //     count == 20 || count == 22 || count == 23 || count == 25 || count == 26 || count == 28 || count == 32 || count == 33 ||
                //     count == 34 || count == 35 || count == 36 || count == 37 || count == 38 || count == 39 || count == 40 || count == 41 ||
                //     count == 42 || count == 44 || count == 46 || count == 47 || count == 49)
                // if [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49]
                // push to goal_list
                    if (std::find(truncated_count_list.begin(), truncated_count_list.end(), count) != truncated_count_list.end() && goal_list.size() <= 9)
                    goal_list.push_back(goal_state);
                count++;
            }
            goal_file.close();

            PRINT_YELLOW("[Planner] Show the loaded goal set:");
            vis_path_mesh(goal_list,
                tracking_traj,
                {185.0/255.0, 198.0/255.0, 232.0/255.0, 1.0}, 2020);

        }
        // visualize all the planner trajectory
        int traj_number = solver_output_list.solver_input_list.size();
        int vis_id = 0;
        for (int i = 0; i < traj_number; i++)
        {   if (video_flag == 0){
                if (i > 9) break; // only visualize first 10 trajectories to avoid too many lines
            }else if (video_flag == 1){
                if (i < 10 || i > 19) continue; // only visualize trajectories 10-19
            }else if (video_flag == 2){
                if (i < 20 || i > 29) continue; // only visualize trajectories
            }else if (video_flag == 3){
                if (i < 30 || i > 39) continue; // only visualize trajectories
            }
            planner::SolverInput solver_output = solver_output_list.solver_input_list[i];
            std::vector<Eigen::VectorXd> traj_states_truncated, traj_states;
            int step_num = solver_output.stateTrajectory.size();
            
            for (int j = 0; j < step_num; j++)
            {   
                planner::State state_msg = solver_output.stateTrajectory[j];
                Eigen::VectorXd state_vec(state_msg.data.size());
                for (size_t k = 0; k < state_msg.data.size(); k++)
                    state_vec(k) = state_msg.data[k];
                traj_states.push_back(state_vec);
                if (j % 4 != 0) continue; // downsample for visualization
                traj_states_truncated.push_back(state_vec);
            }
            // push goal state to the end
            // traj_states_truncated.push_back(goal_list[i]);
            if(goal_list.empty()){
                ROS_WARN("[Planner] goal_list is empty, skip appending goal state.");
            }else if(i >= (int)goal_list.size()){
                ROS_WARN("[Planner] goal_list index %d out of range (size=%zu).", i, goal_list.size());
            }else{
                traj_states.push_back(goal_list[i]);
            }
            vis_path_mesh(traj_states_truncated, 
            tracking_traj,
            {248.0/255.0, 235.0/255.0, 184.0/255.0, 0.1}, (vis_id + 3000)* 1000);
            vis_id++;
            vis_ee_traj(traj_states, plot_traj_ee, {185.0/255.0, 198.0/255.0, 232.0/255.0, 0.5});
        }
        return;
    }


    void Planner::cmdCallback(void *obj)
    {
        Planner *tsvr = reinterpret_cast<Planner *>(obj);
        while (true)
        {
            ros::Time start_time = ros::Time::now();
            if (tsvr->mpc->hasTraj())
            {
                // tsvr->moma_cmd_pub.publish(tsvr->mpc->getCmd(tsvr->now_state));
                tsvr->mpc->pubCmd(tsvr->now_state, tsvr->moma_cmd_pub, tsvr->gripper_open);
                double t_mpc = (ros::Time::now() - start_time).toSec() * 1000.0;
                if (t_mpc > 1000.0 / tsvr->mpc->ctrl_freq)
                    PRINT_YELLOW("[Planner] MPC time too long: " << t_mpc << " ms");
            }

            int cmd_mm_num = 1000.0 / tsvr->mpc->ctrl_freq;
            std::chrono::milliseconds dura(max(cmd_mm_num - (int)((ros::Time::now() - start_time).toSec() * 1000), 1));
            std::this_thread::sleep_for(dura);
        }
        return;
    }

    void Planner::planNewCallBack(const geometry_msgs::PoseStamped msg)
    {
        // Callback used by the fixed-goal comparison experiment.
        PRINT_GREEN("[Planner] planNewCallBack triggered.");
        bool show_goal_set = true;
        auto _checkcollision = [&](Eigen::VectorXd state) -> bool
        {
            return grid_map->isWholeBodyCollision(state);
        };
        std::vector<Eigen::VectorXd> goal_list;
        std::string pcd_path = ros::package::getPath("random_map_generator")+"/env/" + map_name;
        if (!has_goal_set){
            // for the current map, randomly generate a feasible (non-collision) goal set, std::vector<Eigen::VectorXd> goal_list 
            // and save it to a file in the corresponding folder std::string pcd_path = ros::package::getPath("random_map_generator")+"/env/" + pcd_folder
            // for ease of reading it from python and C++ node later.
            PRINT_YELLOW("[Planner] Generating goal set for the current map.");
            int num_goals = 50;
            Eigen::Vector3d min_bound = grid_map->min_boundary;
            Eigen::Vector3d max_bound = grid_map->max_boundary;
            uniform_real_distribution<double> rand_x(min_bound(0)+1.0, max_bound(0)-1.0);
            uniform_real_distribution<double> rand_y(min_bound(1)+1.0, max_bound(1)-1.0);
            uniform_real_distribution<double> rand_theta(-M_PI, M_PI);
            uniform_real_distribution<double> rand_q(0.0, 1.0);
            goal_list.clear();
            
            while (goal_list.size() < num_goals){
                Eigen::VectorXd goal_state(3 + moma_param.dof_num);
                goal_state(0) = rand_x(eng);
                goal_state(1) = rand_y(eng);
                goal_state(2) = rand_theta(eng);
                // check distance of current goal to other goals, if too close, then skip this candidate
                bool too_close = false;
                for (const auto &goal : goal_list)
                {
                    double dist = (goal_state.head(2) - goal.head(2)).norm();
                    if (dist < 1.0) { too_close = true; break; }
                }
                if (too_close) continue;
                // check the feasibility of the chasis
                double dist;
                grid_map->getDistance2d(goal_state.head(2), dist);
                if (dist < moma_param.chassis_colli_radius)
                    continue;
                double startgoal_dist = (goal_state.head(2) - now_state.head(2)).norm();

                if (startgoal_dist < startgoal_dist_range[0] || startgoal_dist > startgoal_dist_range[1])
                    continue;

                for (int i = 3; i < 3 + moma_param.dof_num; i++)
                    goal_state(i) = (moma_param.joint_pos_limit_max(i-3) - 
                                    moma_param.joint_pos_limit_min(i-3)) * rand_q(eng)
                                    + moma_param.joint_pos_limit_min(i-3);
                if (_checkcollision(goal_state))
                    continue;
                goal_list.push_back(goal_state);
            }

            std::ofstream goal_file(pcd_path + "/goal_list.txt");
            for (const auto &goal : goal_list)
            {
                for (int i = 0; i < goal.size(); i++)
                    goal_file << goal(i) << " ";
                goal_file << std::endl;
            }
            goal_file.close();
            if(show_goal_set)
            {
                PRINT_YELLOW("[Planner] Show the generated goal set:");
                vis_path_mesh(goal_list,
                    tracking_traj,
                    {174.0/255.0, 204.0/255.0, 149.0/255.0, 0.1}, 2010);
            }
            return;
        }

        // if already has a goal set, then choose planner_type to do benchmarks for crisp, remani, and topay(moma)
        PRINT_YELLOW("[Planner] Reading goal set for the current map.");
        goal_list.clear();
        std::ifstream goal_file(pcd_path + "/goal_list.txt");
        if (goal_file.is_open())
        {
            std::string line;
            while (std::getline(goal_file, line) && goal_list.size() < (size_t)num_goal)
            {
                std::istringstream iss(line);
                Eigen::VectorXd goal_state(3 + moma_param.dof_num);
                for (int i = 0; i < 3 + moma_param.dof_num; i++)
                    iss >> goal_state(i);
                goal_list.push_back(goal_state);
            }
            goal_file.close();
            if(show_goal_set)
            {
                PRINT_YELLOW("[Planner] Show the loaded goal set:");
                vis_path_mesh(goal_list,
                    tracking_traj,
                    {185.0/255.0, 198.0/255.0, 232.0/255.0, 1.0}, 2020);
            }
        }
        else
        {
            PRINT_RED("[Planner] Unable to open goal set file.");
            return;
        }
        if (planner_type.compare("moma") == 0) {
            std::vector<bool> succ_list;
            std::vector<double> time_list;
            std::vector<double> distance_translate_configuration_space_list;
            std::vector<double> distance_rotate_configuration_space_list;
            succ_list.resize(goal_list.size(), false);
            time_list.resize(goal_list.size(), 0.0);
            distance_translate_configuration_space_list.resize(goal_list.size(), 0.0);
            distance_rotate_configuration_space_list.resize(goal_list.size(), 0.0);
            int vis_id = 0;
            for (size_t i = 0; i < goal_list.size(); i++)
            {
                PRINT_GREEN("[Planner] Planning to goal " << i+1 << "/" << goal_list.size());
                ros::Time start_time = ros::Time::now();
                succ_list[i] = planMomaParallel(now_state, goal_list[i], now_dstate).first;
                time_list[i] = (ros::Time::now() - start_time).toSec();
                if (succ_list[i]) { // if successful, evaluate the quality of the trajectory
                    MomaTraj traj = end_traj;
                    vis_path_mesh(end_traj, 16, 
                    tracking_traj,
                    {248.0/255.0, 235.0/255.0, 184.0/255.0, 0.1}, (vis_id + 3000)* 1000);
                    vis_id++;
                    // sample points on the traj and compute the cumulative translational distance and rotational distance seperately
                    double dist_translate = 0.0;
                    double dist_rotate = 0.0;
                    int num_samples = 40;
                    RowMatrixXd path_m = traj.sampleTimePoints(num_samples);
                    for (int j = 1; j < num_samples; j++)
                    {
                        Eigen::VectorXd prev_state = path_m.row(j-1).head(moma_param.dof_num + 3);
                        Eigen::VectorXd curr_state = path_m.row(j).head(moma_param.dof_num + 3);
                        dist_translate += (curr_state.head(2) - prev_state.head(2)).norm();
                        // need to use warptoPi for yaw difference and normal addition for other joint angles
                        double dyaw = wrapToPi(curr_state(2) - prev_state(2));
                        Eigen::VectorXd dconfig(moma_param.dof_num + 1);
                        dconfig << dyaw, curr_state.tail(moma_param.dof_num) - prev_state.tail(moma_param.dof_num);
                        dist_rotate += dconfig.norm();
                    }
                    distance_translate_configuration_space_list[i] = dist_translate;
                    distance_rotate_configuration_space_list[i] = dist_rotate;
                }
            }
            // save the results
            std::ofstream result_file(pcd_path + "/topay_results.txt");
            for (size_t i = 0; i < goal_list.size(); i++)
            {
                result_file << succ_list[i] << " " << time_list[i] << " "
                            << distance_translate_configuration_space_list[i] << " "
                            << distance_rotate_configuration_space_list[i] << std::endl;
            }
            result_file.close();      
        }
        else if(planner_type.compare("remani") == 0)
        {
            // different node, my idea is to publish the goal message to the corresponding solver
            // TODO: implement remani benchmark publishing
        }
        else if(planner_type.compare("crisp") == 0){
            // different node, my idea is to publish the goal message to the corresponding solver
            // State[]  stateTrajectory
            // State x0
            // State xf
            std::vector<bool> front_fail_flags;
            bool with_initial_guess = false;      
            planner::SolverInputList solver_input_list_msg;
            solver_input_list_msg.solver_input_list.resize(goal_list.size());
            for (size_t i = 0; i < goal_list.size(); i++)
            {
                planner::SolverInput solver_input_msg;
                for (int j = 0; j < 3 + moma_param.dof_num; j++)
                {
                    solver_input_msg.x0.data[j] = now_state(j);
                    solver_input_msg.xf.data[j] = goal_list[i](j);
                    const int N = 40;
                    solver_input_msg.stateTrajectory.resize(N);
                    for (int k = 0; k < N; ++k) {
                        for (int j = 0; j < 3 + moma_param.dof_num; ++j)
                            solver_input_msg.stateTrajectory[k].data[j] = 0.0;
                    }
                }
                if (with_initial_guess)
                {
                    // to be implement: use the front-end part to give initial guess to crisp through stateTrajectory
                    // 1)  topo pathsnon-critical
                    topo_select_paths.clear();
                    list<GraphNode::Ptr> graph;
                    std::vector<std::vector<Eigen::Vector3d>> raw_paths, filtered_paths;

                    // topo search  xy
                    Eigen::Vector3d topo_start(now_state(0), now_state(1), 0.0);
                    Eigen::Vector3d topo_end  (goal_list[i](0), goal_list[i](1), 0.0);
                    std::vector<Eigen::Vector3d> start_pts{topo_start};
                    std::vector<Eigen::Vector3d> end_pts{topo_end};

                    topo_prm->findTopoPaths(topo_start, topo_end,
                                            start_pts, end_pts,
                                            graph, raw_paths, filtered_paths,
                                            topo_select_paths,
                                            /*_critical=*/false);

                    if (topo_select_paths.empty())
                    {
                        PRINT_YELLOW("[CRISP] No topo path (non-critical) for goal " << i << ", fallback no guess.");
                        front_fail_flags.push_back(true);
                    }
                    else
                    {
                        // 2)  topo pathdensify -> MCRRT -> full_path
                        bool best_found = false;
                        double best_cost = 1e100;
                        std::vector<Eigen::VectorXd> best_path;

                        const double start_yaw = now_state(2);
                        const double end_yaw   = goal_list[i](2);

                        for (size_t pid = 0; pid < topo_select_paths.size(); ++pid)
                        {
                            // topo 3D -> 2D
                            std::vector<Eigen::Vector2d> in_path_2d;
                            in_path_2d.reserve(topo_select_paths[pid].size());
                            for (const auto& wp : topo_select_paths[pid])
                                in_path_2d.push_back(wp.head(2));

                            // densify MOMA 
                            auto dense_result = graph_search->getDensePath(
                                in_path_2d,
                                1.414,
                                start_yaw, end_yaw,
                                moma_param.max_v, moma_param.max_w
                            );

                            std::vector<Eigen::VectorXd> full_path;
                            bool ok = mc_rrtsers[0]->plan(now_state, goal_list[i], dense_result, full_path);

                            if (!ok || full_path.empty()){
                                PRINT_YELLOW("[CRISP] MCRRT failed for goal " << i << " on topo path " << pid << ".");
                                continue;
                            }

                            double cost = pathCostConfigSpace(full_path);
                            if (cost < best_cost) {
                                best_cost = cost;
                                best_path = std::move(full_path);
                                best_found = true;
                                break; // 
                            }
                        }

                        if (!best_found)
                        {
                            // PRINT_YELLOW("[CRISP] All MCRRT failed for goal " << i << ", fallback no guess.");
                            PRINT_RED("[CRISP] No valid initial guess found for CRISP for goal " << i << ", consider as failure.");
                            front_fail_flags.push_back(true);
                        }
                        else
                        {   
                            front_fail_flags.push_back(false);
                            const int N = 40; // crisp 
                            auto guess_path = resamplePathToN(best_path, N);

                            //  stateTrajectory
                            solver_input_msg.stateTrajectory.resize(N);
                            for (int k = 0; k < N; ++k) {
                                for (int j = 0; j < 3 + moma_param.dof_num; ++j)
                                    solver_input_msg.stateTrajectory[k].data[j] = guess_path[k](j);
                            }
                        }
                    }
                }
                // store this solver input into the list
                solver_input_list_msg.solver_input_list[i] = solver_input_msg;
            }
            solver_input_pub.publish(solver_input_list_msg);
            // save front fail flags to a text file
            // std::ofstream result_file(pcd_path + "/crisp_front_fail_flags.txt");
            // for (size_t i = 0; i < goal_list.size(); i++)
            // {
            //     result_file << front_fail_flags[i] << std::endl;
            // }
            // result_file.close();
        }


        else
            PRINT_RED("[Planner] Unknown planner type for benchmark experiment.");
        return;
    }









    void Planner::planCallBack(const geometry_msgs::PoseStamped msg)
    {
        Eigen::VectorXd end_state = Eigen::VectorXd::Zero(3+moma_param.dof_num);

        if (msg.header.frame_id.compare("target") == 0)
        {
            if (!wps_list.empty())
                end_state = wps_list[0];
            // Eigen::VectorXd moma_set = Eigen::VectorXd::Zero(3+moma_param.dof_num);
            // moma_set.head(3) = Eigen::Vector3d(msg.pose.position.x, msg.pose.position.y, 0.0);
            // Eigen::VectorXd ee_set = Eigen::VectorXd::Zero(9);
            // ee_set.head(3) = Eigen::Vector3d(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z);
            // Eigen::Matrix3d R = Eigen::Quaterniond(msg.pose.orientation.w, msg.pose.orientation.x, 
            //                                      msg.pose.orientation.y, msg.pose.orientation.z).toRotationMatrix();
            // ee_set.segment(3, 3) = R.row(0);
            // ee_set.tail(3) = R.row(1);
            // if (traj_opter->optimizeEE(moma_set, ee_set))
            //     end_state = moma_set;
            // else
            //     return;
        }else
        {
            se2_set(0) = msg.pose.position.x;
            se2_set(1) = msg.pose.position.y;
            double dist;
            grid_map->getDistance2d(se2_set.head(2), dist);
            if (dist < moma_param.chassis_colli_radius)
                return;

            se2_set(2) = atan2(2.0*msg.pose.orientation.z*msg.pose.orientation.w, 
                                2.0*msg.pose.orientation.w*msg.pose.orientation.w-1.0);
            // visualization_msgs::Marker marker;
            // marker.header.frame_id = "world";
            // marker.header.stamp = ros::Time::now();
            // marker.id = 10086;
            // marker.action = visualization_msgs::Marker::ADD;
            // marker.type = visualization_msgs::Marker::SPHERE;
            // marker.scale.x = moma_param.chassis_colli_radius * 2.0;
            // marker.scale.y = moma_param.chassis_colli_radius * 2.0;
            // marker.scale.z = moma_param.chassis_height;
            // marker.pose.position.x = se2_set[0];
            // marker.pose.position.y = se2_set[1];
            // marker.pose.position.z = moma_param.chassis_height / 2.0;
            // marker.pose.orientation.w = 1.0;
            // marker.color.a = 0.5;
            // marker.color.r = 0.5;
            // marker.color.g = 0.5;
            // marker.color.b = 0.5;
            // car_target_pub.publish(marker);

            uniform_real_distribution<double> rand_q(0.0, 1.0);
            end_state.resize(3+moma_param.dof_num);
            end_state.setZero();
            end_state.head(3) = se2_set;
            
            ros::Time find_time_start = ros::Time::now();
            bool timeout;
            do
            {
                for (size_t i=0; i<moma_param.dof_num; i++)
                    end_state(3+i) = (moma_param.joint_pos_limit_max(i) - 
                                    moma_param.joint_pos_limit_min(i)) * rand_q(eng)
                                    + moma_param.joint_pos_limit_min(i);
            } while (grid_map->isWholeBodyCollision(end_state)
                    && !(timeout = (ros::Time::now() - find_time_start).toSec() > 1.0));

            if(timeout) return;
        }

        global_goal = end_state;
        has_goal = true;
        Eigen::VectorXd local_start = now_state;
        Eigen::VectorXd local_v = now_dstate;
        if (has_traj && local_mode)
        {
            double t = (ros::Time::now() - last_replan_time).toSec() +planning_budget;
            local_start = end_traj.getState(t);
            local_v = end_traj.getDState(t);
        }

        ros::Time start_time = ros::Time::now();
        bool succ = false;
        if (planner_type.compare("moma") == 0)
            succ = planMomaParallel(local_start, end_state, local_v).first;
        
        double duration = (ros::Time::now() - start_time).toSec() * 1000.0;
        PRINT_GREEN("[Planner] Total planning time: " << duration << " ms");

        vis_time(time_txt, duration, 4546);
        vis_path_mesh(end_traj, 32, 
            tracking_traj,
            {226.0/255.0, 145.0/255.0, 53.0/255.0, 0.1}, 2010);

        geometry_msgs::Point pos;
        pos.x = 1.0;
        pos.y = 2.0;
        pos.z = 1.0;
        

        // if (succ)
        // {
        //     global_traj = end_traj;
        //     mpc->setTraj(end_traj, 0.0);
        //     begin_time = ros::Time::now();
        //     has_traj = true;
        // }

        return;
    }

    void Planner::ablationCallback(const geometry_msgs::PoseStamped msg)
    {
        Eigen::Vector3d start = Eigen::Vector3d::Zero(3);
        Eigen::Vector3d end;
        end << 3.0, 3.0, 0.0;
        
        Eigen::VectorXd start_state = Eigen::VectorXd::Zero(3+moma_param.dof_num);
        start_state.head(3) = start;
        Eigen::VectorXd end_state = Eigen::VectorXd::Zero(3+moma_param.dof_num);
        end_state.head(3) = end;

        auto _checkcollision = [&](Eigen::VectorXd state) -> bool
        {
            return grid_map->isWholeBodyCollision(state);
        };
        
        int comparison_num = 0;

        int num_moma_succ = 0;
        double mean_moma_path_length = 0.0;
        double mean_moma_duration = 0.0;

        int num_nontopo_succ = 0;
        double mean_nontopo_path_length = 0.0;
        double mean_nontopo_duration = 0.0;

        int num_seq_succ = 0;
        double mean_seq_path_length = 0.0;
        double mean_seq_duration = 0.0;

        size_t num_plan = 0;
        for (; num_plan<stat_num && ros::ok(); num_plan++) {
            // Randomly generate map, start and end state
            do {
                if (!fixed_startgoal) {
                    Eigen::Vector3d min_bound = grid_map->min_boundary;
                    Eigen::Vector3d max_bound = grid_map->max_boundary;
                    
                    uniform_real_distribution<double> rand_x(min_bound(0)+2.0, max_bound(0)-2.0);
                    uniform_real_distribution<double> rand_y(min_bound(1)+2.0, max_bound(1)-2.0);
                    uniform_real_distribution<double> rand_theta(-M_PI, M_PI);
                    uniform_real_distribution<double> rand_q(0.0, 1.0);
                    
                    end << rand_x(eng), rand_y(eng), rand_theta(eng);
                    end_state.head(3) = end;
                    
                    start_state.head(3) = start;
                    
                    double startgoal_dist = (start.head(2) - end.head(2)).norm();
                    
                    if (startgoal_dist < startgoal_dist_range[0]
                    || startgoal_dist > startgoal_dist_range[1]) continue;

                    
                    if (scene.compare("cuboids") == 0) {
                        grid_map->regenerateMap();
                    } else if (scene.compare("tables") == 0) {
                        grid_map->regenerateDesk({
                            start_state.head(2),
                            end_state.head(2)
                        });
                    }

                    // Early check: if the chassis at start or end collides with the map,
                    // resample the base pose instead of repeatedly randomizing arm joints.
                    // This avoids wasting time when the SE2 pose is infeasible.
                    if (grid_map->isCollision2d(start_state.head(2), moma_param.chassis_colli_radius)
                        || grid_map->isCollision2d(end_state.head(2), moma_param.chassis_colli_radius))
                        continue;

                    // Early check: if the chassis at start or end collides with the map,
                    // resample the base pose instead of repeatedly randomizing arm joints.
                    // This avoids wasting time when the SE2 pose is infeasible.
                    if (grid_map->isCollision2d(start_state.head(2), moma_param.chassis_colli_radius)
                        || grid_map->isCollision2d(end_state.head(2), moma_param.chassis_colli_radius))
                        continue;
                    else {
                        throw std::runtime_error("Invalid scene type.");
                    }

                    ros::Time time_start;
                    bool timeout, start_state_collision, end_state_collision;
                    time_start = ros::Time::now();
                    do
                    {
                        for (size_t i=0; i<moma_param.dof_num; i++)
                            end_state(3+i) = (moma_param.joint_pos_limit_max(i) - 
                                            moma_param.joint_pos_limit_min(i)) * rand_q(eng)
                                            + moma_param.joint_pos_limit_min(i);
                    } while ((end_state_collision = _checkcollision(end_state))
                            && !(timeout = (ros::Time::now() - time_start).toSec() > 1.0));
                    if(timeout || end_state_collision) continue;
    
                    time_start = ros::Time::now();
                    do
                    {
                        for (size_t i=0; i<moma_param.dof_num; i++)
                            start_state(3+i) = (moma_param.joint_pos_limit_max(i) - 
                                            moma_param.joint_pos_limit_min(i)) * rand_q(eng)
                                            + moma_param.joint_pos_limit_min(i);
                    } while ((start_state_collision = _checkcollision(start_state))
                            && !(timeout = (ros::Time::now() - time_start).toSec() > 1.0));
                    if(timeout || start_state_collision) continue;
                    break;
                } else if (fixed_startgoal) {
                    grid_map->regenerateDesk({
                        start_state.head(2),
                        end_state.head(2)
                    });
                    if(_checkcollision(start_state) || _checkcollision(end_state))
                        continue;
                    break;
                }
            } while (true);
            
            

            if (_checkcollision(start_state) || _checkcollision(end_state))
                PRINT_RED("Collision detected, skip this data point.");
            
            Eigen::VectorXd v = Eigen::VectorXd::Zero(3+moma_param.dof_num);
            
            bool moma_succ = false;
            double moma_path = 0.0;
            double moma_duration = 0.0;
            float moma_optim_time = 0.0;
            MomaTraj moma_traj;
            {
                ros::Time start_time = ros::Time::now();
                std::tie(moma_succ, moma_traj) = planMomaParallel(start_state, end_state, v);
                moma_duration = (ros::Time::now() - start_time).toSec();
                if(moma_succ) {
                    num_moma_succ++;
                    moma_path = moma_traj.getTotalDuration();
                }
            }

            bool nontopo_succ = false;
            double nontopo_path = 0.0;
            double nontopo_duration = 0.0;
            float nontopo_optim_time = 0.0;
            {
                ros::Time start_time = ros::Time::now();
                std::tie(nontopo_succ, nontopo_optim_time) = planMomaNonTOPO(start_state, end_state, v);
                nontopo_duration = (ros::Time::now() - start_time).toSec();
                if(nontopo_succ){
                    num_nontopo_succ++;
                    nontopo_path = end_traj.getTotalDuration();
                }
            }


            bool seq_succ = false;
            double seq_path = 0.0;
            double seq_duration = 0.0;
            float seq_optim_time = 0.0;
            {
                ros::Time start_time = ros::Time::now();
                std::tie(seq_succ, seq_optim_time) = planMomaSequential(start_state, end_state, v);
                seq_duration = (ros::Time::now() - start_time).toSec();
                if(seq_succ) {
                    num_seq_succ++;
                    seq_path = end_traj.getTotalDuration();
                }
            }

            if (moma_succ && nontopo_succ && seq_succ) {
                comparison_num++;

                // moma
                mean_moma_path_length += (moma_path - mean_moma_path_length) / comparison_num;
                mean_moma_duration += (moma_duration - mean_moma_duration) / comparison_num;

                // nontopo
                mean_nontopo_path_length += (nontopo_path - mean_nontopo_path_length) / comparison_num;
                mean_nontopo_duration += (nontopo_duration - mean_nontopo_duration) / comparison_num;

                // seq
                mean_seq_path_length += (seq_path - mean_seq_path_length) / comparison_num;
                mean_seq_duration += (seq_duration - mean_seq_duration) / comparison_num;
            }
        }

        PRINT_GREEN("[Planner] benchmark done.");
        PRINT_GREEN("MOMA:\t"       << num_moma_succ    << "/" << num_plan  << "\tAvg. Plan time: " << mean_moma_duration   << "ms\t Avg. Path Length: " << mean_moma_path_length   << "ms" << std::endl);
        PRINT_GREEN("NONTOPO:\t"    << num_nontopo_succ << "/" << num_plan  << "\tAvg. Plan time: " << mean_nontopo_duration<< "ms\t Avg. Path Length: " << mean_nontopo_path_length<< "ms" << std::endl);
        PRINT_GREEN("SEQ:\t"        << num_seq_succ     << "/" << num_plan  << "\tAvg. Plan time: " << mean_seq_duration    << "ms\t Avg. Path Length: " << mean_seq_path_length    << "ms" << std::endl);

        PRINT_GREEN("Comparison:\tNum: " << comparison_num << std::endl);
        return;
    }
    
    void Planner::benchmarkCallback(const geometry_msgs::PoseStamped msg)
    {
        Eigen::Vector3d start = Eigen::Vector3d::Zero(3);
        Eigen::Vector3d end;
        end << 3.0, 3.0, 0.0;
        
        Eigen::VectorXd start_state = Eigen::VectorXd::Zero(3+moma_param.dof_num);
        start_state.head(3) = start;
        Eigen::VectorXd end_state = Eigen::VectorXd::Zero(3+moma_param.dof_num);
        end_state.head(3) = end;

        auto _checkcollision = [&](Eigen::VectorXd state) -> bool
        {
            return grid_map->isWholeBodyCollision(state);
        };
        
        int comparison_num = 0;

        int num_moma_succ = 0;
        double mean_moma_path_length = 0.0;
        double mean_moma_duration = 0.0;

        size_t num_plan = 0;
        for (; num_plan<stat_num && ros::ok(); num_plan++) {
            // Randomly generate map, start and end state
            do {
                if (!fixed_startgoal) {
                    Eigen::Vector3d min_bound = grid_map->min_boundary;
                    Eigen::Vector3d max_bound = grid_map->max_boundary;
                    
                    uniform_real_distribution<double> rand_x(min_bound(0)+2.0, max_bound(0)-2.0);
                    uniform_real_distribution<double> rand_y(min_bound(1)+2.0, max_bound(1)-2.0);
                    uniform_real_distribution<double> rand_theta(-M_PI, M_PI);
                    uniform_real_distribution<double> rand_q(0.0, 1.0);
                    
                    end << rand_x(eng), rand_y(eng), rand_theta(eng);
                    end_state.head(3) = end;
                    
                    start << rand_x(eng), rand_y(eng), rand_theta(eng);
                    start_state.head(3) = start;
                    
                    double startgoal_dist = (start.head(2) - end.head(2)).norm();
                    
                    if (startgoal_dist < startgoal_dist_range[0]
                    || startgoal_dist > startgoal_dist_range[1]) continue;

                    if (scene.compare("cuboids") == 0) {
                        grid_map->regenerateMap();
                    } else if (scene.compare("tables") == 0) {
                        grid_map->regenerateDesk({
                            start_state.head(2),
                            end_state.head(2)
                        });
                    }
                    else {
                        throw std::runtime_error("Invalid scene type.");
                    }

                    ros::Time time_start;
                    bool timeout, start_state_collision, end_state_collision;
                    time_start = ros::Time::now();
                    do
                    {
                        for (size_t i=0; i<moma_param.dof_num; i++)
                            end_state(3+i) = (moma_param.joint_pos_limit_max(i) - 
                                            moma_param.joint_pos_limit_min(i)) * rand_q(eng)
                                            + moma_param.joint_pos_limit_min(i);
                    } while ((end_state_collision = _checkcollision(end_state))
                            && !(timeout = (ros::Time::now() - time_start).toSec() > 1.0));
                    if(timeout || end_state_collision) continue;
    
                    time_start = ros::Time::now();
                    do
                    {
                        for (size_t i=0; i<moma_param.dof_num; i++)
                            start_state(3+i) = (moma_param.joint_pos_limit_max(i) - 
                                            moma_param.joint_pos_limit_min(i)) * rand_q(eng)
                                            + moma_param.joint_pos_limit_min(i);
                    } while ((start_state_collision = _checkcollision(start_state))
                            && !(timeout = (ros::Time::now() - time_start).toSec() > 1.0));
                    if(timeout || start_state_collision) continue;
                    break;
                } else if (fixed_startgoal) {
                    grid_map->regenerateDesk({
                        start_state.head(2),
                        end_state.head(2)
                    });
                    if(_checkcollision(start_state) || _checkcollision(end_state))
                        continue;
                    break;
                }
            } while (true);
            
            if (_checkcollision(start_state) || _checkcollision(end_state))
                PRINT_RED("Collision detected, skip this data point.");
            
            Eigen::VectorXd v = Eigen::VectorXd::Zero(3+moma_param.dof_num);
            
            bool moma_succ = false;
            double moma_path = 0.0;
            double moma_duration = 0.0;
            float moma_optim_time = 0.0;
            MomaTraj moma_traj;
            {
                ros::Time start_time = ros::Time::now();
                std::tie(moma_succ, moma_traj) = planMomaParallel(start_state, end_state, v);
                moma_duration = (ros::Time::now() - start_time).toSec();
                if(moma_succ) {
                    num_moma_succ++;
                    moma_path = moma_traj.getTotalDuration();
                }
            }

            if (moma_succ) {
                comparison_num++;

                // moma
                mean_moma_path_length += (moma_path - mean_moma_path_length) / comparison_num;
                mean_moma_duration += (moma_duration - mean_moma_duration) / comparison_num;
            }
        }

        PRINT_GREEN("[Planner] benchmark done.");
        PRINT_GREEN("MOMA:\t"       << num_moma_succ    << "/" << num_plan  << "\tAvg. Plan time: " << mean_moma_duration   << "ms\t Avg. Path Length: " << mean_moma_path_length   << "ms" << std::endl);

        PRINT_GREEN("Comparison:\tNum: " << comparison_num << std::endl);
        return;
    }
    
    void Planner::safeCallback(void *obj)
    {
        Planner *tsvr = reinterpret_cast<Planner *>(obj);
        while (true)
        {
            ros::Time start_time = ros::Time::now();
            if (tsvr->has_goal && tsvr->has_traj && tsvr->is_safe && (!tsvr->in_plan) )
            {
                Eigen::VectorXd temp_state = Eigen::VectorXd::Zero(tsvr->moma_param.dof_num + 3);
                std::vector<Eigen::Vector4d> min_dist_mani = tsvr->moma_param.getColliPts(temp_state);
                double res = 0.01;
                for (double t=0.0; t<tsvr->end_traj.getTotalDuration(); t+=res)
                {
                    Eigen::VectorXd state = tsvr->end_traj.getState(t);
        
                    double d = 0.0;
                    tsvr->grid_map->getDistance2d(state.head(2), d);
                    if (d < tsvr->moma_param.chassis_colli_radius * 0.99)
                    {
                        tsvr->is_safe = false;
                        break;
                    }
                    std::vector<Eigen::Vector4d> mani_pts = tsvr->moma_param.getColliPts(state);
                    for (size_t i=0; i<mani_pts.size(); i++)
                    {
                        double d = 0.0;
                        tsvr->grid_map->getDistance3d(mani_pts[i].head(3), d);
                        if (d < min_dist_mani[i].w() * 0.99)
                        {
                            tsvr->is_safe = false;
                            break;
                        }
                    }
                    if (!tsvr->is_safe) break;
                }
            }

            int cmd_mm_num = 1000.0 / tsvr->mpc->ctrl_freq;
            std::chrono::milliseconds dura(max(cmd_mm_num - (int)((ros::Time::now() - start_time).toSec() * 1000), 1));
            std::this_thread::sleep_for(dura);
        }
    }

    void Planner::replanCallback(void *obj)
    {
        Planner *tsvr = reinterpret_cast<Planner *>(obj);
        while (true)
        {
            ros::Time start_time = ros::Time::now();
            
            if (tsvr->has_goal && tsvr->has_traj)
            {
                if ((tsvr->now_state.head(2)-tsvr->global_goal.head(2)).norm () < 0.5)
                {
                    tsvr->has_goal = false;
                    tsvr->has_traj = false;
                    if (tsvr->wps_list.size() > 1)
                    {
                        if ((tsvr->global_goal-tsvr->place_vec).norm() < 0.1 ||
                            (tsvr->global_goal-tsvr->pick_vec).norm() < 0.1)
                        {
                            while (!tsvr->mpc->atGoal())
                                ROS_DEBUG("Waiting reaching goal...");
                            // in
                            Eigen::Vector3d direct;
                            direct.head(2) = tsvr->now_state.head(2) + \
                                            0.1 * Eigen::Vector2d(cos(tsvr->now_state(2)), sin(tsvr->now_state(2)));
                            tsvr->mpc->setDirect(direct);
                            while (!tsvr->mpc->atGoal())
                                ROS_DEBUG("Waiting reaching goal...");
                            // pick or place
                            tsvr->gripper_open = !tsvr->gripper_open;
                            this_thread::sleep_for(chrono::milliseconds(1000));
                            // out
                            direct.head(2) = tsvr->now_state.head(2) - \
                                            1.0 * Eigen::Vector2d(cos(tsvr->now_state(2)), sin(tsvr->now_state(2)));
                            tsvr->mpc->setDirect(direct);
                            while (!tsvr->mpc->atGoal())
                                ROS_DEBUG("Waiting reaching goal...");
                            this_thread::sleep_for(chrono::milliseconds(1000));
                        }
                        
                        tsvr->wps_list.erase(tsvr->wps_list.begin());
                        tsvr->global_goal = tsvr->wps_list.front();
                        PRINT_GREEN("New goal: " << tsvr->global_goal.transpose());
                        tsvr->has_goal = true;
                        bool succ;
                        Eigen::VectorXd local_start = tsvr->now_state;
                        Eigen::VectorXd local_v = tsvr->now_dstate;
                        if (tsvr->planner_type.compare("moma") == 0)
                        {
                            tsvr->in_plan = true;
                            succ = tsvr->planMomaParallel(local_start, tsvr->global_goal, local_v).first;
                            tsvr->in_plan = false;
                        }
                        if (succ)
                        {
                            tsvr->global_traj = tsvr->end_traj;
                            tsvr->mpc->setTraj(tsvr->end_traj, 0.0);
                            tsvr->begin_time = ros::Time::now();
                            tsvr->has_traj = true;
                            tsvr->is_safe = true;
                        }
                    }
                }
                else
                {
                    if ((ros::Time::now() - tsvr->last_replan_time).toSec() > tsvr->replan_interval || \
                        ! tsvr->is_safe)
                    {
                        ros::Time plan_start_time = ros::Time::now();
                        Eigen::VectorXd local_goal;
                        Eigen::VectorXd local_start = tsvr->now_state;
                        Eigen::VectorXd local_v = tsvr->now_dstate;
                        {
                            double t = (ros::Time::now() - tsvr->last_replan_time).toSec() + tsvr->planning_budget;
                            local_start = tsvr->end_traj.getState(t);
                            local_v = tsvr->end_traj.getDState(t);
                        }
                        {
                            double t = (ros::Time::now() - tsvr->begin_time).toSec();
                            bool found = false;
                            for (; t<tsvr->global_traj.getTotalDuration(); t+=0.1)
                            {
                                Eigen::VectorXd state = tsvr->global_traj.getState(t);
                                if ((state.head(2)-local_start.head(2)).norm() > tsvr->planning_horizon)
                                {
                                    local_goal = state;
                                    found = true;
                                    break;
                                }
                            }
                            if (!found)
                                local_goal = tsvr->global_goal;
                        }

                        PRINT_GREEN("local goal: " << local_goal.transpose());

                        tsvr->in_plan = true;
                        if (tsvr->planMomaParallel(local_start, local_goal, local_v).first)
                        {
                            tsvr->is_safe = true;
                            while (ros::Time::now() - plan_start_time < ros::Duration(tsvr->planning_budget)) {;}
                            // planning_budget = (ros::Time::now() - start_time).toSec();
                            double wait_time = (ros::Time::now() - plan_start_time).toSec() - tsvr->planning_budget;
                            PRINT_GREEN("Wait time = "<<wait_time<<" s.");
                            tsvr->mpc->setTraj(tsvr->end_traj, std::max(0.0, wait_time));
                            tsvr->begin_time = ros::Time::now();
                            tsvr->last_replan_time = ros::Time::now();
                            tsvr->has_traj = true;
                        }
                        tsvr->in_plan = false;
                    }
                }
            }

            int cmd_mm_num = 1000.0 / tsvr->mpc->ctrl_freq;
            std::chrono::milliseconds dura(max(cmd_mm_num - (int)((ros::Time::now() - start_time).toSec() * 1000), 1));
            std::this_thread::sleep_for(dura);
        }
        return;
    }

    void Planner::rcvStateCallBack(const fake_moma::MomaStatePtr msg)
    {
        has_odom = true;
        now_state[0] = msg->chassis_odom.pose.pose.position.x;
        now_state[1] = msg->chassis_odom.pose.pose.position.y;
        double ori_z = msg->chassis_odom.pose.pose.orientation.z;
        double ori_w = msg->chassis_odom.pose.pose.orientation.w;
        now_state[2] = atan2(2.0*ori_z*ori_w, 
                             2.0*ori_w*ori_w-1.0);
        now_dstate[0] = msg->chassis_odom.twist.twist.linear.x;
        now_dstate[1] = msg->chassis_odom.twist.twist.angular.z;
        // now_dstate[0] = 0.0;
        for (size_t i=0; i<moma_param.dof_num; i++)
        {
            now_state[3+i] = msg->arm_odom[i].twist.twist.linear.x;
            now_dstate[3+i] = msg->arm_odom[i].twist.twist.angular.z;
        }
        return;
    }

    std::vector<Eigen::VectorXd> Planner::planOmpls(const Eigen::VectorXd& start, const Eigen::VectorXd& end, const Eigen::VectorXd& start_v) const
    {
        // front end
        std::vector<Eigen::VectorXd> path;
        // ompls
        PRINT_GREEN("\n[Planner] Begin OMPL planning...");
        ompl::msg::setLogLevel(ompl::msg::LOG_NONE);
        ompl_planner->planRRT(start, end, path);

        return path;
    }

    std::pair<bool, MomaTraj> Planner::planMomaParallel(const Eigen::VectorXd& start, 
                                    const Eigen::VectorXd& end, 
                                    const Eigen::VectorXd& start_v)
    {
        bool succ = false;                              // flag for successful optimization
        bool _critical = true;                         // whether to use critical map
        std::vector<Eigen::Vector4d> colors;            // colors of prm paths
        std::vector<std::pair<bool, MomaTraj>> results; // storing results of optimization
        std::vector<std::vector<Eigen::VectorXd>> front_paths;       // storing pre-optimized paths
        MomaTraj ret_traj;

        do {
            // start first trial with non-critical
            topo_select_paths.clear();
            {
                list<GraphNode::Ptr> graph;
                vector<vector<Eigen::Vector3d>> raw_paths, filtered_paths;
                Eigen::Vector3d topo_start(start(0), start(1), 0.0);
                Eigen::Vector3d topo_end(end(0), end(1), 0.0);
                std::vector<Eigen::Vector3d> start_pts, end_pts;
                start_pts.push_back(topo_start);
                end_pts.push_back(topo_end);
                // time for finding topo paths
                // PRINT_GREEN("[MOMA] Finding topo paths...");
                ros::Time find_path_start_time = ros::Time::now();
                topo_prm->findTopoPaths(topo_start, topo_end, start_pts, end_pts, graph,
                                       raw_paths, filtered_paths, topo_select_paths, false);
                // if (!_critical) {
                //     auto jps_result = graph_search->plan2dJPS(start.head(2), end.head(2), moma_param.chassis_colli_radius+0.1);
                //     if (!jps_result.empty())
                //     {
                //         std::vector<Eigen::Vector3d> jps3_res;
                //         for (size_t i = 0; i < jps_result.size(); ++i)
                //             jps3_res.push_back(Eigen::Vector3d(jps_result[i].x(), jps_result[i].y(), 0.0));
                //         topo_select_paths.push_back(jps3_res);
                //     }
                // }
                
                if (topo_select_paths.empty()){
                    PRINT_RED("[MOMA] Fail to find topo paths.");
                    return std::make_pair(false, ret_traj);
                }
        
                if(topo_select_paths.size() > traj_opters.size()) throw std::runtime_error("Too many paths to optimize");
                
                // PRINT_GREEN("[MOMA] Found " << topo_select_paths.size() << " topo paths in "
                //              << (ros::Time::now() - find_path_start_time).toSec()*1000.0 << " ms");
                colors = vis_prm_paths();
        
                PRINT_GREEN("[MOMA] Start optimization");
        
                results.resize(topo_select_paths.size());
                front_paths.resize(topo_select_paths.size());
                for (auto &res : results) { res.first = false; }
                
                std::promise<bool> promise_succ;
                auto future_succ = promise_succ.get_future();

                boost::mutex mtx;
                boost::condition_variable cv_first; // for the first successful thread
                boost::condition_variable cv_all;   // for all threads to finish
                std::atomic_flag rdy_flag = ATOMIC_FLAG_INIT;
                std::atomic<int> completed_threads{0};
                auto worker = [this, &results, &front_paths, &mtx, &promise_succ, &cv_first, &cv_all, &completed_threads, &rdy_flag] (
                    int idx, 
                    std::vector<Eigen::Vector3d>& topo_path, 
                    const Eigen::VectorXd& start, 
                    const Eigen::VectorXd& end, 
                    const Eigen::VectorXd& start_v)
                {
                    ros::Time start_time = ros::Time::now();
                    std::vector<Eigen::Vector2d> in_path;
                    for(auto &wp : topo_path)
                        in_path.push_back(wp.head(2));
                    auto dense_result = graph_search->getDensePath(in_path, 1.414, start(2), end(2), 
                        moma_param.max_v, moma_param.max_w);
                        
                    boost::this_thread::interruption_point();
                    // print size of dense_result
                    // PRINT_GREEN("[Thread] ID: " << idx << " Dense path size: " << dense_result.size());
                    bool _succ = false;
                    do
                    {
                        if (!mc_rrtsers[idx]->plan(start, end, dense_result, front_paths[idx]) || front_paths[idx].empty())
                        {
                            _succ = false;
                            PRINT_RED("[Thread] ID: " << idx <<"MCRRT fail.");
                            break;
                        }
                        // print size of front_paths[idx]
                        // PRINT_GREEN("[Thread] ID: " << idx << " TIME: MCRRT planning time: " << (ros::Time::now() - start_time).toSec() * 1000.0 << " ms");
                        // PRINT_GREEN("[Thread] ID: " << idx << " Path size after MCRRT: " << front_paths[idx][0].size());
                        boost::this_thread::interruption_point();
                        Eigen::MatrixXd boundary_vel = Eigen::MatrixXd::Zero(moma_param.dof_num + 3, 2);
                        Eigen::MatrixXd boundary_acc = Eigen::MatrixXd::Zero(moma_param.dof_num + 3, 2);
                        boundary_vel.col(0) = start_v;
                        
                        _succ = 
                            this->traj_opters[idx]->optimizeTraj(front_paths[idx], boundary_vel, boundary_acc)
                            && this->traj_opters[idx]->printConstraintsSituations(traj_opters[idx]->getTraj())
                            && this->traj_opters[idx]->getTraj().is_init;
                        
                    } while(false);
                    
                    results[idx].first = _succ;
                    if(_succ) results[idx].second = this->traj_opters[idx]->getTraj();
                    // results[idx] = std::make_pair(_succ, this->traj_opters[idx]->getTraj());
                    
                    if (_succ && !rdy_flag.test_and_set()) {
                        // boost::lock_guard<boost::mutex> lock(mtx);
                        // if (!first_success.exchange(true)) {  // Atomic check-and-set
                        //     cv_first.notify_all();  // Notify all waiters
                        // }
                        promise_succ.set_value(true);
                    }
                    // MomaTraj traj = this->traj_opters[idx]->getTraj();
                    // PRINT_RED("[Thread] Successful optimization with duration: " << traj.getTotalDuration() << std::endl);
                    // promise_traj.set_value(traj);
                    // promise_traj.set_value(this->traj_opters[idx]->getTraj());
                    // optim_time = (ros::Time::now() - start_time).toSec() * 1000.0;
                    // try {
                    // } catch (boost::thread_interrupted&) {
        
                    // } catch (...) {
        
                    // }
                    ros::Time end_time = ros::Time::now();
                    PRINT_GREEN("[Thread] ID: " << idx << " Optimization time: " << (end_time - start_time).toSec() * 1000.0 << " ms");
                    {
                        PRINT_GREEN("[Threads] Thread " << completed_threads+1 << " / " << topo_select_paths.size() << " completed");
                        boost::lock_guard<boost::mutex> lock(mtx);
                        if(++completed_threads == topo_select_paths.size()){
                            if(!rdy_flag.test_and_set()) promise_succ.set_value(true);
                            PRINT_GREEN("[Threads] All threads completed");
                            cv_all.notify_all();
                        }
                    }
                    
                };
                
                PRINT_YELLOW("[Threads] Starting " << topo_select_paths.size() << " threads");
                boost::thread_group threads;
                for (size_t i = 0; i < topo_select_paths.size(); ++i)
                    threads.create_thread(std::bind(
                        worker, i, topo_select_paths[i], start, end, start_v
                    ));
                // bool optSucc = future_traj.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
                // threads.interrupt_all();
                // threads.join_all();
                
                // === wait for first successful thread ===
                future_succ.wait();
                
                bool timeout = false; // indicate early termination of threads
                {
                    boost::unique_lock<boost::mutex> lock(mtx);
                    
                    // PRINT_RED("[Threads] Waiting for First Successful Optimization");
                    // cv_first.wait(lock, [&]() { return first_success.load(); } );
                    // while (!first_success) {
                    //     cv_first.wait(lock);
                
                    // === wait additional 1000ms for other threads to finish ===
                    while (completed_threads < topo_select_paths.size()){
                        PRINT_RED("[Threads] Waiting for All Threads");
                        if(timeout = 
                            boost::cv_status::timeout == cv_all.wait_for(lock, boost::chrono::milliseconds(10000))
                        )
                            break;
                    }
                }
                threads.interrupt_all();
                threads.join_all();
                
                if(timeout) {
                    PRINT_YELLOW("[Threads] Timeout in waiting threads");
                    PRINT_YELLOW("[Threads] " << completed_threads << " / " << topo_select_paths.size() << " completed");
                }
                
                for(auto &res : results) succ = succ || res.first;
            }
            _critical = true;
            if (!succ) PRINT_YELLOW("Non-Critical optimization failed, try critical optimization");
        } while(!succ && !_critical);
        
        // ros::Time t1 = ros::Time::now();
        // vector<thread> optimize_threads;
        // parallel_ends.clear();
        // parallel_ends.resize(topo_select_paths.size());
        // for (size_t i = 0; i < topo_select_paths.size(); ++i) 
        //     optimize_threads.emplace_back(&Planner::optMomaOnce, this, topo_select_paths[i], i, start, end, start_v);
        // for (size_t i = 0; i < topo_select_paths.size(); ++i) optimize_threads[i].join();
        // optim_time = (ros::Time::now() - t1).toSec() * 1000.0;
        // PRINT_GREEN("[MOMA] End optimization");
        // bool ompl_succ = false; // flag for OMPL optimization success
        // if (!succ) {
        // do {
        //     PRINT_GREEN("[planner]: OMPL optimization start!");
        //     topo_select_paths.clear();
        //     auto ompl_path = planOmpls(start, end, start_v);
        //     if (ompl_path.empty()) break; // OMPL failed
        //     Eigen::MatrixXd boundary_vel = Eigen::MatrixXd::Zero(moma_param.dof_num + 3, 2);
        //     Eigen::MatrixXd boundary_acc = Eigen::MatrixXd::Zero(moma_param.dof_num + 3, 2);
        //     boundary_vel.col(0) = start_v;
        //     if (!this->traj_opters[0]->optimizeTraj(ompl_path, boundary_vel, boundary_acc)
        //         || !this->traj_opters[0]->printConstraintsSituations(traj_opters[0]->getTraj())
        //         || !this->traj_opters[0]->getTraj().is_init
        //     ) break; // OMPL optimization failed
        //     end_traj = traj_opters[0]->getTraj();
        //     succ = true;
        //     results.resize(1);
        //     results[0].first = true;
        //     results[0].second = end_traj;
        //     ompl_succ = true;
        //     succ = true;
        //     PRINT_GREEN("[planner]: OMPL optimization success!");
        // } while (false);
        // }

        if (succ)
        {
            PRINT_GREEN("[planner]: successful optimization!");

            int shortest_idx = -1;
            int idx = -1;
            for(auto &res : results) {
                idx++;
                if(!res.first) continue;
                if(shortest_idx == -1) shortest_idx = idx;
                double traj_duration = res.second.getTotalDuration();
                if(traj_duration < results[shortest_idx].second.getTotalDuration())
                    shortest_idx = idx;
            }
            PRINT_YELLOW("Shortest path index: " << shortest_idx+1);
            ret_traj = end_traj = results[shortest_idx].second;
            vis_prm = topo_select_paths[shortest_idx];
            vis_prm_color = colors[shortest_idx];

            // PRINT_YELLOW("Publishing MeshTraj");
            // auto mesh_traj = toMeshMsg(end_traj);
            // mesh_traj_pub.publish(mesh_traj);
            last_replan_time = ros::Time::now();

            vis_ee_traj(end_traj, plot_traj_ee, {185.0/255.0, 198.0/255.0, 232.0/255.0, 0.5});
            // 
            // int res_idx = -1;
            // for (auto &res : results) {
            //     res_idx++;
            //     if(!res.first) continue;
                
            //     bool shortest = res_idx == shortest_idx;
            //     float r,g,b,a;
            //     r = shortest? 1.0 : 0.5;
            //     g = shortest? 0.0 : 0.5;
            //     b = shortest? 0.0 : 0.5;
            //     a = shortest? 1.0 : 0.2;
            //     int nsample = shortest ? 16 : 16;
            //     MomaTraj traj = res.second;
                
            //     if(!ompl_succ) {
            //         vis_isAvailable[res_idx] = true;
            //         vis_front_paths[res_idx] = front_paths[res_idx];
            //         vis_opt_paths[res_idx]   = traj;

            //         // vis_path_mesh(traj, nsample, 
            //         //     opt_traj_pub_list[res_idx], 
            //         //     {r, g, b, a}, 
            //         //     res_idx*1000 + 800);
                    
            //         // vis_path_mesh(sparsifyPath(front_paths[res_idx], 0.5), 
            //         //     front_traj_pub_list[res_idx], 
            //         //     {0.0, 0.0, 1.0, 0.2}, 
            //         //     res_idx*1000 + 900);
            //     }
            // }
            // if(ompl_succ) {
            //     for (size_t i = 0; i < vis_isAvailable.size(); ++i) vis_isAvailable[i] = false;
            // }
            
            // end_traj = future_traj.get();
            // last_replan_time = ros::Time::now();
            // vis_whole_path(end_pub);
        }

        return std::make_pair(succ, ret_traj);
    }

    std::pair<bool, float> Planner::planMomaSequential(const Eigen::VectorXd& start, 
                                    const Eigen::VectorXd& end, 
                                    const Eigen::VectorXd& start_v)
    {
        bool succ = false;                              // flag for successful optimization
        bool _critical = false;                         // whether to use critical map
        std::vector<Eigen::Vector4d> colors;            // colors of prm paths
        std::vector<std::pair<bool, MomaTraj>> results; // storing results of optimization
        do {
            // start first trial with non-critical
            topo_select_paths.clear();
            {
                list<GraphNode::Ptr> graph;
                vector<vector<Eigen::Vector3d>> raw_paths, filtered_paths;
                Eigen::Vector3d topo_start(start(0), start(1), 0.0);
                Eigen::Vector3d topo_end(end(0), end(1), 0.0);
                std::vector<Eigen::Vector3d> start_pts, end_pts;
                start_pts.push_back(topo_start);
                end_pts.push_back(topo_end);
                topo_prm->findTopoPaths(topo_start, topo_end, start_pts, end_pts, graph,
                                       raw_paths, filtered_paths, topo_select_paths, _critical);
                if (!_critical) { // for the first trial, use JPS as backup
                    auto jps_result = graph_search->plan2dJPS(start.head(2), end.head(2), moma_param.chassis_colli_radius+0.1);
                    if (!jps_result.empty())
                    {
                        std::vector<Eigen::Vector3d> jps3_res;
                        for (size_t i = 0; i < jps_result.size(); ++i)
                            jps3_res.push_back(Eigen::Vector3d(jps_result[i].x(), jps_result[i].y(), 0.0));
                        topo_select_paths.push_back(jps3_res);
                    }
                }
                
                if (topo_select_paths.empty())
                    return std::make_pair(false, 0.0);
        
                if(topo_select_paths.size() > traj_opters.size()) throw std::runtime_error("Too many paths to optimize");
                
                colors = vis_prm_paths();
        
                results.resize(topo_select_paths.size());
                for (auto res : results) { res.first = false; }
                
                auto worker = [this, &results] (
                    int idx, 
                    std::vector<Eigen::Vector3d>& topo_path, 
                    const Eigen::VectorXd& start, 
                    const Eigen::VectorXd& end, 
                    const Eigen::VectorXd& start_v)
                {
                    ros::Time start_time = ros::Time::now();
                    std::vector<Eigen::Vector2d> in_path;
                    for(auto &wp : topo_path)
                        in_path.push_back(wp.head(2));
                    auto dense_result = graph_search->getDensePath(in_path, 1.414, start(2), end(2), 
                        moma_param.max_v, moma_param.max_w);
                        
                    std::vector<Eigen::VectorXd> full_path;
                    if (!mc_rrtsers[idx]->plan(start, end, dense_result, full_path) || full_path.empty())
                    {
                        PRINT_RED("MCRRT fail.");
                        return;
                    }
                    
                    Eigen::MatrixXd boundary_vel = Eigen::MatrixXd::Zero(10, 2);
                    Eigen::MatrixXd boundary_acc = Eigen::MatrixXd::Zero(10, 2);
                    boundary_vel.col(0) = start_v;

                    bool _succ = 
                        this->traj_opters[idx]->optimizeTraj(full_path, boundary_vel, boundary_acc)
                        && this->traj_opters[idx]->printConstraintsSituations(traj_opters[idx]->getTraj())
                        && this->traj_opters[idx]->getTraj().is_init;
                        
                    results[idx] = std::make_pair(_succ, this->traj_opters[idx]->getTraj());
                };
                
                // PRINT_YELLOW("[Threads] Starting " << topo_select_paths.size() << " threads");
                // boost::thread_group threads;
                // for (size_t i = 0; i < topo_select_paths.size(); ++i)
                //     threads.create_thread(std::bind(
                //         worker, i, topo_select_paths[i], start, end, start_v
                //     ));
                // bool optSucc = future_traj.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
                // threads.interrupt_all();
                // threads.join_all();

                for (size_t i = 0; i < topo_select_paths.size(); ++i)
                    worker(i, topo_select_paths[i], start, end, start_v);
                
                int n_succ = 0;
                for(auto &res : results) {
                    if(res.first) n_succ++;
                    succ = succ || res.first;
                }
                PRINT_YELLOW("[Threads] " << n_succ << " / " << topo_select_paths.size() << " succeed");
            }
            _critical = true;
            if (!succ) PRINT_YELLOW("Non-Critical optimization failed, try critical optimization");
        } while(!succ && !_critical);

        if (!succ) {
        do {
            auto ompl_path = planOmpls(start, end, start_v);
            if (ompl_path.empty()) break; // OMPL failed
            Eigen::MatrixXd boundary_vel = Eigen::MatrixXd::Zero(10, 2);
            Eigen::MatrixXd boundary_acc = Eigen::MatrixXd::Zero(10, 2);
            boundary_vel.col(0) = start_v;
            if (!this->traj_opters[0]->optimizeTraj(ompl_path, boundary_vel, boundary_acc)
                || !this->traj_opters[0]->printConstraintsSituations(traj_opters[0]->getTraj())
                || !this->traj_opters[0]->getTraj().is_init
            ) break; // OMPL optimization failed
            end_traj = traj_opters[0]->getTraj();
            succ = true;
            results.resize(1);
            results[0].first = true;
            results[0].second = end_traj;
        } while (false);
        }

        if (succ)
        {
            PRINT_GREEN("[planner]: First successful optimization!");

            int shortest_idx = -1;
            int idx = -1;
            for(auto &res : results) {
                idx++;
                if(!res.first) continue;
                if(shortest_idx == -1) shortest_idx = idx;
                double traj_duration = res.second.getTotalDuration();
                if(traj_duration < results[shortest_idx].second.getTotalDuration())
                    shortest_idx = idx;
            }
            PRINT_YELLOW("Shortest path index: " << shortest_idx);
            end_traj = results[shortest_idx].second;

            int res_idx = -1;
            for (auto &res : results) {
                res_idx++;
                if(!res.first) continue;
                
                bool shortest = res_idx == shortest_idx;
                float r,g,b,a;
                r = shortest? 1.0 : 0.5;
                g = shortest? 0.0 : 0.5;
                b = shortest? 0.0 : 0.5;
                a = shortest? 1.0 : 0.2;
                int nsample = shortest ? 16 : 16;
                
                MomaTraj traj = res.second;
                vis_path_mesh(traj, nsample, 
                    opt_traj_pub_list[res_idx], 
                    {r, g, b, a}, 
                    res_idx*1000 + 800);
            }
            
            // end_traj = future_traj.get();
            // last_replan_time = ros::Time::now();
            // vis_whole_path(end_pub);
        }

        return std::make_pair(succ, 0.0);
    }

    
    std::pair<bool, float> Planner::planMomaNonTOPO(const Eigen::VectorXd& start, 
                                    const Eigen::VectorXd& end, 
                                    const Eigen::VectorXd& start_v)
    {
        bool succ = false;                              // flag for successful optimization
        bool _critical = false;                         // whether to use critical map
        std::vector<Eigen::Vector4d> colors;            // colors of prm paths
        std::vector<std::pair<bool, MomaTraj>> results; // storing results of optimization
        do {
            // start first trial with non-critical
            topo_select_paths.clear();
            {
                // list<GraphNode::Ptr> graph;
                // vector<vector<Eigen::Vector3d>> raw_paths, filtered_paths;
                // Eigen::Vector3d topo_start(start(0), start(1), 0.0);
                // Eigen::Vector3d topo_end(end(0), end(1), 0.0);
                // std::vector<Eigen::Vector3d> start_pts, end_pts;
                // start_pts.push_back(topo_start);
                // end_pts.push_back(topo_end);
                // topo_prm->findTopoPaths(topo_start, topo_end, start_pts, end_pts, graph,
                //                        raw_paths, filtered_paths, topo_select_paths, _critical);
                if (!_critical) {
                    auto jps_result = graph_search->plan2dJPS(start.head(2), end.head(2), moma_param.chassis_colli_radius+0.1);
                    if (!jps_result.empty())
                    {
                        std::vector<Eigen::Vector3d> jps3_res;
                        for (size_t i = 0; i < jps_result.size(); ++i)
                            jps3_res.push_back(Eigen::Vector3d(jps_result[i].x(), jps_result[i].y(), 0.0));
                        topo_select_paths.push_back(jps3_res);
                    }
                }
                
                if (topo_select_paths.empty())
                    return std::make_pair(false, 0.0);
        
                if(topo_select_paths.size() > traj_opters.size()) throw std::runtime_error("Too many paths to optimize");
                
                colors = vis_prm_paths();
        
                PRINT_GREEN("[MOMA] Start optimization");
        
                results.resize(topo_select_paths.size());
                for (auto res : results) { res.first = false; }
                
                std::promise<bool> promise_succ;
                auto future_succ = promise_succ.get_future();

                boost::mutex mtx;
                boost::condition_variable cv_first; // for the first successful thread
                boost::condition_variable cv_all;   // for all threads to finish
                std::atomic_flag rdy_flag = ATOMIC_FLAG_INIT;
                std::atomic<int> completed_threads{0};
                auto worker = [this, &results, &mtx, &promise_succ, &cv_first, &cv_all, &completed_threads, &rdy_flag] (
                    int idx, 
                    std::vector<Eigen::Vector3d>& topo_path, 
                    const Eigen::VectorXd& start, 
                    const Eigen::VectorXd& end, 
                    const Eigen::VectorXd& start_v)
                {
                    ros::Time start_time = ros::Time::now();
                    std::vector<Eigen::Vector2d> in_path;
                    for(auto &wp : topo_path)
                        in_path.push_back(wp.head(2));
                    auto dense_result = graph_search->getDensePath(in_path, 1.414, start(2), end(2), 
                        moma_param.max_v, moma_param.max_w);
                        
                    boost::this_thread::interruption_point();
                    std::vector<Eigen::VectorXd> full_path;

                    bool _succ = false;
                    do
                    {
                        if (!mc_rrtsers[idx]->plan(start, end, dense_result, full_path) || full_path.empty())
                        {
                            _succ = false;
                            PRINT_RED("MCRRT fail.");
                            break;
                        }
                        boost::this_thread::interruption_point();
                        Eigen::MatrixXd boundary_vel = Eigen::MatrixXd::Zero(10, 2);
                        Eigen::MatrixXd boundary_acc = Eigen::MatrixXd::Zero(10, 2);
                        boundary_vel.col(0) = start_v;
    
                        _succ = 
                            this->traj_opters[idx]->optimizeTraj(full_path, boundary_vel, boundary_acc)
                            && this->traj_opters[idx]->printConstraintsSituations(traj_opters[idx]->getTraj())
                            && this->traj_opters[idx]->getTraj().is_init;
                        
                    } while(false);
                    
                    results[idx].first = _succ;
                    if(_succ) results[idx].second = this->traj_opters[idx]->getTraj();
                    
                    if (_succ && !rdy_flag.test_and_set())
                        promise_succ.set_value(true);
                    
                    ros::Time end_time = ros::Time::now();
                    PRINT_GREEN("[Thread] ID: " << idx << " Optimization time: " << (end_time - start_time).toSec() * 1000.0 << " ms");
                    {
                        PRINT_GREEN("[Threads] Thread " << completed_threads+1 << " / " << topo_select_paths.size() << " completed");
                        boost::lock_guard<boost::mutex> lock(mtx);
                        if(++completed_threads == topo_select_paths.size()){
                            if(!rdy_flag.test_and_set()) promise_succ.set_value(true);
                            PRINT_GREEN("[Threads] All threads completed");
                            cv_all.notify_all();
                        }
                    }
                    
                };
                
                PRINT_YELLOW("[Threads] Starting " << topo_select_paths.size() << " threads");
                boost::thread_group threads;
                for (size_t i = 0; i < topo_select_paths.size(); ++i)
                    threads.create_thread(std::bind(
                        worker, i, topo_select_paths[i], start, end, start_v
                    ));
                
                // === wait for first successful thread ===
                future_succ.wait();
                
                bool timeout; // indicate early termination of threads
                {
                    boost::unique_lock<boost::mutex> lock(mtx);
                    
                    PRINT_RED("[Threads] Waiting for First Successful Optimization");
                
                    // === wait additional 100ms for other threads to finish ===
                    while (completed_threads < topo_select_paths.size()){
                        PRINT_RED("[Threads] Waiting for All Threads");
                        if(timeout = 
                            boost::cv_status::timeout == cv_all.wait_for(lock, boost::chrono::milliseconds(100))
                        )
                            break;
                    }
                }
                threads.interrupt_all();
                threads.join_all();
                
                if(timeout) {
                    PRINT_YELLOW("[Threads] Timeout in waiting threads");
                    PRINT_YELLOW("[Threads] " << completed_threads << " / " << topo_select_paths.size() << " completed");
                }
                
                for(auto &res : results) succ = succ || res.first;
            }
            _critical = true;
            if (!succ) PRINT_YELLOW("Non-Critical optimization failed, try critical optimization");
        } while(false);
        
        if (!succ) {
        do {
            auto ompl_path = planOmpls(start, end, start_v);
            if (ompl_path.empty()) break; // OMPL failed
            Eigen::MatrixXd boundary_vel = Eigen::MatrixXd::Zero(10, 2);
            Eigen::MatrixXd boundary_acc = Eigen::MatrixXd::Zero(10, 2);
            boundary_vel.col(0) = start_v;
            if (!this->traj_opters[0]->optimizeTraj(ompl_path, boundary_vel, boundary_acc)
                || !this->traj_opters[0]->printConstraintsSituations(traj_opters[0]->getTraj())
                || !this->traj_opters[0]->getTraj().is_init
            ) break; // OMPL optimization failed
            end_traj = traj_opters[0]->getTraj();
            succ = true;
            results.resize(1);
            results[0].first = true;
            results[0].second = end_traj;
        } while (false);
        }

        if (succ)
        {
            PRINT_GREEN("[planner]: First successful optimization!");

            int shortest_idx = -1;
            int idx = -1;
            for(auto &res : results) {
                idx++;
                if(!res.first) continue;
                if(shortest_idx == -1) shortest_idx = idx;
                double traj_duration = res.second.getTotalDuration();
                if(traj_duration < results[shortest_idx].second.getTotalDuration())
                    shortest_idx = idx;
            }
            PRINT_YELLOW("Shortest path index: " << shortest_idx);
            end_traj = results[shortest_idx].second;

            int res_idx = -1;
            for (auto &res : results) {
                res_idx++;
                if(!res.first) continue;
                
                bool shortest = res_idx == shortest_idx;
                float r,g,b,a;
                r = shortest? 1.0 : 0.5;
                g = shortest? 0.0 : 0.5;
                b = shortest? 0.0 : 0.5;
                a = shortest? 1.0 : 0.2;
                int nsample = shortest ? 16 : 16;
                
                MomaTraj traj = res.second;
                vis_path_mesh(traj, nsample, 
                    opt_traj_pub_list[res_idx], 
                    {r, g, b, a}, 
                    res_idx*1000 + 800);
            }
        }
        return std::make_pair(succ, 0.0);
    }

    bool Planner::optMomaOnce(const std::vector<Eigen::Vector3d>& topo_path, int idx, 
                            const Eigen::VectorXd& start, const Eigen::VectorXd& end,
                            const Eigen::VectorXd& start_v)
    {
        std::vector<Eigen::Vector2d> in_path;
        for (size_t i = 0; i < topo_path.size(); ++i)
            in_path.push_back(topo_path[i].head(2));
        auto dense_result = graph_search->getDensePath(in_path, 1.414, start(2), end(2), 
                                                       moma_param.max_v, moma_param.max_w);
        std::vector<Eigen::VectorXd> full_path;
        if (!mc_rrtsers[idx]->plan(start, end, dense_result, full_path))
        {
            PRINT_RED("MCRRTs fail, idx = "<<idx);
            return false;
        }
        if (full_path.empty())
            return false;
        Eigen::MatrixXd boundary_vel = Eigen::MatrixXd::Zero(3+moma_param.dof_num, 2);
        Eigen::MatrixXd boundary_acc = Eigen::MatrixXd::Zero(3+moma_param.dof_num, 2);
        boundary_vel.col(0) = start_v;
        if (!traj_opters[idx]->optimizeTraj(full_path, boundary_vel, boundary_acc)
            || !traj_opters[idx]->printConstraintsSituations(traj_opters[idx]->getTraj()) 
            )
            return false;
        else
        {
            parallel_ends[idx] = traj_opters[idx]->getTraj();
            return true;
        }
        return true;
    }

    bool Planner::optDenseOnce(const std::vector<Eigen::VectorXd>& full_path, int idx, 
                            const Eigen::VectorXd& start, const Eigen::VectorXd& end,
                            const Eigen::VectorXd& start_v)
    {
        Eigen::MatrixXd boundary_vel = Eigen::MatrixXd::Zero(3+moma_param.dof_num, 2);
        Eigen::MatrixXd boundary_acc = Eigen::MatrixXd::Zero(3+moma_param.dof_num, 2);
        boundary_vel.col(0) = start_v;
        if (!traj_opters[idx]->optimizeTraj(full_path, boundary_vel, boundary_acc)
            || !traj_opters[idx]->printConstraintsSituations(traj_opters[idx]->getTraj()) 
            )
            return false;
        else
        {
            parallel_ends[idx] = traj_opters[idx]->getTraj();
            return true;
        }
        return true;
    }


    std::vector<Eigen::Vector4d> Planner::vis_prm_paths()
    {
        visualization_msgs::MarkerArray markers;
        visualization_msgs::Marker line_strip, delet_p;

        delet_p.action = visualization_msgs::Marker::DELETEALL;
        delet_p.id = 0;
        markers.markers.push_back(delet_p);

        line_strip.type = visualization_msgs::Marker::LINE_STRIP;
        line_strip.header.frame_id = "world";
        line_strip.pose.orientation.w = 1.0;
        line_strip.scale.x = 0.10;
        line_strip.scale.y = 0.10;
        line_strip.scale.z = 0.10;
        line_strip.color.a = 1.0;

        std::vector<Eigen::Vector4d> colors;

        for (size_t i=0; i<topo_select_paths.size(); i++)
        {
            Eigen::Vector4d color = {
                1.0 * (rand() % 1000) / 1000.0, 
                1.0 * (rand() % 1000) / 1000.0, 
                1.0 * (rand() % 1000) / 1000.0, 
                1.0};
            colors.push_back(color);
            line_strip.header.stamp = ros::Time::now();
            line_strip.id = i + 1;
            line_strip.color.r = color[0];
            line_strip.color.g = color[1];
            line_strip.color.b = color[2];
            line_strip.points.clear();
            for (size_t j=0; j<topo_select_paths[i].size(); j++)
            {
                geometry_msgs::Point pt;
                pt.x = topo_select_paths[i][j].x();
                pt.y = topo_select_paths[i][j].y();
                pt.z = 0.0;
                line_strip.points.push_back(pt);
            }
            markers.markers.push_back(line_strip);
        }
        prm_pub.publish(markers);
        return colors;
    }

    void Planner::vis_path(const std::vector<Eigen::VectorXd>& path, ros::Publisher& puber, vector<float> rgba, vector<int> ids)
    {
        if (path.empty())
            return;

        visualization_msgs::Marker line_strip, arrow, text;
        arrow.header.frame_id = line_strip.header.frame_id = text.header.frame_id = "world";
        arrow.header.stamp = line_strip.header.stamp = text.header.stamp = ros::Time::now();
        line_strip.type = visualization_msgs::Marker::LINE_STRIP;
        arrow.type = visualization_msgs::Marker::ARROW;
        text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text.id = 1886666;
        text.action = visualization_msgs::Marker::ADD;
        text.scale.z = 0.1;
        text.color.a = 1.0;
        line_strip.id = 10086;
        arrow.scale.x = 0.03;
        arrow.scale.y = 0.05;
        arrow.color.a = 1.0;
        arrow.color.r = 1.0;
        arrow.color.g = 0.0;
        arrow.color.b = 0.0;
        arrow.pose.orientation.w = 1.0;
        line_strip.pose.orientation.w = 1.0;
        line_strip.scale.x = 0.03;
        line_strip.scale.y = 0.03;
        line_strip.scale.z = 0.03;
        line_strip.color.a = 1.0;
        line_strip.color.r = 0.0;
        line_strip.color.g = 1.0;
        line_strip.color.b = 0.0;

        visualization_msgs::MarkerArray array_msg;
        visualization_msgs::Marker p;
        p.action = visualization_msgs::Marker::DELETEALL;
        p.id = 0;
        array_msg.markers.push_back(p);
        for (size_t i=0; i<path.size(); i++)
        {
            visualization_msgs::MarkerArray node_array = moma_param.getColliCylinderArray(path[i]);
            size_t array_size = node_array.markers.size();
            for (size_t j=0; j<array_size; j++)
            {
                node_array.markers[j].id = i*array_size+j;
                node_array.markers[j].color.a = rgba[3];
                node_array.markers[j].color.r = rgba[0];
                node_array.markers[j].color.g = rgba[1];
                node_array.markers[j].color.b = rgba[2];
                array_msg.markers.push_back(node_array.markers[j]);
            }
            geometry_msgs::Point pt;
            pt.x = path[i].x();
            pt.y = path[i].y();
            pt.z = 0.0;
            line_strip.points.push_back(pt);
            geometry_msgs::Point pt_arrow;
            pt_arrow.x = path[i].x() + moma_param.chassis_colli_radius*cos(path[i].z());
            pt_arrow.y = path[i].y() + moma_param.chassis_colli_radius*sin(path[i].z());
            arrow.points.clear();
            arrow.points.push_back(pt);
            arrow.points.push_back(pt_arrow);
            arrow.id = line_strip.id + i + 1;
            array_msg.markers.push_back(arrow);
            text.color.r = 0.0;
            arrow.color.b = 0.0;
            for (size_t j=0; j<ids.size(); j++)
            {
                if (ids[j] == (int)i)
                {
                    text.color.r = 1.0;
                    arrow.color.b = 1.0;
                    break;
                }
            }
            text.text = std::to_string(i);
            text.id = text.id + 1;
            text.pose.orientation.w = 1.0;
            text.pose.position = node_array.markers.back().pose.position;
            text.pose.position.z = text.pose.position.z + 0.1;
            array_msg.markers.push_back(text);
        }
        array_msg.markers.push_back(line_strip);
        puber.publish(array_msg);
        return;
    }

    void Planner::vis_whole_path(ros::Publisher& pub)
    {
        std::vector<Eigen::VectorXd> end_path;
        nav_msgs::Path car_traj;
        for (double t=0.0; t<end_traj.getTotalDuration(); t+=0.1)
        {
            Eigen::VectorXd state = end_traj.getState(t);
            end_path.push_back(state);
            car_traj.header.frame_id = "world";
            car_traj.header.stamp = ros::Time::now();
            geometry_msgs::PoseStamped gp;
            gp.pose.position.x = state.x();
            gp.pose.position.y = state.y();
            gp.pose.position.z = 0.0;
            gp.pose.orientation.w = cos(state.z()/2.0);
            gp.pose.orientation.x = 0.0;
            gp.pose.orientation.y = 0.0;
            gp.pose.orientation.z = sin(state.z()/2.0);
            car_traj.poses.push_back(gp);
        }
        if (pub == end_pub)
            car_traj_pub.publish(car_traj);
        vis_path(end_path, pub, {1.0, 0.0, 0.0, 0.15});
    }

    Eigen::Quaterniond euler2rotation(double r, double p, double y)
    {
        return Eigen::AngleAxisd(y, Eigen::Vector3d::UnitZ()) 
                * Eigen::AngleAxisd(p, Eigen::Vector3d::UnitY()) 
                * Eigen::AngleAxisd(r, Eigen::Vector3d::UnitX());
    }

    Eigen::Quaterniond euler2rotation(Eigen::Vector3d rpy)
    {
        return Eigen::AngleAxisd(rpy(2), Eigen::Vector3d::UnitZ()) 
        * Eigen::AngleAxisd(rpy(1), Eigen::Vector3d::UnitY()) 
        * Eigen::AngleAxisd(rpy(0), Eigen::Vector3d::UnitX());
    }

    void Planner::vis_path_mesh(const std::vector<Eigen::VectorXd>& path,
                                ros::Publisher& pub,
                                std::vector<float> rgba,
                                int id)
    {
        // {
        //     visualization_msgs::MarkerArray clear_arr;
        //     visualization_msgs::Marker delet_p;
        //     delet_p.action = visualization_msgs::Marker::DELETEALL;
        //     delet_p.id = 9871;
        //     clear_arr.markers.push_back(delet_p);
        //     pub.publish(clear_arr);
        // }

        if (path.empty())
            return;

        visualization_msgs::MarkerArray moma_marker;
        for (const auto& moma_pos : path)
        {
            std::vector<Vector3d> joint_pos, joint_axis;
            std::vector<Vector4d> sphere_pos_radius;
            std::vector<Vector3d> p_all_link;
            std::vector<Matrix3d> R_all_link;
            moma_param.computeKinematics(
                moma_pos,
                joint_pos,
                joint_axis,
                sphere_pos_radius,
                &p_all_link,
                &R_all_link
            );

            if (p_all_link.empty() || R_all_link.empty())
                continue;

            // chassis ===
            {
                const Vector3d& p_base = p_all_link[0];
                const Matrix3d& R_base = R_all_link[0];
                Eigen::Quaterniond q_base(R_base);

                visualization_msgs::Marker base_marker;
                base_marker.header.frame_id = "world";
                base_marker.id   = id++;
                base_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
                base_marker.action = visualization_msgs::Marker::ADD;
                base_marker.mesh_resource = moma_param.chassis_mesh;

                base_marker.pose.position.x = p_base.x();
                base_marker.pose.position.y = p_base.y();
                base_marker.pose.position.z = p_base.z();
                base_marker.pose.orientation.w = q_base.w();
                base_marker.pose.orientation.x = q_base.x();
                base_marker.pose.orientation.y = q_base.y();
                base_marker.pose.orientation.z = q_base.z();

                base_marker.color.a = rgba[3];
                base_marker.color.r = rgba[0];
                base_marker.color.g = rgba[1];
                base_marker.color.b = rgba[2];

                base_marker.scale.x = 1.0;
                base_marker.scale.y = 1.0;
                base_marker.scale.z = 1.0;

                moma_marker.markers.push_back(base_marker);
            }

            // arm links

            const size_t n_frames = p_all_link.size(); // = dof_num + 2
            for (size_t frame_id = 1; frame_id < n_frames; ++frame_id)
            {
                const Vector3d& pL = p_all_link[frame_id];
                const Matrix3d& RL = R_all_link[frame_id];
                Eigen::Quaterniond qL(RL);

                visualization_msgs::Marker link_marker;
                link_marker.header.frame_id = "world";
                link_marker.id   = id++;
                link_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
                link_marker.action = visualization_msgs::Marker::ADD;
                int link_idx = static_cast<int>(frame_id);
                link_marker.mesh_resource = moma_param.link_meshes[link_idx - 1];

                link_marker.pose.position.x = pL.x();
                link_marker.pose.position.y = pL.y();
                link_marker.pose.position.z = pL.z();
                link_marker.pose.orientation.w = qL.w();
                link_marker.pose.orientation.x = qL.x();
                link_marker.pose.orientation.y = qL.y();
                link_marker.pose.orientation.z = qL.z();

                link_marker.color.a = rgba[3];
                link_marker.color.r = rgba[0];
                link_marker.color.g = rgba[1];
                link_marker.color.b = rgba[2];

                link_marker.scale.x = 1.0;
                link_marker.scale.y = 1.0;
                link_marker.scale.z = 1.0;

                moma_marker.markers.push_back(link_marker);
            }
        }

        if (!moma_marker.markers.empty())
            pub.publish(moma_marker);
    }



    void Planner::vis_path_mesh(const MomaTraj& traj, int nsample, ros::Publisher& pub, vector<float> rgba, int id) {
        RowMatrixXd path_m = traj.sampleTimePoints(nsample);
        std::vector<Eigen::VectorXd> path;

        Eigen::VectorXd prev_state = path_m.row(0).head(moma_param.dof_num + 3);
        Eigen::VectorXd end_state = path_m.row(path_m.rows()-1).head(moma_param.dof_num + 3);
        path.push_back(prev_state);

        for (int i = 0; i < path_m.rows(); i++) {
            Eigen::VectorXd state = path_m.row(i).head(moma_param.dof_num + 3);
            // if ((state.head(2) - prev_state.head(2)).norm() > 0.5
            // && (state.head(2) - end_state.head(2)).norm() > 0.5) {
            //     path.push_back(state);
            //     prev_state = state;
            // }
            path.push_back(state);
            // path.push_back(path_m.row(i).head(10));
        }
        path.push_back(end_state);

        vis_path_mesh(path, pub, rgba, id);
    }

    Eigen::VectorXd Planner::generateRandomState(void) const {
        Eigen::VectorXd ret;
        ret.resize(3+moma_param.dof_num);
        // Eigen::Vector3d min_bound = grid_map->min_boundary;
        // Eigen::Vector3d max_bound = grid_map->max_boundary;

        // uniform_real_distribution<double> rand_x(min_bound(0)+2.0, max_bound(0)-2.0);
        // uniform_real_distribution<double> rand_y(min_bound(1)+2.0, max_bound(1)-2.0);
        // uniform_real_distribution<double> rand_theta(-M_PI, M_PI);
        // uniform_real_distribution<double> rand_q(0.0, 1.0);
        
        // eng = default_random_engine(42);
        // ret(0) = rand_x(eng);
        // ret(1) = rand_y(eng);
        // ret(2) = rand_theta(eng);
        // for (int i = 0; i < moma_param.dof_num; i++)
        //     ret(3+i) = (moma_param.joint_pos_limit_max(i) - moma_param.joint_pos_limit_min(i)) * rand_q(eng)
        //             + moma_param.joint_pos_limit_min(i);

        return ret;
    }

    std::vector<Eigen::VectorXd> Planner::sparsifyPath(const std::vector<Eigen::VectorXd>& path, double dist) const {
        std::vector<Eigen::VectorXd> ret;
        if (path.empty()) return ret;

        Eigen::VectorXd end_state = path.back();
        
        ret.push_back(path.front());
        for (size_t i = 1; i < path.size(); i++) {
            if ((path[i].head(2) - ret.back().head(2)).norm() > dist
                && (path[i].head(2) - end_state.head(2)).norm() > dist) {
                ret.push_back(path[i]);
            }
        }
        ret.push_back(end_state);
        return ret;
    }

    void Planner::timerCallback (const ros::TimerEvent& event) {

        bool vis_prm_available = false;
        for (size_t i = 0; i < vis_isAvailable.size(); i++) {
            vis_prm_available = vis_prm_available || vis_isAvailable[i];
            if (vis_isAvailable[i]) {
                vis_path_mesh(vis_opt_paths[i], 16, opt_traj_pub_list[i],   {226.0/255.0, 145.0/255.0, 53.0/255.0, 0.5}, i*800);
                vis_path_mesh(sparsifyPath(vis_front_paths[i], 0.5),   front_traj_pub_list[i], {0.0, 0.0, 1.0, 0.3}, i*900);
            } else {
                vis_path_mesh(std::vector<Eigen::VectorXd>(), opt_traj_pub_list[i], {1.0, 0.0, 0.0, 0.15}, i*800);
                vis_path_mesh(std::vector<Eigen::VectorXd>(), front_traj_pub_list[i], {1.0, 0.0, 0.0, 0.15}, i*900);
            }
        }

        if (vis_prm_available) {
            visualization_msgs::MarkerArray markers;
            visualization_msgs::Marker line_strip, delet_p;

            delet_p.action = visualization_msgs::Marker::DELETEALL;
            delet_p.id = 0;
            markers.markers.push_back(delet_p);

            line_strip.type = visualization_msgs::Marker::LINE_STRIP;
            line_strip.header.frame_id = "world";
            line_strip.pose.orientation.w = 1.0;
            line_strip.scale.x = 0.10;
            line_strip.scale.y = 0.10;
            line_strip.scale.z = 0.10;
            line_strip.color.a = 1.0;

            line_strip.header.stamp = ros::Time::now();
            line_strip.id = 1;
            line_strip.color.r = vis_prm_color[0];
            line_strip.color.g = vis_prm_color[1];
            line_strip.color.b = vis_prm_color[2];
            line_strip.points.clear();
            for (size_t j=0; j<vis_prm.size(); j++)
            {
                geometry_msgs::Point pt;
                pt.x = vis_prm[j].x();
                pt.y = vis_prm[j].y();
                pt.z = 0.0;
                line_strip.points.push_back(pt);
            }
            markers.markers.push_back(line_strip);
            vis_prm_pub.publish(markers);
        }
    }

    void Planner::vis_ee_traj(const std::vector<Eigen::VectorXd>& path, ros::Publisher& pub, vector<float> rgba) const {
        if (path.empty())
            return;

        visualization_msgs::Marker line_strip;
        {
            line_strip.header.frame_id = "world";
            line_strip.header.stamp = ros::Time::now();
            line_strip.ns = "velocity_trajectory";
            line_strip.action = visualization_msgs::Marker::ADD;
            line_strip.pose.orientation.w = 1.0;
            // use a unique id so previously published ee trajectories are not overwritten
            line_strip.id = this->traj_vis_counter.fetch_add(1) + 2077;
            line_strip.type = visualization_msgs::Marker::LINE_STRIP;
            line_strip.scale.x = 0.10;
            line_strip.scale.y = 0.10;
            line_strip.scale.z = 0.10;
        }

        Eigen::Vector4d gripper_prev = moma_param.getColliPts(    
            path[0]
        ).back();

        std::vector<double> velocities;
        for (size_t i = 0; i < path.size(); i++) {
            Eigen::Vector4d gripper;
            gripper = moma_param.getColliPts(path[i]).back();

            geometry_msgs::Point pt;
            pt.x = gripper (0);
            pt.y = gripper (1);
            pt.z = gripper (2);
            line_strip.points.push_back(pt);

            if (i > 0) {
                velocities.push_back((gripper.head(3) - gripper_prev.head(3)).norm() / 0.1);
            }
            gripper_prev = gripper;
        }
        Eigen::Vector4d rgba_random = {
        1.0 * (rand() % 1000) / 1000.0, 
        1.0 * (rand() % 1000) / 1000.0, 
        1.0 * (rand() % 1000) / 1000.0, 
        1.0};
        for (size_t i = 0; i < path.size(); i++){
            std_msgs::ColorRGBA color;
            {
                color.r = rgba_random[0];
                color.g = rgba_random[1];
                color.b = rgba_random[2];
                color.a = rgba_random[3];
            }
            line_strip.colors.push_back(color);
        }
        pub.publish(line_strip);
        return;
    }







    void Planner::vis_ee_traj(const MomaTraj& traj, ros::Publisher& pub, vector<float> rgba) const {
        const int res = 200;
        const double intvl = traj.getTotalDuration() / res;
        double t = 0.0;

        visualization_msgs::Marker line_strip;
        {
            line_strip.header.frame_id = "world";
            line_strip.header.stamp = ros::Time::now();
            line_strip.ns = "velocity_trajectory";
            line_strip.action = visualization_msgs::Marker::ADD;
            line_strip.pose.orientation.w = 1.0;
            // use a unique id so previously published ee trajectories are not overwritten
            line_strip.id = this->traj_vis_counter.fetch_add(1) + 2077;
            line_strip.type = visualization_msgs::Marker::LINE_STRIP;
            line_strip.scale.x = 0.10;
            line_strip.scale.y = 0.10;
            line_strip.scale.z = 0.10;
        }

        std::vector<double> velocities;
        Eigen::Vector4d gripper_prev = moma_param.getColliPts(    
            traj.getState(0)
        ).back();

        for (size_t i = 0; i < res; i++) {
            Eigen::VectorXd state = traj.getState(t);
            Eigen::Vector4d gripper;
            gripper = moma_param.getColliPts(state).back();

            geometry_msgs::Point pt;
            pt.x = gripper (0);
            pt.y = gripper (1);
            pt.z = gripper (2);
            line_strip.points.push_back(pt);

            velocities.push_back((gripper.head(3) - gripper_prev.head(3)).norm() / intvl);
            t += intvl;
            gripper_prev = gripper;
        }
        velocities[0] = velocities[1]; // because the first velocity is not reliable

        double max_vel = *std::max_element(velocities.begin(), velocities.end());
        double min_vel = *std::min_element(velocities.begin(), velocities.end());
        // double avg_vel = std::accumulate(velocities.begin(), velocities.end(), 0.0) / velocities.size();


        std::cout << "max vel: " << max_vel << " min vel: " << min_vel << std::endl;
        Eigen::Vector4d rgba_random = {
        1.0 * (rand() % 1000) / 1000.0, 
        1.0 * (rand() % 1000) / 1000.0, 
        1.0 * (rand() % 1000) / 1000.0, 
        0.5};
        for (size_t i = 0; i < res; i++){
            double vel = velocities[i];
            double r = (vel - min_vel) / (max_vel - min_vel);

            std_msgs::ColorRGBA color;
            {
                // Viridis
                // color.r = 0.267004 + 0.031242 * r - 1.17733 * pow(r, 2) + 
                //         0.781638 * pow(r, 3) + 0.46992 * pow(r, 4);
                // color.g = 0.004874 + 1.05819 * r - 0.218094 * pow(r, 2) - 
                //         1.52621 * pow(r, 3) + 1.80664 * pow(r, 4);
                // color.b = 0.329415 - 0.197112 * r - 5.8219 * pow(r, 3) + 
                //         5.30237 * pow(r, 4);
                // color.a = 1.0;
            }

            {
                // inferno
                // r = r > 0.9 ? 0.9 : r;
                // if (r < 0.25) {
                //     color.r = 0.2 * r * 4.0;
                //     color.g = 0.0;
                //     color.b = 0.3 + 0.7 * r * 4.0;
                // } 
                // else if (r < 0.5) {
                //     color.r = 0.2 + 0.8 * (r-0.25)*4.0;
                //     color.g = 0.1 * (r-0.25)*4.0;
                //     color.b = 1.0 - 0.8 * (r-0.25)*4.0;
                // }
                // else if (r < 0.75) {
                //     color.r = 1.0;
                //     color.g = 0.1 + 0.9 * (r-0.5)*4.0;
                //     color.b = 0.2 - 0.2 * (r-0.5)*4.0;
                // }
                // else {
                //     color.r = 1.0;
                //     color.g = 1.0;
                //     color.b = 0.0 + 1.0 * (r-0.75)*4.0;
                // }
                
                // color.a = 0.5;
            }
            {
                color.r = rgba_random[0];
                color.g = rgba_random[1];
                color.b = rgba_random[2];
                color.a = rgba_random[3];
            }
            line_strip.colors.push_back(color);
        }

        pub.publish(line_strip);
    }

    planner::MeshTraj Planner::toMeshMsg(const MomaTraj& traj) const {
        // double traj_duration = traj.getTotalDuration();
        // const int res = 1000;
        // double intvl = traj_duration / res;


        planner::MeshTraj ret;
        // std::vector<planner::MeshState> states;
        // std::vector<double> arc_lengths;
        // std::vector<double> yaws;

        // double acc_arc_length = 0.0;
        // Eigen::VectorXd prev_state = traj.getState(0);
        
        // for (double t = 0.0; t < traj_duration; t += intvl) {
        //     Eigen::VectorXd state = traj.getState(t);
        //     std::vector<Eigen::VectorXd> mesh_poses = moma_param.getMeshPose(state);

        //     planner::MeshState mesh_state;
        //     std::vector<planner::MeshPart> mesh_poses_msg;

        //     for (Eigen::VectorXd mesh_pose : mesh_poses) {
        //         planner::MeshPart mesh_pose_msg;
                

        //         mesh_pose_msg.pos_x = mesh_pose(0);
        //         mesh_pose_msg.pos_y = mesh_pose(1);
        //         mesh_pose_msg.pos_z = mesh_pose(2);
        //         mesh_pose_msg.orient_w = mesh_pose(3);
        //         mesh_pose_msg.orient_x = mesh_pose(4);
        //         mesh_pose_msg.orient_y = mesh_pose(5);
        //         mesh_pose_msg.orient_z = mesh_pose(6);

        //         mesh_poses_msg.push_back(mesh_pose_msg);
        //     }


        //     mesh_state.parts = mesh_poses_msg;
        //     states.push_back(mesh_state);

        //     acc_arc_length += (state.head(2) - prev_state.head(2)).norm();

        //     arc_lengths.push_back(acc_arc_length);
        //     yaws.push_back(state(2));


        //     prev_state = state;
        // }


        // ret.states = states;
        // ret.arc_lengths = arc_lengths;
        // ret.yaws = yaws;
        return ret;
    }

    void Planner::vis_time(ros::Publisher& pub, double time, int id) const {
        visualization_msgs::MarkerArray array_msg;
        visualization_msgs::Marker p;
        p.action = visualization_msgs::Marker::DELETEALL;
        p.id = 0;
        array_msg.markers.push_back(p);

        visualization_msgs::Marker text;
        text.header.stamp = ros::Time::now();
        text.header.frame_id = "world";
        text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text.id = 1886666;
        text.action = visualization_msgs::Marker::ADD;
        text.scale.z = 1.2;
        text.color.a = 1.0;

        int rounded = std::round(time);
        // Remove extra zeros if needed

        text.text = std::to_string(rounded) + " ms";
        text.id = id;
        text.pose.orientation.w = 1.0;
        text.pose.position.x = 0.0;
        text.pose.position.y = 0.0;
        text.pose.position.z = 2.0;
        array_msg.markers.push_back(text);

        pub.publish(array_msg);
    }

}
