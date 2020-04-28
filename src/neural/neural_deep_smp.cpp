//#define DEBUG 1
#include "neural/neural_deep_smp.hpp"
#include <time.h>
MPNetSMP::MPNetSMP(std::string mlp_path, std::string encoder_path,
                   system_t* system,
                   int num_iters_in, int num_steps_in, double step_sz_in
                   )
                   : psopt_num_iters(num_iters_in)
                   , psopt_num_steps(num_steps_in)
                   , psopt_step_sz(step_sz_in)
{
    // neural network
    MLP.reset(new torch::jit::script::Module(torch::jit::load(mlp_path)));
    #ifdef DEBUG
        std::cout << "loaded MLP" << std::endl;
    #endif
    encoder.reset(new torch::jit::script::Module(torch::jit::load(encoder_path)));

    #ifdef DEBUG
        std::cout << "loaded modules" << std::endl;
    #endif

    // obtain bound from system
    state_dim = system->get_state_dimension();
    control_dim = system->get_control_dimension();
    #ifdef DEBUG
        std::cout << "loaded dims" << std::endl;
        std::cout << "state_dim:" << state_dim << std::endl;
    #endif
    for (unsigned i=0; i < state_dim; i++)
    {
        lower_bound.push_back(system->get_state_bounds()[i].first);
        upper_bound.push_back(system->get_state_bounds()[i].second);
        #ifdef DEBUG
            std::cout << "lower_bound[i]" << lower_bound[i] << std::endl;
        #endif
        bound.push_back((upper_bound[i]-lower_bound[i]) / 2);
    }
    for (unsigned i=0; i < control_dim; i++)
    {
        control_lower_bound.push_back(system->get_control_bounds()[i].first);
        control_upper_bound.push_back(system->get_control_bounds()[i].second);
    }
    #ifdef DEBUG
        std::cout << "copied bounds" << std::endl;
    #endif
    is_circular = system->is_circular_topology();
}

MPNetSMP::~MPNetSMP()
{
    MLP.reset();
    encoder.reset();
}


void MPNetSMP::normalize(const std::vector<double>& state, std::vector<double>& res)
{
    for (int i=0; i<this->state_dim; i++)
    {
        res.push_back((state[i]-lower_bound[i])/bound[i]-1.0);
    }
}
void MPNetSMP::unnormalize(const std::vector<double>& state, std::vector<double>& res)
{
    // normalize in the 3D state space first
    //std::vector<double> res;
    for (int i=0; i<this->state_dim; i++)
    {
        res.push_back((state[i]+1.0)*bound[i]+lower_bound[i]);
    }
}

torch::Tensor MPNetSMP::getStartGoalTensor(const std::vector<double>& start_state, const std::vector<double>& goal_state)
{
    std::vector<double> normalized_start_vec;
    this->normalize(start_state, normalized_start_vec);
    std::vector<double> normalized_goal_vec;
    this->normalize(goal_state, normalized_goal_vec);
    // double->float
    std::vector<float> float_normalized_start_vec;
    std::vector<float> float_normalized_goal_vec;
    for (unsigned i=0; i<this->state_dim; i++)
    {
        float_normalized_start_vec.push_back(float(normalized_start_vec[i]));
        float_normalized_goal_vec.push_back(float(normalized_goal_vec[i]));
    }
    torch::Tensor start_tensor = torch::from_blob(float_normalized_start_vec.data(), {1, this->state_dim});
    torch::Tensor goal_tensor = torch::from_blob(float_normalized_goal_vec.data(), {1, this->state_dim});

    #ifdef DEBUG
        std::cout << "Start Vec: \n" << start_state << "\n";
        std::cout << "Goal Vec: \n" << goal_state << "\n";



        std::cout << "Start Vec: " << start_state << "\n"
                << "Start Tensor: " << start_tensor << "\n"
                << "Goal Vec: " << goal_state << "\n"
                << "Goal Tensor: " << goal_tensor << "\n";
    #endif

    torch::Tensor sg_cat;
    sg_cat = torch::cat({start_tensor, goal_tensor}, 1);


    #ifdef DEBUG
        std::cout << "\n\n\nCONCATENATED START/GOAL\n\n\n" << sg_cat << "\n\n\n";
    #endif

    return sg_cat;
}


void MPNetSMP::informer(at::Tensor obs, const std::vector<double>& start_state, const std::vector<double>& goal_state, std::vector<double>& next_state)
{
    // given the start and goal, and the internal obstacle representation
    // convert them to torch::Tensor, and feed into MPNet
    // return the next state to the "next" parameter
    #ifdef DEBUG
        std::cout << "starting mpnet_predict..." << std::endl;
    #endif

    //int dim = si_->getStateDimension();
    int dim = this->state_dim;
    // get start, goal in tensor form

    torch::Tensor sg = getStartGoalTensor(start_state, goal_state);
    //torch::Tensor gs = getStartGoalTensor(goal, start, dim);

    torch::Tensor mlp_input_tensor;
    // Note the order of the cat
    mlp_input_tensor = torch::cat({obs,sg}, 1).to(at::kCUDA);
    //mlp_input_tensor = torch::cat({obs_enc,sg}, 1);

    std::vector<torch::jit::IValue> mlp_input;
    mlp_input.push_back(mlp_input_tensor);
    auto mlp_output = MLP->forward(mlp_input);
    torch::Tensor res = mlp_output.toTensor().to(at::kCPU);

    auto res_a = res.accessor<float,2>(); // accesor for the tensor

    std::vector<double> state_vec;
    for (int i = 0; i < dim; i++)
    {
        state_vec.push_back(res_a[0][i]);
    }
    std::vector<double> unnormalized_state_vec;
    this->unnormalize(state_vec, unnormalized_state_vec);
    for (int i = 0; i < dim; i++)
    {
        //TODO: better assign by using angleAxis
        //next->as<base::RealVectorStateSpace::StateType>()->values[i] = res_a[0][i];

        next_state[i] = unnormalized_state_vec[i];
    }
    #ifdef DEBUG
        std::cout << "next_state = [" << next_state[0] << ", " << next_state[1] << ", " << next_state[2] << ", " << next_state[3] <<"]" << std::endl;
        std::cout << "finished mpnet_predict." << std::endl;
    #endif
}

void MPNetSMP::init_informer(at::Tensor obs, const std::vector<double>& start_state, const std::vector<double>& goal_state, traj_t& res)
{
    /**
    x_init:
    1. obtain delta_x from start_state and goal_state
    2. wrap_angle: if delta_x[i] is circular, then wrap the angle. If the angle is close to pi, then randomly decide the wrapping
    3. add randomness to the x_init by normal sampling

    u_init:
    random uniform

    t_init:
    1. (option) use the neural cost estimator
    2. (option) use one step for each case
    */

    // calculate x_init
    #ifdef DEBUG
        std::cout << "inside init_informer..." << std::endl;
    #endif
    std::vector<double> delta_x(this->state_dim);
    for (unsigned i=0; i<this->state_dim; i++)
    {
        delta_x[i] = goal_state[i] - start_state[i];
        if (this->is_circular[i])
        {
            delta_x[i] = delta_x[i] - floor(delta_x[i] / (2*M_PI))*(2*M_PI);
            if (delta_x[i] > M_PI)
            {
                delta_x[i] = delta_x[i] - 2*M_PI;
            }
            int rand_d = rand() % 2;  // use this to decide when angle close to PI
            if (rand_d < 1 && abs(delta_x[i]) >= M_PI*0.5)
            {
                if (delta_x[i] > 0.)
                {
                    delta_x[i] = delta_x[i] - 2*M_PI;
                }
                else
                {
                    delta_x[i] = delta_x[i] + 2*M_PI;
                }
            }
        }
        delta_x[i] = delta_x[i] / (this->psopt_num_steps-1);
    }
    std::normal_distribution<double> distribution(0.0,0.02);
    // create x_init from delta_x
    for (unsigned i=0; i<this->psopt_num_steps; i++)
    {
        std::vector<double> state_i;
        for (unsigned j=0; j < this->state_dim; j++)
        {
            state_i.push_back(start_state[j] + delta_x[j] * i);
            if (i != 0 && i != this->psopt_num_steps-1)
            {
                // add randomness
                state_i[j] = state_i[j] + distribution(generator);
            }
        }
        //std::cout << "x_init[" << i << " = [" << state_i[0] << ", " << state_i[1] << ", " << state_i[2] << ", " << state_i[3] <<"]" << std::endl;

        res.x.push_back(state_i);
    }

    // obtain u_init by unform sampling
    std::uniform_real_distribution<double> uni_distribution(0.0,1.0);
    for (unsigned i=0; i<this->psopt_num_steps; i++)
    {
        std::vector<double> control_i;
        for (unsigned j=0; j < control_dim; j++)
        {
            control_i.push_back(uni_distribution(generator)*(control_upper_bound[j]-control_lower_bound[j])+control_lower_bound[j]);
        }
        //std::cout << "u_init[" << i << " = [" << control_i[0] <<"]" << std::endl;
        res.u.push_back(control_i);
    }

    // obtain t_init by setting to step_sz
    res.t.push_back(0.);
    for (unsigned i=1; i<this->psopt_num_steps; i++)
    {
        res.t.push_back(res.t[i-1]+this->psopt_step_sz);
    }
    #ifdef DEBUG
        std::cout << "complete init_informer." << std::endl;
    #endif

}


void MPNetSMP::plan(planner_t* SMP, system_t* system, psopt_system_t* psopt_system, at::Tensor &obs, std::vector<double>& start_state, std::vector<double>& goal_state, std::vector<double>& goal_inform_state,
                    int max_iteration, double goal_radius,
                    std::vector<std::vector<double>>& res_x, std::vector<std::vector<double>>& res_u, std::vector<double>& res_t)
{
    /**
        each iteration:
            x_hat = informer(x_t, x_G)
            if for some frequency, x_hat = x_G
            x_traj, u_traj, t_traj = init_informer(x_t, x_hat)
            x_t_1, edge, valid = planner->step_bvp(x_t, x_hat, x_traj, u_traj, t_traj)
            if not valid:
                x_t = x0
            else:
                x_t = x_t_1
    */
    std::vector<double> state_t = start_state;
    torch::Tensor obs_tensor = obs.to(at::kCUDA);
    clock_t begin_time;
    //mlp_input_tensor = torch::cat({obs_enc,sg}, 1);

    std::vector<torch::jit::IValue> obs_input;
    obs_input.push_back(obs_tensor);
    at::Tensor obs_enc = encoder->forward(obs_input).toTensor().to(at::kCPU);
    double* state_t_ptr = new double[this->state_dim];
    double* next_state_ptr = new double[this->state_dim];
    std::cout << "this->psopt_num_iters: " << this->psopt_num_iters << std::endl;

    for (unsigned i=1; i<=max_iteration; i++)
    {
        std::cout << "iteration " << i << std::endl;
        #ifdef DEBUG
            std::cout << "iteration " << i << std::endl;
            std::cout << "state_t = [" << state_t[0] << ", " << state_t[1] << ", " << state_t[2] << ", " << state_t[3] <<"]" << std::endl;
        #endif
        // given the previous result of bvp, find the next starting point (nearest in the tree)
        for (unsigned j=0; j < this->state_dim; j++)
        {
            state_t_ptr[j] = state_t[j];
        }
        SMP->nearest_state(state_t_ptr, state_t);

        std::vector<double> next_state(state_dim);
        if (i % 40 == 0)
        {
            // sample the goal instead
            next_state = goal_state;
        }
        else if (i % 20 == 0)
        {
            // sample the goal instead
            next_state = goal_inform_state;
        }
        else
        {
            begin_time = clock();
            this->informer(obs_enc, state_t, goal_inform_state, next_state);
            std::cout << "informer time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;

        }
        // according to next_state (MPNet sample), change start state to nearest_neighbors of next_state to
        // use search tree
        //for (unsigned j=0; j < this->state_dim; j++)
        //{
        //    state_t_ptr[j] = next_state[j];
        //}
        //SMP->nearest_state(state_t_ptr, state_t);

        // obtain init
        traj_t init_traj;
        begin_time = clock();
        this->init_informer(obs_enc, state_t, next_state, init_traj);
        std::cout << "init_informer time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;

        psopt_result_t res;
        for (unsigned j=0; j < this->state_dim; j++)
        {
            state_t_ptr[j] = state_t[j];
            next_state_ptr[j] = next_state[j];
        }
        #ifdef DEBUG
            //std::cout << "after copying state" << std::endl;
            //std::cout << "this->psopt_num_iters: " << this->psopt_num_iters << std::endl;
        #endif
        begin_time = clock();
        std::cout << "step_bvp num_iters: " << this->psopt_num_iters << std::endl;
        std::cout << "step_bvp start_state = [" << state_t[0] << ", " << state_t[1] << ", " << state_t[2] << ", " << state_t[3] <<"]" << std::endl;
        std::cout << "step_bvp next_state = [" << next_state[0] << ", " << next_state[1] << ", " << next_state[2] << ", " << next_state[3] <<"]" << std::endl;

        SMP->step_bvp(system, psopt_system, res, state_t_ptr, next_state_ptr, this->psopt_num_iters, this->psopt_num_steps, this->psopt_step_sz,
   	                 init_traj.x, init_traj.u, init_traj.t);
        std::cout << "step_bvp time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;

         #ifdef DEBUG
             std::cout << "after step_bvp" << std::endl;
         #endif
        if (res.u.size() == 0)
        {
            #ifdef DEBUG
                std::cout << "step_bvp unsuccessful." << std::endl;
            #endif
            // not valid path
            state_t = start_state;
        }
        else
        {
            // use the endpoint
            state_t = res.x.back();
            //#ifdef DEBUG
                std::cout << "step_bvp successful." << std::endl;
                // print out the result of bvp
                for (unsigned j=0; j < res.x.size(); j++)
                {
                    std::cout << "res.x[" << j << " = [" << res.x[j][0] << ", " << res.x[j][1] << ", " << res.x[j][2] << ", " << res.x[j][3] <<"]" << std::endl;
                }
                for (unsigned j=0; j < res.t.size(); j++)
                {
                    std::cout << "res.t[" << j << " = " << res.t << std::endl;
                }
            //#endif

        }
    // check if solution exists
    SMP->get_solution(res_x, res_u, res_t);
    if (res_x.size() != 0)
    {
        // solved
        delete state_t_ptr;
        delete next_state_ptr;
        return;
    }
    }
    // check if solved
    SMP->get_solution(res_x, res_u, res_t);
    // visualize

    delete state_t_ptr;
    delete next_state_ptr;
    if (res_x.size() != 0)
    {
        // solved
        return;
    }
}

void MPNetSMP::plan_step(planner_t* SMP, system_t* system, psopt_system_t* psopt_system, at::Tensor &obs, std::vector<double>& start_state, std::vector<double>& goal_state, std::vector<double>& goal_inform_state,
                    int max_iteration, double goal_radius,
                    std::vector<std::vector<double>>& res_x, std::vector<std::vector<double>>& res_u, std::vector<double>& res_t)
{
    std::vector<double> state_t = start_state;
    torch::Tensor obs_tensor = obs.to(at::kCUDA);
    clock_t begin_time;
    //mlp_input_tensor = torch::cat({obs_enc,sg}, 1);

    std::vector<torch::jit::IValue> obs_input;
    obs_input.push_back(obs_tensor);
    at::Tensor obs_enc = encoder->forward(obs_input).toTensor().to(at::kCPU);
    double* state_t_ptr = new double[this->state_dim];
    double* next_state_ptr = new double[this->state_dim];
    std::cout << "this->psopt_num_iters: " << this->psopt_num_iters << std::endl;


    // given the previous result of bvp, find the next starting point (nearest in the tree)
    for (unsigned j=0; j < this->state_dim; j++)
    {
        state_t_ptr[j] = state_t[j];
    }
    SMP->nearest_state(state_t_ptr, state_t);

    std::vector<double> next_state(state_dim);
    /**
    if (i % 40 == 0)
    {
        // sample the goal instead
        next_state = goal_state;
    }
    else if (i % 20 == 0)
    {
        // sample the goal instead
        next_state = goal_inform_state;
    }
    else
    */
    std::cout << "before informer" << std::endl;
    {
        begin_time = clock();
        this->informer(obs_enc, state_t, goal_inform_state, next_state);
        std::cout << "informer time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;

    }
    // according to next_state (MPNet sample), change start state to nearest_neighbors of next_state to
    // use search tree
    //for (unsigned j=0; j < this->state_dim; j++)
    //{
    //    state_t_ptr[j] = next_state[j];
    //}
    //SMP->nearest_state(state_t_ptr, state_t);

    // obtain init
    traj_t init_traj;
    begin_time = clock();
    this->init_informer(obs_enc, state_t, next_state, init_traj);
    std::cout << "init_informer time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;

    psopt_result_t res;
    for (unsigned j=0; j < this->state_dim; j++)
    {
        state_t_ptr[j] = state_t[j];
        next_state_ptr[j] = next_state[j];
    }
    #ifdef DEBUG
        //std::cout << "after copying state" << std::endl;
        //std::cout << "this->psopt_num_iters: " << this->psopt_num_iters << std::endl;
    #endif
    begin_time = clock();
    std::cout << "step_bvp num_iters: " << this->psopt_num_iters << std::endl;
    std::cout << "step_bvp start_state = [" << state_t[0] << ", " << state_t[1] << ", " << state_t[2] << ", " << state_t[3] <<"]" << std::endl;
    std::cout << "step_bvp next_state = [" << next_state[0] << ", " << next_state[1] << ", " << next_state[2] << ", " << next_state[3] <<"]" << std::endl;

    SMP->step_bvp(system, psopt_system, res, state_t_ptr, next_state_ptr, this->psopt_num_iters, this->psopt_num_steps, this->psopt_step_sz,
                 init_traj.x, init_traj.u, init_traj.t);
    std::cout << "step_bvp time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
    res_x = res.x;
    res_u = res.u;
    res_t = res.t;

}
