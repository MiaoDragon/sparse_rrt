#include "neural/neural_deep_smp.hpp"
#define DEBUG 1
MPNetSMP::MPNetSMP(std::string mlp_path, std::string encoder_path,
                   int num_iters_in, int num_steps_in, double step_sz_in,
                   system_t& system_in, psopt_system_t& psopt_system_in  //TODO: add clone to make code more secure
                   )
                   : system(&system_in)
                   , psopt_system(&psopt_system_in)
                   , psopt_num_iters(num_iters_in)
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
    #ifdef DEBUG
        std::cout << "state dimension: "  << dim << std::endl;
    #endif

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
    #ifdef DEBUG
        std::cout << "after planning..." << std::endl;
        std::cout << "tensor..." << std::endl;
    #endif
    for (int i = 0; i < dim; i++)
    {
        //TODO: better assign by using angleAxis
        //next->as<base::RealVectorStateSpace::StateType>()->values[i] = res_a[0][i];
        #ifdef DEBUG
            std::cout << "res_a[0][" << i << "]: " << res_a[0][i] << std::endl;
        #endif
        next_state[i] = unnormalized_state_vec[i];
    }
    #ifdef DEBUG
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
    for (unsigned i=0; i<this->psopt_num_steps-1; i++)
    {
        std::vector<double> state_i;
        for (unsigned j=0; j < this->state_dim; j++)
        {
            state_i.push_back(start_state[j] + delta_x[j] * (i+1));
            if (i != this->psopt_num_steps-1)
            {
                // add randomness
                state_i[j] = state_i[j] + distribution(generator);
            }
        }
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
        res.u.push_back(control_i);
    }

    // obtain t_init by setting to step_sz
    for (unsigned i=0; i<this->psopt_num_steps; i++)
    {
        res.t.push_back(this->psopt_step_sz);
    }
    #ifdef DEBUG
        std::cout << "complete init_informer." << std::endl;
    #endif

}


void MPNetSMP::plan(planner_t& SMP, at::Tensor obs, std::vector<double> start_state, std::vector<double> goal_state, int max_iteration, double goal_radius,
                    std::vector<std::vector<double>> res_x, std::vector<std::vector<double>> res_u, std::vector<double> res_t)
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
    for (unsigned i=0; i<max_iteration; i++)
    {
        #ifdef DEBUG
            std::cout << "iteration " << i << std::endl;
        #endif
        std::vector<double> next_state(state_dim);
        if (i % 10 == 0)
        {
            // sample the goal instead
            next_state = goal_state;
        }
        else
        {
            this->informer(obs, state_t, next_state, next_state);
        }
        // obtain init
        traj_t init_traj;
        this->init_informer(obs, state_t, next_state, init_traj);
        psopt_result_t res;
        double* state_t_ptr = new double[this->state_dim];
        double* next_state_ptr = new double[this->state_dim];
        for (unsigned j=0; j < this->state_dim; j++)
        {
            state_t_ptr[j] = state_t[j];
            next_state_ptr[j] = next_state[j];
        }
        #ifdef DEBUG
            std::cout << "after copying state" << std::endl;
            std::cout << "SMP.state_dimension:" << SMP.state_dimension << std::endl;
        #endif
        SMP.step_bvp(this->system.get(), this->psopt_system.get(), res, state_t_ptr, next_state_ptr, this->psopt_num_iters, this->psopt_num_steps, this->psopt_step_sz,
   	     init_traj.x, init_traj.u, init_traj.t);
         #ifdef DEBUG
             std::cout << "after step_bvp" << std::endl;
         #endif
        if (init_traj.u.size() == 0)
        {
            // not valid path
            state_t = start_state;
        }
        else
        {
            // use the endpoint
            state_t = init_traj.x.back();
        }
    }
    // check if solved
    SMP.get_solution(res_x, res_u, res_t);
    if (res_x.size() != 0)
    {
        // solved
        return;
    }
}
