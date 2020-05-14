//#define DEBUG 1
#include "neural/neural_deep_smp.hpp"
#include <time.h>
MPNetSMP::MPNetSMP(std::string mlp_path, std::string encoder_path,
                   std::string cost_mlp_path, std::string cost_encoder_path,
                   system_t* system,
                   int num_iters_in, int num_steps_in, double step_sz_in,
                   int device
                   )
                   : psopt_num_iters(num_iters_in)
                   , psopt_num_steps(num_steps_in)
                   , psopt_step_sz(step_sz_in)
                   , gpu_device(device)
{
    // neural network
    MLP.reset(new torch::jit::script::Module(torch::jit::load(mlp_path)));
    MLP->to(at::Device("cuda:"+std::to_string(device)));
    #ifdef DEBUG
        std::cout << "loaded MLP" << std::endl;
    #endif
    encoder.reset(new torch::jit::script::Module(torch::jit::load(encoder_path)));
    encoder->to(at::Device("cuda:"+std::to_string(device)));
    #ifdef DEBUG
        std::cout << "loaded modules" << std::endl;
    #endif

    cost_MLP.reset(new torch::jit::script::Module(torch::jit::load(cost_mlp_path)));
    cost_MLP->to(at::Device("cuda:"+std::to_string(device)));
    cost_encoder.reset(new torch::jit::script::Module(torch::jit::load(cost_encoder_path)));
    cost_encoder->to(at::Device("cuda:"+std::to_string(device)));
    #ifdef DEBUG
        std::cout << "loaded cost modules" << std::endl;
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
    cost_MLP.reset();
    cost_encoder.reset();
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

torch::Tensor MPNetSMP::getStateTensorWithNormalization(const std::vector<double>& state)
{
    std::vector<double> normalized_vec;
    this->normalize(state, normalized_vec);
    // double->float
    std::vector<float> float_normalized_vec;
    for (unsigned i=0; i<this->state_dim; i++)
    {
        float_normalized_vec.push_back(float(normalized_vec[i]));
    }
    std::cout << "in getStateTensorWithNormalization:" << std::endl;
    torch::Tensor tensor = torch::from_blob(float_normalized_vec.data(), {1, this->state_dim}).clone();
    std::cout << "float_normalized_vec: " << float_normalized_vec << std::endl;
    std::cout << "tensor: " << tensor << std::endl;
    return tensor;
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

torch::Tensor MPNetSMP::getStartGoalTensorBatch(const std::vector<std::vector<double>>& start_state, const std::vector<std::vector<double>>& goal_state)
{
    std::vector<float> float_normalized_start_vec;  // flattened vector for tensor input
    std::vector<float> float_normalized_goal_vec;
    for (unsigned i=0; i<start_state.size(); i++)
    {
        std::vector<double> normalized_start_vec_i;
        std::vector<double> normalized_goal_vec_i;
        this->normalize(start_state[i], normalized_start_vec_i);
        this->normalize(goal_state[i], normalized_goal_vec_i);
        // double->float
        //std::vector<float> float_normalized_start_vec_i;
        //std::vector<float> float_normalized_goal_vec_i;
        for (unsigned j=0; j<this->state_dim; j++)
        {
            //float_normalized_start_vec_i.push_back(float(normalized_start_vec_i[j]));
            //float_normalized_goal_vec_i.push_back(float(normalized_goal_vec_i[j]));
            //std::cout << "float_normalized_start_vec[" << i << "][" << j << "]: " << float_normalized_start_vec_i[j] << std::endl;
            //std::cout << "float_normalized_goal_vec[" << i << "][" << j << "]: " << float_normalized_goal_vec_i[j] << std::endl;
            float_normalized_start_vec.push_back(float(normalized_start_vec_i[j]));
            float_normalized_goal_vec.push_back(float(normalized_goal_vec_i[j]));

        }
    }
    torch::Tensor start_tensor = torch::from_blob(float_normalized_start_vec.data(), {start_state.size(), this->state_dim});
    torch::Tensor goal_tensor = torch::from_blob(float_normalized_goal_vec.data(), {start_state.size(), this->state_dim});

    //std::cout << "before concatenation: " << std::endl;
    //std::cout << "start tensor: " << start_tensor << std::endl;
    //std::cout << "goal tensor: " << goal_tensor << std::endl;

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
    //std::cout << "after concatenation:" << std::endl;
    //std::cout << sg_cat <<std::endl;

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
        // after normalization, wrap angle (to nearest state)

        double delta_x = next_state[i] - start_state[i];
        if (this->is_circular[i])
        {
            delta_x = delta_x - floor(delta_x / (2*M_PI))*(2*M_PI);
            if (delta_x > M_PI)
            {
                delta_x = delta_x - 2*M_PI;
            }
        }
        next_state[i] = start_state[i] + delta_x;
    }
    #ifdef DEBUG
        std::cout << "next_state = [" << next_state[0] << ", " << next_state[1] << ", " << next_state[2] << ", " << next_state[3] <<"]" << std::endl;
        std::cout << "finished mpnet_predict." << std::endl;
    #endif
}

void MPNetSMP::informer_batch(at::Tensor obs, const std::vector<double>& start_state, const std::vector<double>& goal_state, std::vector<std::vector<double>>& next_state, int num_sample)
{
    // given the start and goal, and the internal obstacle representation
    // convert them to torch::Tensor, and feed into MPNet
    // return a batch of next states to the "next" parameter
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
    torch::Tensor mlp_input_tensor_expand = mlp_input_tensor.repeat({num_sample, 1});

    // batch obtain result
    std::vector<torch::jit::IValue> mlp_input;
    mlp_input.push_back(mlp_input_tensor_expand);
    auto mlp_output = MLP->forward(mlp_input);
    torch::Tensor res = mlp_output.toTensor().to(at::kCPU);

    auto res_a = res.accessor<float,2>(); // accesor for the tensor
    for (int i = 0; i < num_sample; i++)
    {
        std::vector<double> state_vec;
        for (int j = 0; j < dim; j++)
        {
            state_vec.push_back(res_a[i][j]);
        }
        std::vector<double> unnormalized_state_vec;
        this->unnormalize(state_vec, unnormalized_state_vec);
        for (int j = 0; j < dim; j++)
        {
            //TODO: better assign by using angleAxis
            //next->as<base::RealVectorStateSpace::StateType>()->values[i] = res_a[0][i];

            next_state[i][j] = unnormalized_state_vec[j];
            // after normalization, wrap angle (to nearest state)

            double delta_x = next_state[i][j] - start_state[j];
            if (this->is_circular[j])
            {
                delta_x = delta_x - floor(delta_x / (2*M_PI))*(2*M_PI);
                if (delta_x > M_PI)
                {
                    delta_x = delta_x - 2*M_PI;
                }
            }
            next_state[i][j] = start_state[j] + delta_x;
        }
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
            if (rand_d < 1 && abs(delta_x[i]) >= M_PI*0.9)
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
        //std::cout << "t_init[" << i << "] =" << res.t[i]  << std::endl;

    }
    #ifdef DEBUG
        std::cout << "complete init_informer." << std::endl;
    #endif

}

void MPNetSMP::cost_informer(at::Tensor obs, const std::vector<double>& start_state, const std::vector<double>& goal_state, double& cost)
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
    auto mlp_output = cost_MLP->forward(mlp_input);
    torch::Tensor res = mlp_output.toTensor().to(at::kCPU);

    auto res_a = res.accessor<float,2>(); // accesor for the tensor

    cost = res_a[0][0];
    #ifdef DEBUG
        std::cout << "cost predictor result: " << res_a[0][0] << std::endl;
        std::cout << "finished mpnet_predict." << std::endl;
    #endif
}

void MPNetSMP::cost_informer_batch(at::Tensor obs, const std::vector<std::vector<double>>& start_state, const std::vector<std::vector<double>>& goal_state, std::vector<double>& cost, int num_sample)
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

    torch::Tensor sg = getStartGoalTensorBatch(start_state, goal_state);
    //torch::Tensor gs = getStartGoalTensor(goal, start, dim);

    torch::Tensor mlp_input_tensor_expand;
    // Note the order of the cat
    at::Tensor obs_expand = obs.repeat({num_sample, 1});
    mlp_input_tensor_expand = torch::cat({obs_expand,sg}, 1).to(at::kCUDA);
    //mlp_input_tensor = torch::cat({obs_enc,sg}, 1);
    //torch::Tensor mlp_input_tensor_expand = mlp_input_tensor.repeat({num_sample, 1});

    std::vector<torch::jit::IValue> mlp_input;
    mlp_input.push_back(mlp_input_tensor_expand);
    auto mlp_output = cost_MLP->forward(mlp_input);
    torch::Tensor res = mlp_output.toTensor().to(at::kCPU);

    auto res_a = res.accessor<float,2>(); // accesor for the tensor
    for (unsigned i=0; i<num_sample; i++)
    {
        cost[i] = res_a[i][0];
    }
    #ifdef DEBUG
        std::cout << "cost predictor result: " << res_a[0][0] << std::endl;
        std::cout << "finished mpnet_predict." << std::endl;
    #endif
}

// these acheive tensor-level functions
// obtain normalized result, need to unnormalize later
torch::Tensor MPNetSMP::tensor_informer(at::Tensor obs, at::Tensor start_state, at::Tensor goal_state)
{
    // achieve batch
    torch::Tensor mlp_input_tensor;
    // Note the order of the cat
    // assuming tensors are all on GPU
    mlp_input_tensor = torch::cat({obs,start_state, goal_state}, 1).to(at::Device("cuda:"+std::to_string(this->gpu_device)));
    // batch obtain result
    std::vector<torch::jit::IValue> mlp_input;
    mlp_input.push_back(mlp_input_tensor);
    auto mlp_output = MLP->forward(mlp_input);
    torch::Tensor res = mlp_output.toTensor();
    return res;
}


torch::Tensor MPNetSMP::tensor_cost_informer(at::Tensor obs, at::Tensor start_state, at::Tensor goal_state)
{
    // achieve batch
    torch::Tensor mlp_input_tensor;
    // Note the order of the cat
    mlp_input_tensor = torch::cat({obs,start_state,goal_state}, 1).to(at::Device("cuda:"+std::to_string(this->gpu_device)));
    std::vector<torch::jit::IValue> mlp_input;
    mlp_input.push_back(mlp_input_tensor);
    auto mlp_output = cost_MLP->forward(mlp_input);
    torch::Tensor res = mlp_output.toTensor();
    return res;
}

void MPNetSMP::plan_tree(planner_t* SMP, system_t* system, psopt_system_t* psopt_system, at::Tensor &obs, std::vector<double>& start_state, std::vector<double>& goal_state, std::vector<double>& goal_inform_state,
                    int max_iteration, double goal_radius,
                    std::vector<std::vector<double>>& res_x, std::vector<std::vector<double>>& res_u, std::vector<double>& res_t)
{
    /**
        new method: from linjun
        each iteration:
            x <- random_sample()
            x <- nearest_neighbor(x)
            informed_extend(x, xG)   ---  find x_t_1, then BVP(xt, x_t_1)
    */


    /**
        old method: (idea from DeepSMP)
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
    //std::cout << "this->psopt_num_iters: " << this->psopt_num_iters << std::endl;
    int flag=1; // flag=1: using MPNet
                // flag=0: using goal
    double pick_goal_threshold = 0.05;
    std::uniform_real_distribution<double> uni_distribution(0.0,1.0); // based on this sample goal
    int goal_linear_inc_start_iter = floor(0.7*max_iteration);
    int goal_linear_inc_end_iter = max_iteration;
    double goal_linear_inc_end_threshold = 0.8;
    double goal_linear_inc = (goal_linear_inc_end_threshold - pick_goal_threshold) / (goal_linear_inc_end_iter - goal_linear_inc_start_iter);
    for (unsigned i=1; i<=max_iteration; i++)
    {
        //std::cout << "iteration " << i << std::endl;
        #ifdef DEBUG
            std::cout << "iteration " << i << std::endl;
            std::cout << "state_t = [" << state_t[0] << ", " << state_t[1] << ", " << state_t[2] << ", " << state_t[3] <<"]" << std::endl;
        #endif
        // given the previous result of bvp, find the next starting point (nearest in the tree)
        //for (unsigned j=0; j < this->state_dim; j++)
        //{
        //    state_t_ptr[j] = state_t[j];
        //}
        //SMP->nearest_state(state_t_ptr, state_t);
        // randomly sample and find nearest_state as BVP starting point
        SMP->random_state(state_t_ptr); // random sample
        // find nearest_neighbor of random sample state_t_ptr, and assign to state_t
        SMP->nearest_state(state_t_ptr, state_t);

        std::vector<double> next_state(this->state_dim);
        double use_goal_prob = uni_distribution(generator);
        if (i > goal_linear_inc_start_iter)
        {
            pick_goal_threshold += goal_linear_inc;
        }

        if (use_goal_prob <= pick_goal_threshold)
        {
            // sample the goal instead when enough max_iteration is used
            next_state = goal_state;
            flag=0;
        }
        else
        {
            begin_time = clock();
            this->informer(obs_enc, state_t, goal_inform_state, next_state);
        #ifdef COUNT_TIME
            std::cout << "informer time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
        #endif
            flag=1;
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
        #ifdef COUNT_TIME
        std::cout << "init_informer time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
        #endif
        psopt_result_t res;
        // copy to c++ double* list from std::vector
        for (unsigned j=0; j < this->state_dim; j++)
        {
            state_t_ptr[j] = state_t[j];
            next_state_ptr[j] = next_state[j];
        }
        #ifdef DEBUG
            std::cout << "after copying state" << std::endl;
            std::cout << "this->psopt_num_iters: " << this->psopt_num_iters << std::endl;
        #endif
        begin_time = clock();
        #ifdef DEBUG
            std::cout << "step_bvp num_iters: " << this->psopt_num_iters << std::endl;
            std::cout << "step_bvp start_state = [" << state_t[0] << ", " << state_t[1] << ", " << state_t[2] << ", " << state_t[3] <<"]" << std::endl;
            std::cout << "step_bvp next_state = [" << next_state[0] << ", " << next_state[1] << ", " << next_state[2] << ", " << next_state[3] <<"]" << std::endl;
        #endif

        SMP->step_bvp(system, psopt_system, res, state_t_ptr, next_state_ptr, this->psopt_num_iters, this->psopt_num_steps, this->psopt_step_sz,
   	                 init_traj.x, init_traj.u, init_traj.t);
        #ifdef COUNT_TIME
        std::cout << "step_bvp time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
        #endif
        #ifdef DEBUG
             std::cout << "after step_bvp" << std::endl;
        #endif

        /**
        //****** Remove the comment if want to use step_bvp for tree search
        if (flag)
        {
            // flag=1: MPNet. if using MPNet path, then update state_t, otherwise keep it fixed
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
                #ifdef DEBUG
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
                #endif
            }
        }
        */

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

void MPNetSMP::plan_line(planner_t* SMP, system_t* system, psopt_system_t* psopt_system, at::Tensor &obs, std::vector<double>& start_state, std::vector<double>& goal_state, std::vector<double>& goal_inform_state,
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
    //std::cout << "this->psopt_num_iters: " << this->psopt_num_iters << std::endl;
    double pick_goal_threshold = 0.25;
    std::uniform_real_distribution<double> uni_distribution(0.0,1.0); // based on this sample goal
    int goal_linear_inc_start_iter = floor(0.4*max_iteration);
    int goal_linear_inc_end_iter = max_iteration;
    double goal_linear_inc_end_threshold = 0.95;
    double goal_linear_inc = (goal_linear_inc_end_threshold - pick_goal_threshold) / (goal_linear_inc_end_iter - goal_linear_inc_start_iter);

    int flag=1; // flag=1: using MPNet path
    for (unsigned i=1; i<=max_iteration; i++)
    {
        //if (i % 50 == 0)
        //{
        //    //std::cout << "iteration: " << i << std::endl;
        //}
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

        std::vector<double> next_state(this->state_dim);
        double use_goal_prob = uni_distribution(generator);
        if (i > goal_linear_inc_start_iter)
        {
            pick_goal_threshold += goal_linear_inc;
        }

        if (use_goal_prob <= pick_goal_threshold)
        {
            // sample the goal instead when enough max_iteration is used
            next_state = goal_state;
            flag=0;
        }
        else
        {
            flag=1;
            begin_time = clock();
            this->informer(obs_enc, state_t, goal_inform_state, next_state);
        #ifdef COUNT_TIME
            std::cout << "informer time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
        #endif
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
        #ifdef COUNT_TIME
        std::cout << "init_informer time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
        #endif
        psopt_result_t res;
        for (unsigned j=0; j < this->state_dim; j++)
        {
            state_t_ptr[j] = state_t[j];
            next_state_ptr[j] = next_state[j];
        }
        #ifdef DEBUG
            std::cout << "after copying state" << std::endl;
            std::cout << "this->psopt_num_iters: " << this->psopt_num_iters << std::endl;
        #endif
        begin_time = clock();
        #ifdef DEBUG
            std::cout << "step_bvp num_iters: " << this->psopt_num_iters << std::endl;
            std::cout << "step_bvp start_state = [" << state_t[0] << ", " << state_t[1] << ", " << state_t[2] << ", " << state_t[3] <<"]" << std::endl;
            std::cout << "step_bvp next_state = [" << next_state[0] << ", " << next_state[1] << ", " << next_state[2] << ", " << next_state[3] <<"]" << std::endl;
        #endif

        SMP->step_bvp(system, psopt_system, res, state_t_ptr, next_state_ptr, this->psopt_num_iters, this->psopt_num_steps, this->psopt_step_sz,
   	                 init_traj.x, init_traj.u, init_traj.t);
        #ifdef COUNT_TIME
        std::cout << "step_bvp time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
        #endif
         #ifdef DEBUG
             std::cout << "after step_bvp" << std::endl;
         #endif

         if (flag)  // if using MPNet prediction, then update state_t
         {
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
                 #ifdef DEBUG
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
                 #endif

             }
         }

        // check if solution exists
        SMP->get_solution(res_x, res_u, res_t);
        if (res_x.size() != 0)
        {
            // solved
            std::cout << "solved in neural smp" << std::endl;
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


// Using original DeepSMP method
void MPNetSMP::plan_tree_SMP(planner_t* SMP, system_t* system, psopt_system_t* psopt_system, at::Tensor &obs, std::vector<double>& start_state, std::vector<double>& goal_state, std::vector<double>& goal_inform_state,
                    int max_iteration, double goal_radius, double cost_threshold,
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
    double* new_state = new double[this->state_dim];
    double* new_control = new double[this->control_dim];
    double* from_state = new double[this->state_dim];
    //std::cout << "this->psopt_num_iters: " << this->psopt_num_iters << std::endl;
    int flag=1;  // flag=1: using MPNet
                 // flag=0: not using MPNet
     double pick_goal_threshold = 0.25;
     std::uniform_real_distribution<double> uni_distribution(0.0,1.0); // based on this sample goal
     int goal_linear_inc_start_iter = floor(0.4*max_iteration);
     int goal_linear_inc_end_iter = max_iteration;
     double goal_linear_inc_end_threshold = 0.95;
     double goal_linear_inc = (goal_linear_inc_end_threshold - pick_goal_threshold) / (goal_linear_inc_end_iter - goal_linear_inc_start_iter);
    for (unsigned i=1; i<=max_iteration; i++)
    {
        //std::cout << "iteration " << i << std::endl;
        #ifdef DEBUG
            std::cout << "iteration " << i << std::endl;
            std::cout << "state_t = [" << state_t[0] << ", " << state_t[1] << ", " << state_t[2] << ", " << state_t[3] <<"]" << std::endl;
        #endif
        // given the previous result of bvp, find the next starting point (nearest in the tree)
        //for (unsigned j=0; j < this->state_dim; j++)
        //{
        //    state_t_ptr[j] = state_t[j];
        //}
        //SMP->nearest_state(state_t_ptr, state_t);

        std::vector<double> next_state(this->state_dim);
        double use_goal_prob = uni_distribution(generator);
        // update pick_goal_threshold based on iteration number
        if (i > goal_linear_inc_start_iter)
        {
            pick_goal_threshold += goal_linear_inc;
        }

        if (use_goal_prob <= pick_goal_threshold)
        {
            // sample the goal instead when enough max_iteration is used
            next_state = goal_state;
            flag=0;
        }
        else
        {
            flag=1;
            begin_time = clock();
            this->informer(obs_enc, state_t, goal_inform_state, next_state);
        #ifdef COUNT_TIME
            std::cout << "informer time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
        #endif
        }
        // according to next_state (MPNet sample), change start state to nearest_neighbors of next_state to
        // use search tree
        //for (unsigned j=0; j < this->state_dim; j++)
        //{
        //    state_t_ptr[j] = next_state[j];
        //}
        //SMP->nearest_state(state_t_ptr, state_t);
        // copy to c++ double* list from std::vector
        for (unsigned j=0; j < this->state_dim; j++)
        {
            state_t_ptr[j] = state_t[j];
            next_state_ptr[j] = next_state[j];
        }
        // below tries to use step_with_sample to imitate DeepSMP
        double new_time = 0.;
        int min_time_steps = 5;
        int max_time_steps = 100;
        SMP->step_with_sample(system, next_state_ptr, from_state, new_state, new_control, new_time, min_time_steps, max_time_steps, 0.02);

        // only when using MPNet, update the state_t using next_state. Otherwise not change
        if (flag)//flag=1: using MPNet.
        {
            if (new_time <= 0.01)
            {
                // propagate fails, back to origin
                state_t = start_state;
            }
            else
            {
                // propagation success
                state_t = next_state; // this using MPNet next sample instead of propagated state
                //for (unsigned j=0; j<this->state_dim; j++)
                //{
                //    state_t[j] = new_state[j];  // this uses propagated state after radom extension
                //}
            }
        }
         // check if solution exists
         SMP->get_solution(res_x, res_u, res_t);

        double total_t = 0.;
        for (unsigned j=0; j<res_t.size(); j++)
        {
            total_t += res_t[j];
        }
        if (res_x.size() != 0 && total_t <= cost_threshold)
        {
            // solved
            delete state_t_ptr;
            delete next_state_ptr;

            delete new_state;
            delete new_control;
            delete from_state;

            return;
        }
    }
    // check if solved
    SMP->get_solution(res_x, res_u, res_t);

    delete state_t_ptr;
    delete next_state_ptr;

    delete new_state;
    delete new_control;
    delete from_state;

    double total_t = 0.;
    for (unsigned j=0; j<res_t.size(); j++)
    {
        total_t += res_t[j];
    }
    if (res_x.size() != 0 && total_t <= cost_threshold)
    {
        // solved
        return;
    }
    else if (res_x.size() != 0)
    {
        // solved but cost not low enough
        res_x.clear();
        res_u.clear();
        res_t.clear();
    }
}

//*** Hybrid method, use sst samples sometimes
void MPNetSMP::plan_tree_SMP_hybrid(planner_t* SMP, system_t* system, psopt_system_t* psopt_system, at::Tensor &obs, std::vector<double>& start_state, std::vector<double>& goal_state, std::vector<double>& goal_inform_state,
                    int max_iteration, double goal_radius, double cost_threshold,
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
    double* new_state = new double[this->state_dim];
    double* new_control = new double[this->control_dim];
    double* from_state = new double[this->state_dim];
    //std::cout << "this->psopt_num_iters: " << this->psopt_num_iters << std::endl;
    int flag=1;  // flag=1: using MPNet sample, state_t will take next_state
                 // flag=0: not using MPNet sample, state_t don't change
    for (unsigned i=1; i<=max_iteration; i++)
    {
        //std::cout << "iteration " << i << std::endl;
        #ifdef DEBUG
            std::cout << "iteration " << i << std::endl;
            std::cout << "state_t = [" << state_t[0] << ", " << state_t[1] << ", " << state_t[2] << ", " << state_t[3] <<"]" << std::endl;
        #endif
        // given the previous result of bvp, find the next starting point (nearest in the tree)
        //for (unsigned j=0; j < this->state_dim; j++)
        //{
        //    state_t_ptr[j] = state_t[j];
        //}
        //SMP->nearest_state(state_t_ptr, state_t);

        std::vector<double> next_state(this->state_dim);
        if (i % 20 == 0)
        {
            // unifromly sample for fine-tuning
            SMP->random_state(next_state_ptr);
            for (unsigned j=0; j<this->state_dim; j++)
            {
                next_state[j] = next_state_ptr[j];
            }
            flag=0;
        }
        else if (i % 10 == 0)
        {
            // sample the goal instead
            next_state = goal_inform_state;
            flag=0;
        }
        else
        {
            begin_time = clock();
            this->informer(obs_enc, state_t, goal_inform_state, next_state);
        #ifdef COUNT_TIME
            std::cout << "informer time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
        #endif
            flag=1;
        }
        // according to next_state (MPNet sample), change start state to nearest_neighbors of next_state to
        // use search tree
        //for (unsigned j=0; j < this->state_dim; j++)
        //{
        //    state_t_ptr[j] = next_state[j];
        //}
        //SMP->nearest_state(state_t_ptr, state_t);
        // copy to c++ double* list from std::vector
        for (unsigned j=0; j < this->state_dim; j++)
        {
            state_t_ptr[j] = state_t[j];
            next_state_ptr[j] = next_state[j];
        }
        // below tries to use step_with_sample to imitate DeepSMP
        double new_time = 0.;
        int min_time_steps = 5;
        int max_time_steps = 100;
        // given next_state_ptr, find nearest neighbor in the tree, and extend it
        SMP->step_with_sample(system, next_state_ptr, from_state, new_state, new_control, new_time, min_time_steps, max_time_steps, 0.02);


         if (flag)// flag: using MPNet. If not using, then won't change state_t
         {
             if (new_time == 0.)
             {
                 // propagate fails, back to origin
                 state_t = start_state;
             }
             else
             {
                 // propagation success
                 state_t = next_state; // using MPNet next sample
             }
         }
         // check if solution exists
         SMP->get_solution(res_x, res_u, res_t);

        double total_t = 0.;
        for (unsigned j=0; j<res_t.size(); j++)
        {
            total_t += res_t[j];
        }
        if (res_x.size() != 0 && total_t <= cost_threshold)
        {
            // solved
            delete state_t_ptr;
            delete next_state_ptr;

            delete new_state;
            delete new_control;
            delete from_state;

            return;
        }
    }
    // check if solved
    SMP->get_solution(res_x, res_u, res_t);

    delete state_t_ptr;
    delete next_state_ptr;

    delete new_state;
    delete new_control;
    delete from_state;

    double total_t = 0.;
    for (unsigned j=0; j<res_t.size(); j++)
    {
        total_t += res_t[j];
    }
    if (res_x.size() != 0 && total_t <= cost_threshold)
    {
        // solved
        return;
    }
    else if (res_x.size() != 0)
    {
        // solved but cost not low enough
        res_x.clear();
        res_u.clear();
        res_t.clear();
    }
}
//**********

//*** tree_SMP with cost-to-go function
// Using original DeepSMP method
void MPNetSMP::plan_tree_SMP_cost(planner_t* SMP, system_t* system, psopt_system_t* psopt_system, at::Tensor &obs, std::vector<double>& start_state, std::vector<double>& goal_state, std::vector<double>& goal_inform_state,
                    int max_iteration, double goal_radius, double cost_threshold,
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
    at::Tensor cost_obs_enc = cost_encoder->forward(obs_input).toTensor().to(at::kCPU);
    double* state_t_ptr = new double[this->state_dim];
    double* next_state_ptr = new double[this->state_dim];
    double* new_state = new double[this->state_dim];
    double* new_control = new double[this->control_dim];
    double* from_state = new double[this->state_dim];
    //std::cout << "this->psopt_num_iters: " << this->psopt_num_iters << std::endl;
    int flag=1;  // flag=1: using MPNet
                 // flag=0: not using MPNet
     double pick_goal_threshold = 0.25;
     std::uniform_real_distribution<double> uni_distribution(0.0,1.0); // based on this sample goal
     int goal_linear_inc_start_iter = floor(0.4*max_iteration);
     int goal_linear_inc_end_iter = max_iteration;
     double goal_linear_inc_end_threshold = 0.95;
     double goal_linear_inc = (goal_linear_inc_end_threshold - pick_goal_threshold) / (goal_linear_inc_end_iter - goal_linear_inc_start_iter);
    for (unsigned i=1; i<=max_iteration; i++)
    {
        if (i % 100 == 0)
        {
            std::cout << "iteration " << i << std::endl;

        }
        #ifdef DEBUG
            std::cout << "iteration " << i << std::endl;
            std::cout << "state_t = [" << state_t[0] << ", " << state_t[1] << ", " << state_t[2] << ", " << state_t[3] <<"]" << std::endl;
        #endif
        // given the previous result of bvp, find the next starting point (nearest in the tree)
        //for (unsigned j=0; j < this->state_dim; j++)
        //{
        //    state_t_ptr[j] = state_t[j];
        //}
        //SMP->nearest_state(state_t_ptr, state_t);

        std::vector<double> next_state(this->state_dim);
        double use_goal_prob = uni_distribution(generator);
        // update pick_goal_threshold based on iteration number
        if (i > goal_linear_inc_start_iter)
        {
            pick_goal_threshold += goal_linear_inc;
        }

        if (use_goal_prob <= pick_goal_threshold)
        {
            // sample the goal instead when enough max_iteration is used
            next_state = goal_state;
            flag=0;
        }
        else
        {
            //std::cout << "inside cost sampling" << std::endl;
            flag=1;
            begin_time = clock();
            // first sample several mpnet points, then use the costnet to find the best point
            int num_sample = 15;
            std::vector<std::vector<double>> next_state_candidate(num_sample,std::vector<double>(this->state_dim));
            std::vector<double> next_state_cost(num_sample);
            std::vector<std::vector<double>> cost_end_state(num_sample,std::vector<double>(this->state_dim));
            // construct cost_end_state
            for (unsigned j=0; j<num_sample; j++)
            {
                cost_end_state[j] = goal_inform_state;
            }
            this->informer_batch(obs_enc, state_t, goal_inform_state, next_state_candidate, num_sample);
            // calculate cost
            //std::cout << "after informer_batch" << std::endl;

            this->cost_informer_batch(cost_obs_enc, next_state_candidate, cost_end_state, next_state_cost, num_sample);

            //std::cout << "after cost_informer_batch" << std::endl;

            double best_cost = 100000.;
            int best_ind = -1;
            for (unsigned j=0; j<num_sample; j++)
            {
                //std::cout << "next_State_candidate[j]: [" << next_state_candidate[j] << "]" << std::endl;

                //std::cout << "next_state_cost[j]: " << next_state_cost[j] << std::endl;
                if (next_state_cost[j] < best_cost)
                {
                    best_cost = next_state_cost[j];
                    best_ind = j;
                }
            }
            next_state = next_state_candidate[best_ind];
            //std::cout << "best_cost: " << best_cost << std::endl;
            //std::cout << "after cost sampling" << std::endl;
            //std::cout << "best_ind: " << best_ind << std::endl;

            //this->informer(obs_enc, state_t, goal_inform_state, next_state);
        #ifdef COUNT_TIME
            std::cout << "informer time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
        #endif
        }
        // according to next_state (MPNet sample), change start state to nearest_neighbors of next_state to
        // use search tree
        //for (unsigned j=0; j < this->state_dim; j++)
        //{
        //    state_t_ptr[j] = next_state[j];
        //}
        //SMP->nearest_state(state_t_ptr, state_t);
        // copy to c++ double* list from std::vector
        for (unsigned j=0; j < this->state_dim; j++)
        {
            state_t_ptr[j] = state_t[j];
            next_state_ptr[j] = next_state[j];
        }
        // below tries to use step_with_sample to imitate DeepSMP
        double new_time = 0.;
        int min_time_steps = 5;
        int max_time_steps = 100;
        SMP->step_with_sample(system, next_state_ptr, from_state, new_state, new_control, new_time, min_time_steps, max_time_steps, 0.02);

        // only when using MPNet, update the state_t using next_state. Otherwise not change
        if (flag)//flag=1: using MPNet.
        {
            if (new_time <= 0.01)
            {
                // propagate fails, back to origin
                state_t = start_state;
            }
            else
            {
                // propagation success
                state_t = next_state; // this using MPNet next sample instead of propagated state
                //for (unsigned j=0; j<this->state_dim; j++)
                //{
                //    state_t[j] = new_state[j];  // this uses propagated state after radom extension
                //}
            }
        }
         // check if solution exists
         SMP->get_solution(res_x, res_u, res_t);

        double total_t = 0.;
        for (unsigned j=0; j<res_t.size(); j++)
        {
            total_t += res_t[j];
        }
        if (res_x.size() != 0 && total_t <= cost_threshold)
        {
            // solved
            delete state_t_ptr;
            delete next_state_ptr;

            delete new_state;
            delete new_control;
            delete from_state;

            return;
        }
    }
    // check if solved
    SMP->get_solution(res_x, res_u, res_t);

    delete state_t_ptr;
    delete next_state_ptr;

    delete new_state;
    delete new_control;
    delete from_state;

    double total_t = 0.;
    for (unsigned j=0; j<res_t.size(); j++)
    {
        total_t += res_t[j];
    }
    if (res_x.size() != 0 && total_t <= cost_threshold)
    {
        // solved
        return;
    }
    else if (res_x.size() != 0)
    {
        // solved but cost not low enough
        res_x.clear();
        res_u.clear();
        res_t.clear();
    }
}
//**********


//*** tree_SMP with cost_to_go, and use cost_to_go gradient to update sample
//*** tree_SMP with cost-to-go function
// Using original DeepSMP method
void MPNetSMP::plan_tree_SMP_cost_gradient(planner_t* SMP, system_t* system, psopt_system_t* psopt_system, at::Tensor &obs, std::vector<double>& start_state, std::vector<double>& goal_state, std::vector<double>& goal_inform_state,
                    int max_iteration, double goal_radius, double cost_threshold,
                    std::vector<std::vector<double>>& res_x, std::vector<std::vector<double>>& res_u, std::vector<double>& res_t, int num_sample)
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
    torch::Tensor obs_tensor = obs.to(at::Device("cuda:"+std::to_string(this->gpu_device)));


    clock_t begin_time;
    //mlp_input_tensor = torch::cat({obs_enc,sg}, 1);

    std::vector<torch::jit::IValue> obs_input;
    obs_input.push_back(obs_tensor);
    at::Tensor obs_enc = encoder->forward(obs_input).toTensor().detach();
    at::Tensor obs_expand_enc = obs_enc.repeat({num_sample, 1}).detach();
    at::Tensor cost_obs_enc = cost_encoder->forward(obs_input).toTensor().detach();
    at::Tensor cost_obs_expand_enc = cost_obs_enc.repeat({num_sample,1}).detach();

    double* state_t_ptr = new double[this->state_dim];
    double* next_state_ptr = new double[this->state_dim];
    double* new_state = new double[this->state_dim];
    double* new_control = new double[this->control_dim];
    double* from_state = new double[this->state_dim];
    //std::cout << "this->psopt_num_iters: " << this->psopt_num_iters << std::endl;
    int flag=1;  // flag=1: using MPNet
                 // flag=0: not using MPNet
     double pick_goal_threshold = 0.25;
     std::uniform_real_distribution<double> uni_distribution(0.0,1.0); // based on this sample goal
     int goal_linear_inc_start_iter = floor(0.4*max_iteration);
     int goal_linear_inc_end_iter = max_iteration;
     double goal_linear_inc_end_threshold = 0.95;
     double goal_linear_inc = (goal_linear_inc_end_threshold - pick_goal_threshold) / (goal_linear_inc_end_iter - goal_linear_inc_start_iter);
    for (unsigned i=1; i<=max_iteration; i++)
    {
        #ifdef DEBUG
            std::cout << "iteration " << i << std::endl;
            std::cout << "state_t = [" << state_t[0] << ", " << state_t[1] << ", " << state_t[2] << ", " << state_t[3] <<"]" << std::endl;
        #endif
        // given the previous result of bvp, find the next starting point (nearest in the tree)
        //for (unsigned j=0; j < this->state_dim; j++)
        //{
        //    state_t_ptr[j] = state_t[j];
        //}
        //SMP->nearest_state(state_t_ptr, state_t);

        std::vector<double> next_state(this->state_dim);
        double use_goal_prob = uni_distribution(generator);
        // update pick_goal_threshold based on iteration number
        if (i > goal_linear_inc_start_iter)
        {
            pick_goal_threshold += goal_linear_inc;
        }

        if (use_goal_prob <= pick_goal_threshold)
        {
            // sample the goal instead when enough max_iteration is used
            next_state = goal_state;
            flag=0;
        }
        else
        {
            //std::cout << "inside cost sampling" << std::endl;
            flag=1;
            begin_time = clock();

            // state std::vector to tensor
            torch::Tensor state_tensor = getStateTensorWithNormalization(state_t).to(at::Device("cuda:"+std::to_string(this->gpu_device)));
            torch::Tensor goal_tensor = getStateTensorWithNormalization(goal_inform_state).to(at::Device("cuda:"+std::to_string(this->gpu_device)));
            std::cout << "outside of getStateTensorWithNormalization" << std::endl;
            std::cout << "state_tensor: " << state_tensor << std::endl;
            std::cout << "goal_tensor: " << goal_tensor << std::endl;
            std::cout << "num_sample: " << num_sample <<std::endl;
            torch::Tensor state_tensor_expand = state_tensor.repeat({num_sample,1});
            torch::Tensor goal_tensor_expand = goal_tensor.repeat({num_sample,1});
            std::cout << "state_tensor_expand: " << state_tensor_expand << std::endl;
            std::cout << "goal_tensor_expand: " << goal_tensor_expand << std::endl;
            // construct cost_end_state
            torch::Tensor next_tensor_expand = this->tensor_informer(obs_expand_enc, state_tensor_expand, goal_tensor_expand);
            torch::Tensor next_tensor_expand_with_grad = torch::autograd::Variable(next_tensor_expand.clone()).detach().set_requires_grad(true); // add gradient
            std::cout << "before cost_informer...  next_tensor_expand:" << next_tensor_expand_with_grad << std::endl;
            std::cout << "before cost_informer...  goal_tensor_expand:" << goal_tensor_expand << std::endl;
            torch::Tensor cost_tensor_expand = this->tensor_cost_informer(cost_obs_expand_enc, next_tensor_expand_with_grad, goal_tensor_expand);
            std::cout << "before sum...  cost_tensor:" << cost_tensor_expand << std::endl;
            cost_tensor_expand = cost_tensor_expand.sum();
            std::cout << "before backward...  cost_tensor:" << cost_tensor_expand << std::endl;
            cost_tensor_expand.backward();
            std::cout << "after backward." << std::endl;
            torch::Tensor next_tensor_expand_grad = next_tensor_expand_with_grad.grad();
            std::cout << "next_tensor_expand_grad: " << next_tensor_expand_grad << std::endl;
            // perform gradient descent to optimize cost w.r.t. next_state
            next_tensor_expand = next_tensor_expand - 0.1*next_tensor_expand_grad;

            // obtain the cost at the end of optimization
            torch::Tensor cost_tensor_expand_after_grad = this->tensor_cost_informer(cost_obs_expand_enc, next_tensor_expand, goal_tensor_expand);
            next_tensor_expand = next_tensor_expand.to(at::kCPU);
            cost_tensor_expand_after_grad = cost_tensor_expand_after_grad.to(at::kCPU);
            auto cost_tensor_expand_after_grad_a = cost_tensor_expand_after_grad.accessor<float,2>();
            auto next_tensor_expand_a = next_tensor_expand.accessor<float,2>(); // accesor for the tensor
            std::cout << "next_tensor_expand: " << next_tensor_expand << std::endl;

            std::cout << "cost_tensor_expand_after_grad: " << cost_tensor_expand_after_grad << std::endl;

            double best_cost = 100000.;
            int best_ind = -1;
            for (unsigned j=0; j<num_sample; j++)
            {
                //std::cout << "next_State_candidate[j]: [" << next_state_candidate[j] << "]" << std::endl;

                //std::cout << "next_state_cost[j]: " << next_state_cost[j] << std::endl;
                if (cost_tensor_expand_after_grad_a[j][0] < best_cost)
                {
                    best_cost = cost_tensor_expand_after_grad_a[j][0];
                    best_ind = j;
                }
            }
            std::vector<double> next_state_before_unnorm;
            // copy to vector and unnormalize
            for (unsigned j=0; j<this->state_dim; j++)
            {
                next_state_before_unnorm.push_back(next_tensor_expand_a[best_ind][j]);
            }
            unnormalize(next_state_before_unnorm, next_state);
            std::cout << "next_state: " << next_state << std::endl;
            //std::cout << "best_cost: " << best_cost << std::endl;
            //std::cout << "after cost sampling" << std::endl;
            //std::cout << "best_ind: " << best_ind << std::endl;

            //this->informer(obs_enc, state_t, goal_inform_state, next_state);
        #ifdef COUNT_TIME
            std::cout << "informer time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
        #endif
        }
        // according to next_state (MPNet sample), change start state to nearest_neighbors of next_state to
        // use search tree
        //for (unsigned j=0; j < this->state_dim; j++)
        //{
        //    state_t_ptr[j] = next_state[j];
        //}
        //SMP->nearest_state(state_t_ptr, state_t);
        // copy to c++ double* list from std::vector
        for (unsigned j=0; j < this->state_dim; j++)
        {
            state_t_ptr[j] = state_t[j];
            next_state_ptr[j] = next_state[j];
        }
        // below tries to use step_with_sample to imitate DeepSMP
        double new_time = 0.;
        int min_time_steps = 5;
        int max_time_steps = 100;
        SMP->step_with_sample(system, next_state_ptr, from_state, new_state, new_control, new_time, min_time_steps, max_time_steps, 0.02);

        // only when using MPNet, update the state_t using next_state. Otherwise not change
        if (flag)//flag=1: using MPNet.
        {
            if (new_time <= 0.01)
            {
                // propagate fails, back to origin
                state_t = start_state;
            }
            else
            {
                // propagation success
                state_t = next_state; // this using MPNet next sample instead of propagated state
                //for (unsigned j=0; j<this->state_dim; j++)
                //{
                //    state_t[j] = new_state[j];  // this uses propagated state after radom extension
                //}
            }
        }
         // check if solution exists
         SMP->get_solution(res_x, res_u, res_t);

        double total_t = 0.;
        for (unsigned j=0; j<res_t.size(); j++)
        {
            total_t += res_t[j];
        }
        if (res_x.size() != 0 && total_t <= cost_threshold)
        {
            // solved
            delete state_t_ptr;
            delete next_state_ptr;

            delete new_state;
            delete new_control;
            delete from_state;

            return;
        }
    }
    // check if solved
    SMP->get_solution(res_x, res_u, res_t);

    delete state_t_ptr;
    delete next_state_ptr;

    delete new_state;
    delete new_control;
    delete from_state;

    double total_t = 0.;
    for (unsigned j=0; j<res_t.size(); j++)
    {
        total_t += res_t[j];
    }
    if (res_x.size() != 0 && total_t <= cost_threshold)
    {
        // solved
        return;
    }
    else if (res_x.size() != 0)
    {
        // solved but cost not low enough
        res_x.clear();
        res_u.clear();
        res_t.clear();
    }
}
//**********



//**********





//****  step method for visualization of tree_SMP
void MPNetSMP::plan_tree_SMP_step(planner_t* SMP, system_t* system, psopt_system_t* psopt_system, at::Tensor &obs, std::vector<double>& start_state, std::vector<double>& goal_state, std::vector<double>& goal_inform_state,
                    int flag, int max_iteration, double goal_radius, double cost_threshold,
                    std::vector<std::vector<double>>& res_x, std::vector<std::vector<double>>& res_u, std::vector<double>& res_t, std::vector<double>& mpnet_res)
{
    // flag: determine if using goal or not
    // flag=1: using MPNet
    // flag=0: not using MPNet
    std::vector<double> state_t = start_state;
    torch::Tensor obs_tensor = obs.to(at::kCUDA);
    clock_t begin_time;
    //mlp_input_tensor = torch::cat({obs_enc,sg}, 1);

    std::vector<torch::jit::IValue> obs_input;
    obs_input.push_back(obs_tensor);
    at::Tensor obs_enc = encoder->forward(obs_input).toTensor().to(at::kCPU);
    double* state_t_ptr = new double[this->state_dim];
    double* next_state_ptr = new double[this->state_dim];
    double* new_state = new double[this->state_dim];
    double* new_control = new double[this->control_dim];
    double* from_state = new double[this->state_dim];
    //std::cout << "this->psopt_num_iters: " << this->psopt_num_iters << std::endl;
    //int flag=1;  // flag=1: using MPNet
                 // flag=0: not using MPNet
     //double pick_goal_threshold = 0.1;
     //std::uniform_real_distribution<double> uni_distribution(0.0,1.0); // based on this sample goal

    //std::cout << "iteration " << i << std::endl;
    #ifdef DEBUG
        std::cout << "state_t = [" << state_t[0] << ", " << state_t[1] << ", " << state_t[2] << ", " << state_t[3] <<"]" << std::endl;
    #endif
    // given the previous result of bvp, find the next starting point (nearest in the tree)
    //for (unsigned j=0; j < this->state_dim; j++)
    //{
    //    state_t_ptr[j] = state_t[j];
    //}
    //SMP->nearest_state(state_t_ptr, state_t);

    std::vector<double> next_state(this->state_dim);
    if (!flag)
    {
        next_state = goal_state;
        mpnet_res = goal_state;
    }
    else
    {
        begin_time = clock();
        this->informer(obs_enc, state_t, goal_inform_state, next_state);
        mpnet_res = next_state;
    #ifdef COUNT_TIME
        std::cout << "informer time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
    #endif
    }
    // according to next_state (MPNet sample), change start state to nearest_neighbors of next_state to
    // use search tree
    //for (unsigned j=0; j < this->state_dim; j++)
    //{
    //    state_t_ptr[j] = next_state[j];
    //}
    //SMP->nearest_state(state_t_ptr, state_t);
    // copy to c++ double* list from std::vector
    for (unsigned j=0; j < this->state_dim; j++)
    {
        state_t_ptr[j] = state_t[j];
        next_state_ptr[j] = next_state[j];
    }
    // below tries to use step_with_sample to imitate DeepSMP
    double new_time = 0.;
    int min_time_steps = 5;
    int max_time_steps = 100;
    SMP->step_with_sample(system, next_state_ptr, from_state, new_state, new_control, new_time, min_time_steps, max_time_steps, 0.02);
    std::cout << "neural_deep_smp: after step_with_sample, new_time: " << new_time << std::endl;
    // only when using MPNet, update the state_t using next_state. Otherwise not change
    /**
    if (flag)//flag=1: using MPNet.
    {
        if (new_time <= 0.01)
        {
            // propagate fails, back to origin
            state_t = start_state;
        }
        else
        {
            // propagation success
            // state_t = next_state; // this using MPNet next sample instead of propagated state
            state_t = new_state; // this uses propagated state after radom extension
        }
    }
    */
    if (new_time >= 0.02)
    {
        // if success, then return the newly added edge
        std::vector<double> res_x0;
        for (unsigned j=0; j<this->state_dim; j++)
        {
            res_x0.push_back(from_state[j]);
        }
        std::vector<double> res_x1;
        for (unsigned j=0; j<this->state_dim; j++)
        {
            res_x1.push_back(new_state[j]);
        }
        res_x.push_back(res_x0);
        res_x.push_back(res_x1);

        std::vector<double> res_u0;
        for (unsigned j=0; j<this->control_dim; j++)
        {
            res_u0.push_back(new_control[j]);
        }
        res_u.push_back(res_u0);

        res_t.push_back(new_time);
    }

    delete state_t_ptr;
    delete next_state_ptr;

    delete new_state;
    delete new_control;
    delete from_state;
}


//****


//**** step method with cost_to_go for visualization of tree_SMP
//****
//****  step method for visualization of tree_SMP
void MPNetSMP::plan_tree_SMP_cost_step(planner_t* SMP, system_t* system, psopt_system_t* psopt_system, at::Tensor &obs, std::vector<double>& start_state, std::vector<double>& goal_state, std::vector<double>& goal_inform_state,
                    int flag, int max_iteration, double goal_radius, double cost_threshold,
                    std::vector<std::vector<double>>& res_x, std::vector<std::vector<double>>& res_u, std::vector<double>& res_t, std::vector<double>& mpnet_res)
{
    // flag: determine if using goal or not
    // flag=1: using MPNet
    // flag=0: not using MPNet
    std::vector<double> state_t = start_state;
    torch::Tensor obs_tensor = obs.to(at::kCUDA);
    clock_t begin_time;
    //mlp_input_tensor = torch::cat({obs_enc,sg}, 1);

    std::vector<torch::jit::IValue> obs_input;
    obs_input.push_back(obs_tensor);
    at::Tensor obs_enc = encoder->forward(obs_input).toTensor().to(at::kCPU);
    at::Tensor cost_obs_enc = cost_encoder->forward(obs_input).toTensor().to(at::kCPU);
    double* state_t_ptr = new double[this->state_dim];
    double* next_state_ptr = new double[this->state_dim];
    double* new_state = new double[this->state_dim];
    double* new_control = new double[this->control_dim];
    double* from_state = new double[this->state_dim];
    //std::cout << "this->psopt_num_iters: " << this->psopt_num_iters << std::endl;
    //int flag=1;  // flag=1: using MPNet
                 // flag=0: not using MPNet
     //double pick_goal_threshold = 0.1;
     //std::uniform_real_distribution<double> uni_distribution(0.0,1.0); // based on this sample goal

    //std::cout << "iteration " << i << std::endl;
    #ifdef DEBUG
        std::cout << "state_t = [" << state_t[0] << ", " << state_t[1] << ", " << state_t[2] << ", " << state_t[3] <<"]" << std::endl;
    #endif
    // given the previous result of bvp, find the next starting point (nearest in the tree)
    //for (unsigned j=0; j < this->state_dim; j++)
    //{
    //    state_t_ptr[j] = state_t[j];
    //}
    //SMP->nearest_state(state_t_ptr, state_t);

    std::vector<double> next_state(this->state_dim);
    if (!flag)
    {
        // picking goal
        next_state = goal_state;
        mpnet_res = goal_state;
    }
    else
    {
        flag=1;
        begin_time = clock();
        // first sample several mpnet points, then use the costnet to find the best point
        int num_sample = 15;
        std::vector<std::vector<double>> next_state_candidate(num_sample,std::vector<double>(this->state_dim));
        std::vector<double> next_state_cost(num_sample);
        std::vector<std::vector<double>> cost_end_state(num_sample,std::vector<double>(this->state_dim));
        // construct cost_end_state
        for (unsigned j=0; j<num_sample; j++)
        {
            cost_end_state[j] = goal_inform_state;
        }
        this->informer_batch(obs_enc, state_t, goal_inform_state, next_state_candidate, num_sample);
        // calculate cost
        this->cost_informer_batch(cost_obs_enc, next_state_candidate, cost_end_state, next_state_cost, num_sample);

        double best_cost = 100000.;
        int best_ind = -1;
        for (unsigned j=0; j<num_sample; j++)
        {
            if (next_state_cost[j] < best_cost)
            {
                best_cost = next_state_cost[j];
                best_ind = j;
            }
        }
        next_state = next_state_candidate[best_ind];
        //std::cout << "best_cost: " << best_cost << std::endl;

        //this->informer(obs_enc, state_t, goal_inform_state, next_state);

        mpnet_res = next_state;
    #ifdef COUNT_TIME
        std::cout << "informer time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
    #endif
    }
    // according to next_state (MPNet sample), change start state to nearest_neighbors of next_state to
    // use search tree
    //for (unsigned j=0; j < this->state_dim; j++)
    //{
    //    state_t_ptr[j] = next_state[j];
    //}
    //SMP->nearest_state(state_t_ptr, state_t);
    // copy to c++ double* list from std::vector
    for (unsigned j=0; j < this->state_dim; j++)
    {
        state_t_ptr[j] = state_t[j];
        next_state_ptr[j] = next_state[j];
    }
    // below tries to use step_with_sample to imitate DeepSMP
    double new_time = 0.;
    int min_time_steps = 5;
    int max_time_steps = 100;
    SMP->step_with_sample(system, next_state_ptr, from_state, new_state, new_control, new_time, min_time_steps, max_time_steps, 0.02);
    std::cout << "neural_deep_smp: after step_with_sample, new_time: " << new_time << std::endl;
    // only when using MPNet, update the state_t using next_state. Otherwise not change
    /**
    if (flag)//flag=1: using MPNet.
    {
        if (new_time <= 0.01)
        {
            // propagate fails, back to origin
            state_t = start_state;
        }
        else
        {
            // propagation success
            // state_t = next_state; // this using MPNet next sample instead of propagated state
            state_t = new_state; // this uses propagated state after radom extension
        }
    }
    */
    if (new_time >= 0.02)
    {
        // if success, then return the newly added edge
        std::vector<double> res_x0;
        for (unsigned j=0; j<this->state_dim; j++)
        {
            res_x0.push_back(from_state[j]);
        }
        std::vector<double> res_x1;
        for (unsigned j=0; j<this->state_dim; j++)
        {
            res_x1.push_back(new_state[j]);
        }
        res_x.push_back(res_x0);
        res_x.push_back(res_x1);

        std::vector<double> res_u0;
        for (unsigned j=0; j<this->control_dim; j++)
        {
            res_u0.push_back(new_control[j]);
        }
        res_u.push_back(res_u0);

        res_t.push_back(new_time);
    }

    delete state_t_ptr;
    delete next_state_ptr;

    delete new_state;
    delete new_control;
    delete from_state;
}




void MPNetSMP::plan_step(planner_t* SMP, system_t* system, psopt_system_t* psopt_system, at::Tensor &obs, std::vector<double>& start_state, std::vector<double>& goal_state, std::vector<double>& goal_inform_state,
                    int flag, int max_iteration, double goal_radius,
                    std::vector<std::vector<double>>& res_x, std::vector<std::vector<double>>& res_u, std::vector<double>& res_t, std::vector<double>& mpnet_res)
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
    //std::cout << "this->psopt_num_iters: " << this->psopt_num_iters << std::endl;

    // given the previous result of bvp, find the next starting point (nearest in the tree)
    //for (unsigned j=0; j < this->state_dim; j++)
    //{
    //    state_t_ptr[j] = state_t[j];
        //std::cout << "state_t_ptr[" << j << "]: " << state_t_ptr[j] << std::endl;
        //std::cout << "state_t[" << j << "]: " << state_t[j] << std::endl;

    //}
    //SMP->nearest_state(state_t_ptr, state_t);

    // randomly sample and find nearest_state as BVP starting point
    SMP->random_state(state_t_ptr); // random sample
    // find nearest_neighbor of random sample state_t_ptr, and assign to state_t
    SMP->nearest_state(state_t_ptr, state_t);


    std::vector<double> next_state(this->state_dim);
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
    //std::cout << "before informer" << std::endl;

    if (!flag)
    {
        // use goal
        next_state = goal_state;
        mpnet_res = next_state;
    }
    else
    {
        begin_time = clock();
        this->informer(obs_enc, state_t, goal_inform_state, next_state);
        mpnet_res = next_state;
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
    // *** below is using bvp solver
    SMP->step_bvp(system, psopt_system, res, state_t_ptr, next_state_ptr, this->psopt_num_iters, this->psopt_num_steps, this->psopt_step_sz,
                 init_traj.x, init_traj.u, init_traj.t);
    std::cout << "step_bvp time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;


    res_x = res.x;
    res_u = res.u;
    res_t = res.t;

}
