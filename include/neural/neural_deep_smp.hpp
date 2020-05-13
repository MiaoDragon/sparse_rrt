#ifndef _DEEP_SMP_
#define _DEEP_SMP_

#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include <math.h>       /* floor */
#include <stdlib.h>     /* srand, rand */
#include <random>
#include <memory>


#include "systems/system.hpp"
#include "bvp/psopt_system.hpp"
#include "motion_planners/planner.hpp"
#define _USE_MATH_DEFINES


struct traj_t
{
    std::vector<std::vector<double>> x;  // (T x X)
    std::vector<std::vector<double>> u;  // (T x U)
    std::vector<double> t;
};

class MPNetSMP
{
public:
    /** \brief Constructor */
    MPNetSMP(std::string mlp_path, std::string encoder_path,
             std::string cost_mlp_path, std::string cost_encoder_path,
             system_t* system,
             int num_iters_in, int num_steps_in, double step_sz_in
             );
    void plan_tree(planner_t* SMP, system_t* system, psopt_system_t* psopt_system, at::Tensor &obs, std::vector<double>& start_state, std::vector<double>& goal_state, std::vector<double>& goal_inform_state,
              int max_iteration, double goal_radius,
              std::vector<std::vector<double>>& res_x, std::vector<std::vector<double>>& res_u, std::vector<double>& res_t);
    void plan_line(planner_t* SMP, system_t* system, psopt_system_t* psopt_system, at::Tensor &obs, std::vector<double>& start_state, std::vector<double>& goal_state, std::vector<double>& goal_inform_state,
              int max_iteration, double goal_radius,
              std::vector<std::vector<double>>& res_x, std::vector<std::vector<double>>& res_u, std::vector<double>& res_t);
    void plan_tree_SMP(planner_t* SMP, system_t* system, psopt_system_t* psopt_system, at::Tensor &obs, std::vector<double>& start_state, std::vector<double>& goal_state, std::vector<double>& goal_inform_state,
                        int max_iteration, double goal_radius, double cost_threshold,
                        std::vector<std::vector<double>>& res_x, std::vector<std::vector<double>>& res_u, std::vector<double>& res_t);
    void plan_tree_SMP_hybrid(planner_t* SMP, system_t* system, psopt_system_t* psopt_system, at::Tensor &obs, std::vector<double>& start_state, std::vector<double>& goal_state, std::vector<double>& goal_inform_state,
                        int max_iteration, double goal_radius, double cost_threshold,
                        std::vector<std::vector<double>>& res_x, std::vector<std::vector<double>>& res_u, std::vector<double>& res_t);
    void plan_tree_SMP_cost(planner_t* SMP, system_t* system, psopt_system_t* psopt_system, at::Tensor &obs, std::vector<double>& start_state, std::vector<double>& goal_state, std::vector<double>& goal_inform_state,
                        int max_iteration, double goal_radius, double cost_threshold,
                        std::vector<std::vector<double>>& res_x, std::vector<std::vector<double>>& res_u, std::vector<double>& res_t);
    void plan_tree_SMP_step(planner_t* SMP, system_t* system, psopt_system_t* psopt_system, at::Tensor &obs, std::vector<double>& start_state, std::vector<double>& goal_state, std::vector<double>& goal_inform_state,
                        int flag, int max_iteration, double goal_radius, double cost_threshold,
                        std::vector<std::vector<double>>& res_x, std::vector<std::vector<double>>& res_u, std::vector<double>& res_t, std::vector<double>& mpnet_res);

    void plan_tree_SMP_cost_step(planner_t* SMP, system_t* system, psopt_system_t* psopt_system, at::Tensor &obs, std::vector<double>& start_state, std::vector<double>& goal_state, std::vector<double>& goal_inform_state,
                        int flag, int max_iteration, double goal_radius, double cost_threshold,
                        std::vector<std::vector<double>>& res_x, std::vector<std::vector<double>>& res_u, std::vector<double>& res_t, std::vector<double>& mpnet_res);

    void plan_step(planner_t* SMP, system_t* system, psopt_system_t* psopt_system, at::Tensor &obs, std::vector<double>& start_state, std::vector<double>& goal_state, std::vector<double>& goal_inform_state,
              int flag, int max_iteration, double goal_radius,
              std::vector<std::vector<double>>& res_x, std::vector<std::vector<double>>& res_u, std::vector<double>& res_t, std::vector<double>& mpnet_res);
              // return bvp result
    ~MPNetSMP();

protected:
    /** \brief Representation of a motion
        This only contains pointers to parent motions as we
        only need to go backwards in the tree. */
    int _max_replan;
    int _max_length;
    int state_dim, control_dim;
    double psopt_step_sz;
    int psopt_num_iters, psopt_num_steps;
    //at::Tensor obs_enc; // two dimensional or one dimensional
    std::shared_ptr<torch::jit::script::Module> encoder;
    std::shared_ptr<torch::jit::script::Module> MLP;
    std::shared_ptr<torch::jit::script::Module> cost_encoder;
    std::shared_ptr<torch::jit::script::Module> cost_MLP;
    std::unique_ptr<planner_t> SMP;  // the underlying SMP module
    std::vector<double> lower_bound;
    std::vector<double> upper_bound;
    std::vector<double> bound;
    std::vector<double> control_lower_bound;
    std::vector<double> control_upper_bound;
    std::vector<bool> is_circular;
    // random generator
    std::default_random_engine generator;
    // MPNet specific:
    virtual void informer(at::Tensor obs, const std::vector<double>& start_state, const std::vector<double>& goal_state, std::vector<double>& next_state);
    virtual void init_informer(at::Tensor obs, const std::vector<double>& start_state, const std::vector<double>& goal_state, traj_t& res);
    virtual void cost_informer(at::Tensor obs, const std::vector<double>& start_state, const std::vector<double>& goal_state, double& cost);
    virtual void informer_batch(at::Tensor obs, const std::vector<double>& start_state, const std::vector<double>& goal_state, std::vector<std::vector<double>>& next_state, int num_sample);
    virtual void cost_informer_batch(at::Tensor obs, const std::vector<double>& start_state, const std::vector<double>& goal_state, std::vector<double>& cost, int num_sample);

    virtual void normalize(const std::vector<double>& state, std::vector<double>& res);
    virtual void unnormalize(const std::vector<double>& state, std::vector<double>& res);
    torch::Tensor getStartGoalTensor(const std::vector<double>& start_state, const std::vector<double>& goal_state);

};


#endif
