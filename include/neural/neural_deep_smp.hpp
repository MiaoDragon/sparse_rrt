#ifndef _DEEP_SMP_
#define _DEEP_SMP_

#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include <math.h>       /* floor */
#include <stdlib.h>     /* srand, rand */
#include <random>

#define _USE_MATH_DEFINES


using namespace ompl;
typedef std::vector<ompl::base::State *> StatePtrVec;
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
             int num_iter_in, int num_steps_in, double step_sz_in,
             system_t& system_in, psopt_system_t& psopt_system_in,  //TODO: add clone to make code more secure
             );
    void plan(planner_t& SMP, at::Tensor obs, std::vector<double> start_state, std::vector<double> goal_state, int max_iteration, double goal_radius,
              std::vector<std::vector<double>> res_x, std::vector<std::vector<double>> res_u, std::vector<double> res_t);
    ~MPNetSMP();

protected:
    /** \brief Representation of a motion
        This only contains pointers to parent motions as we
        only need to go backwards in the tree. */
    int _max_replan;
    int _max_length;
    int state_dim, control_dim;
    int num_steps;
    double step_sz;
    //at::Tensor obs_enc; // two dimensional or one dimensional
    std::shared_ptr<torch::jit::script::Module> encoder;
    std::shared_ptr<torch::jit::script::Module> MLP;
    std::unique_ptr<planner_t> SMP;  // the underlying SMP module
    std::shared_ptr<system_t> system;
    std::shared_ptr<psopt_system_t> psopt_system;
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
    virtual void init_informer(at::Tensor obs, const std::vector<double>& start_state, const std::vector<double>& goal_state, traj_j& res);
    virtual void normalize(const std::vector<double>& state, std::vector<double>& res);
    virtual void unnormalize(const std::vector<double>& state, std::vector<double>& res);
    torch::Tensor getStartGoalTensor(const std::vector<double>& start_state, const std::vector<double>& goal_state);
    void lvc(StatePtrVec& path, StatePtrVec& res);

};


#endif
