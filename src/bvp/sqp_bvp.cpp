/**
 * @file pendulum.hpp
 * @authors: Yinglong Miao
 *
 * @description:
 * Implementation of an approximate BVP (Boundary Value Problem) solver
 * using SQP (Squential Quadratic Programming) from trajopt library. Adapted
 * Specifically for SST system Implementation.
 */
 #include "bvp/sqp_bvp.hpp"
 #include <Eigen/Dense>
 using namespace Eigen;

 CostWithSystem::CostWithSystem(system_t* system, int n_steps)
    : ScalarOfVector()
    , _system(system)
    , _n_steps(n_steps)
    , state_dim(system->get_state_dimension())
    , action_dim(system->get_control_dimension())
 {}
 CostWithSystem::~CostWithSystem()
 {}
 CostWithSystem::operator()(const VectorXd& x) const
 {
   // from system, obtain state space dim, action space dim
   int state_dim = _system->get_state_dimension();
   int action_dim = _system->get_control_dimension();
   // x: state (n_steps*state_dim) | control ((n_steps-1)*control_dim) | duration (n_steps-1)
   double sum_cost = 0.;
   int control_start = this->_n_steps*state_dim;
   int duration_start = control_start + (this->_n_steps-1)*control_dim;
   for (unsigned i=0; i < this->_n_steps; i+=1)
   {
     // calculate single-step cost
     // eigen::seq returns [a,b]
     sum_cost += single_cost_dt(x(seq(i*state_dim,(i+1)*state_dim-1)),
                                x(seq()),
                                x())
   }
 }
