/**
 * @file cost.cpp
 *
 * @authors: Yinglong Miao
 *
 * @description:
 * Implementation of a cost function class that can be used to serve in the optimization
 * problem.
 */
 #include "bvp/cost.hpp"
 #include <Eigen/Dense>
 #include <Eigen/Core>
 using namespace Eigen;

 /* CostWithSystem class */
 CostWithSystem::CostWithSystem(system_interface* system, int state_dim_in, int control_dim_in, int n_steps, double integration_step)
    : ScalarOfVector()
    , _n_steps(n_steps)
    , state_dim(state_dim_in)
    , control_dim(control_dim_in)
    , _integration_step(integration_step)
 {
     _system.reset(system);
     start_x = VectorXd::Zero(state_dim_in);
     end_x = VectorXd::Zero(state_dim_in);
 }

 CostWithSystem::~CostWithSystem()
 {
     _system.reset();
 }

 double CostWithSystem::operator()(const VectorXd& x) const
 {
   // x: state (n_steps*state_dim) | control ((n_steps-1)*control_dim) | duration (n_steps-1)
   double sum_cost = 0.;
   int control_start = _n_steps*state_dim;
   int duration_start = control_start + (_n_steps-1)*control_dim;
   for (unsigned i=0; i < _n_steps-1; i+=1)
   {
     // calculate single-step cost
     // eigen::seq returns [a,b]
     sum_cost += single_cost_dt(x.segment(i*state_dim,state_dim),
                                x.segment(control_start+i*control_dim, control_dim),
                                x(duration_start+i));
   }
   sum_cost += term_cost(x.segment(control_start-state_dim,state_dim));
   return sum_cost;
 }

 double CostWithSystem::single_cost_dt(const VectorXd& x, const VectorXd& u, const double dt) const
 {
     /**
     *  This uses Runge-Kutta-4 integration to calculate single step cost
     */

     // compute x_k1, x_k2, x_k3 by using system dynamics
     /**
     *
     propagate(
         const double* start_state, unsigned int state_dimension,
         const double* control, unsigned int control_dimension,
         int num_steps, double* result_state, double integration_step)
     */
     VectorXd x_k1(state_dim);
     VectorXd x_k2(state_dim);
     VectorXd x_k3(state_dim);
     VectorXd x_k4(state_dim);
     VectorXd temp(state_dim);
     double* _x_k1 = new double[state_dim];
     double* _x_k2 = new double[state_dim];
     double* _x_k3 = new double[state_dim];
     double* _x_k4 = new double[state_dim];

     _system->propagate(x.data(), state_dim, u.data(), control_dim,
                        1, _x_k1, _integration_step);
     // calculate x_k1 from _x_k1
     // formula: x_k1 = dt * _x_k1
     for (unsigned i=0; i<state_dim; i++)
     {
         x_k1(i) = _x_k1[i] * dt;
     }
     // calculate _x_k2 from x_k1
     // formula: _x_k2 = f(x+x_k1/2, u)
     temp = x+x_k1/2;
     _system->propagate(temp.data(), state_dim, u.data(), control_dim,
                        1, _x_k2, _integration_step);
     // calculate x_k2 from _x_k2
     // formula: x_k2 = dt * _x_k2
     for (unsigned i=0; i<state_dim; i++)
     {
         x_k2(i) = _x_k2[i] * dt;
     }
     //calculate _x_k3 from x_k2
     // formula: _x_k3 = f(x+x_k2/2, u)
     temp = x+x_k2/2;
     _system->propagate(temp.data(), state_dim, u.data(), control_dim,
                        1, _x_k3, _integration_step);
     // calculate x_k3 from _x_k3
     // formula: x_k3 = dt * _x_k3
     for (unsigned i=0; i<state_dim; i++)
     {
         x_k3(i) = _x_k3[i] * dt;
     }
     // calculate _x_k4 from x_k3
     // formula: _x_k4 = f(x+x_k3, u)
     temp = x+x_k3;
     _system->propagate(temp.data(), state_dim, u.data(), control_dim,
                        1, _x_k4, _integration_step);
     // calculate x_k4 from _x_k4
     // formula: x_k4 = dt * _x_k4
     for (unsigned i=0; i<state_dim; i++)
     {
         x_k4(i) = _x_k4[i] * dt;
     }
     // from state, calculate loss
     double l_k1, l_k2, l_k3, l_k4;
     // calculate l_k1
     // formula: l_k1 = dt * l(x,u)
     l_k1 = dt * single_cost(x, u);
     // calculate l_k2
     // formula: l_k2 = dt * l(x+x_k1/2, u)
     l_k2 = dt * single_cost(x+x_k1/2, u);
     // calculate l_k3
     // formula: l_k3 = dt * l(x+x_k2/2, u)
     l_k3 = dt * single_cost(x+x_k2/2, u);
     // calculate l_k4
     // formula: l_k4 = dt * l(x+x_k3, u)
     l_k4 = dt * single_cost(x+x_k3, u);
     // calculate l
     // formula: l = (l_k1 + 2*l_k2 + 2*l_k3 + l_k4) / 6
     double res = (l_k1 + 2*l_k2 + 2*l_k3 + l_k4) / 6;
     //*****
     // to delete array, use delete[]
     // For debugging
     //*****
     delete[] _x_k1;
     delete[] _x_k2;
     delete[] _x_k3;
     delete[] _x_k4;
     return res;
 }


 double CostWithSystem::single_cost(const VectorXd& x, const VectorXd& u) const
 {
     // for this problem, we are using minimum time, and hence the cost is 1.0
     return 1.0;
 }

 double CostWithSystem::start_cost(const VectorXd& x) const
 {
     // we don't have any preference for the start state. It is enforced by constraint
     return 0.;
 }

 double CostWithSystem::term_cost(const VectorXd& x) const
 {
     // we don't have any preference for the terminal state. It is enforced by constraint
     return 0.;
 }

void CostWithSystem::set_start_state(const VectorXd& x)
{
    // not sure if there is a more elegant way of copying
    for (unsigned i=0; i < x.size(); i++)
    {
        start_x(i) = x(i);
    }
}
void CostWithSystem::set_end_state(const VectorXd& x)
{
    // not sure if there is a more elegant way of copying
    for (unsigned i=0; i < x.size(); i++)
    {
        end_x(i) = x(i);
    }
}


/** CostWithSystemGoal class */
double CostWithSystemGoal::term_cost(const VectorXd& x) const
{
    return (x - end_x).squaredNorm();
}
