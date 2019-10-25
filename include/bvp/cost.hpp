/**
 * @file cost.hpp
 *
 * @authors: Yinglong Miao
 *
 * @description:
 * Implementation of a cost function class that can be used to serve in the optimization
 * problem.
 */
#ifndef COST_HPP
#define COST_HPP

#include "sco/modeling.hpp"
#include "sco/num_diff.hpp"
#include "systems/system.hpp"
#include <Eigen/Dense>
using namespace sco;
using namespace Eigen;

class CostWithSystem : public ScalarOfVector {
public:
  CostWithSystem(system_interface* system, int state_dim_in, int action_dim_in, int n_steps, double integration_step);  // store pointer to the system for further usage
  double operator()(const VectorXd& x) const;  // const function: no modification of the class members is possible
  // void set_integration_var(const VectorXd& kx1, const VectorXd& kx2, const VectorXd& kx3, const VectorXd& kx4);
  ~CostWithSystem();
  void set_start_state(const VectorXd& x);
  void set_end_state(const VectorXd& x);
protected:
  system_interface* _system;  // pointer to the system interface
  int _n_steps;  // store how many intervals are needed
  int state_dim, action_dim;
  double _integration_step;
  VectorXd start_x;  // to be set before calling operator, otherwise the end-point constraints might be wrong
  VectorXd end_x;
  // for Punge-Kutta-4 integration
  // VectorXd _kx1, _kx2, _kx3, _kx4;  // can't be modified in operator()
  virtual double single_cost_dt(const VectorXd& x, const VectorXd& u, const double dt) const = 0;
  virtual double single_cost(const VectorXd& x, const VectorXd& u) const = 0;
  virtual double start_cost(const VectorXd& x) const = 0; // useful if we want to enforce cost on start state instead of constraint
  virtual double term_cost(const VectorXd& x) const = 0;  // useful if we want to enforce cost on terminal state instead of constraint
};

#endif
