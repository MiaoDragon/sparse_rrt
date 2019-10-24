/**
 * @file constraint.hpp
 *
 * @authors: Yinglong Miao
 *
 * @description:
 * Implementation of a constraint function class that can be used to serve in the optimization
 * problem.
 */
#ifndef CONSTRAINT_HPP
#define CONSTRAINT_HPP

#include "sco/modeling.hpp"
#include "sco/num_diff.hpp"
#include "systems/system.h"
#include <Eigen/Dense>
using namespace sco;
using namespace Eigen;

class ConstraintWithSystem : public VectorOfVector {
public:
  ConstraintWithSystem(system_t* system, int n_steps, double integration_step);  // store pointer to the system for further usage
  VetorXd operator()(const VectorXd& x) const;  // const function: no modification of the class members is possible
  ~ConstraintWithSystem();
  void set_start_state(const VectorXd& x);
  void set_end_state(const VectorXd& x);
protected:
  system_t* _system;  // pointer to the system interface
  int _n_steps;  // store how many intervals are needed
  int state_dim, action_dim;
  double _integration_step;
  VectorXd start_x;
  VectorXd end_x;
  virtual double dynamic_constraint(const VectorXd& x, const VectorXd& u, const double dt, const VectorXd& x_) const = 0;
  // x_ denotes the next state
  virtual double start_constraint(const VectorXd& x) const = 0;
  virtual double term_constraint(const VectorXd& x) const = 0;
  virtual double time_min_constraint(const double dt) const = 0;
  virtual double time_max_constraint(const double dt) const = 0;
};
