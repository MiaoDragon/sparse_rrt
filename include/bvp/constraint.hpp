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
#include "systems/system.hpp"
#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
using namespace sco;
using namespace Eigen;

typedef boost::shared_ptr<system_interface> SystemPtr;


class ConstraintWithSystem : public VectorOfVector {
public:
  ConstraintWithSystem(system_interface* system, int state_dim_in, int control_dim_in, int n_steps, double integration_step);  // store pointer to the system for further usage
  VectorXd operator()(const VectorXd& x) const;  // const function: no modification of the class members is possible
  ~ConstraintWithSystem();
  void set_start_state(const VectorXd& x);
  void set_end_state(const VectorXd& x);
protected:
  SystemPtr _system;  // pointer to the system interface
  int _n_steps;  // store how many intervals are needed
  int state_dim, control_dim;
  double _integration_step;
  VectorXd start_x;
  VectorXd end_x;
  virtual double dynamic_constraint(const VectorXd& x, const VectorXd& u, const double dt, const VectorXd& x_) const;
  // x_ denotes the next state
  virtual double start_constraint(const VectorXd& x) const;
  virtual double term_constraint(const VectorXd& x) const;
  virtual double start_dynamics(const VectorXd& x, const VectorXd& u, const double dt, const VectorXd& x_) const;
  virtual double term_dynamics(const VectorXd& x, const VectorXd& u, const double dt, const VectorXd& x_) const;

  virtual double time_min_constraint(const double dt) const;
  virtual double time_max_constraint(const double dt) const;
};

class ConstraintWithSystemGoalFree : public ConstraintWithSystem {
public:
    ConstraintWithSystemGoalFree(system_interface* system, int state_dim_in, int control_dim_in, int n_steps, double integration_step)
        : ConstraintWithSystem(system, state_dim_in, control_dim_in, n_steps, integration_step)
    {}
protected:
    double term_constraint(const VectorXd& x) const override;
    double start_constraint(const VectorXd& x) const override;
    double start_dynamics(const VectorXd& x, const VectorXd& u, const double dt, const VectorXd& x_) const override;

};
#endif
