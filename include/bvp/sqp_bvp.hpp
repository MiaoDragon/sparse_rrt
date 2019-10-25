/**
 * @file sqp_bvp.hpp
 *
 * @authors: Yinglong Miao
 *
 * @description:
 * Implementation of an approximate BVP (Boundary Value Problem) solver
 * using SQP (Squential Quadratic Programming) from trajopt library. Adapted
 * Specifically for SST system Implementation.
 */
#ifndef SQP_BVP_HPP
#define SQP_BVP_HPP

#include "sco/modeling.hpp"
#include "sco/num_diff.hpp"
#include "sco/sco_fwd.hpp"
#include "systems/system.hpp"
#include <Eigen/Dense>
#include "bvp/cost.hpp"
#include "bvp/constraint.hpp"
using namespace sco;
using namespace Eigen;


class SQPBVP {
public:
  SQPBVP(system_interface* system, int state_dim_in, int control_dim_in, int n_steps, double integration_step);
  ~SQPBVP();
  std::vector<double> solve(const VectorXd& start, const VectorXd& goal) const;
protected:
  CostWithSystem* costPtr;
  ConstraintWithSystem* constraintPtr;
  system_interface* _system;
  int _n_steps;  // store how many intervals are needed
  int state_dim, control_dim;
  double _integration_step;
};

#endif
