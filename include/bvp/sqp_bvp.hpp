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
 #ifndef BVP_HPP
 #define BVP_HPP
 #include "sco/modeling.hpp"
 #include "sco/num_diff.hpp"
 #include "systems/system.h"
 #include <Eigen/Dense>
 using namespace sco;
 using namespace Eigen;

 class CostWithSystem : public ScalarOfVector {
 public:
   CostWithSystem(system_t* system, int n_steps);  // store pointer to the system for further usage
   double operator()(const VectorXd& x) const = 0;  // const function: no modification of the class members is possible
   // void set_integration_var(const VectorXd& kx1, const VectorXd& kx2, const VectorXd& kx3, const VectorXd& kx4);
   ~CostWithSystem();
 protected:
   system_t* _system;  // pointer to the system interface
   int _n_steps;  // store how many intervals are needed
   int _state_dim, _action_dim;
   // for Punge-Kutta-4 integration
   // VectorXd _kx1, _kx2, _kx3, _kx4;  // can't be modified in operator()
   virtual double single_cost_dt(const VectorXd& x, const VectorXd& u, const double dt) const = 0;
   virtual double single_cost(const VectorXd& x, const VectorXd& u) const = 0;
 };

 class ConstraintWithSystem : public VectorOfVector {
 public:
   ConstraintWithSystem(system_t* system, int n_steps);  // store pointer to the system for further usage
   VetorXd operator()(const VectorXd& x) const = 0;  // const function: no modification of the class members is possible
   ~ConstraintWithSystem();
 protected:
   system_t* _system;  // pointer to the system interface
   int _n_steps;  // store how many intervals are needed
   int state_dim, action_dim;
   virtual double single_prop_dt(const VectorXd& x, const VectorXd& u, const double dt) const = 0;
 };

 class SQPBVP {
 public:
   SQPBVP(system_t* system, int n_steps);
   double solve(const VectorXd& start, const VectorXd& goal) const = 0;
 protected:
   CostWithSystem cost;
   ConstraintWithSystem constraint;
 }

 #endif
