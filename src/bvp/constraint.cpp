/**
 * @file constraint.hpp
 *
 * @authors: Yinglong Miao
 *
 * @description:
 * Implementation of a constraint function class that can be used to serve in the optimization
 * problem.
 */

#include "bvp/constraint.hpp"
#include <Eigen/Dense>
#include <Eigen/Core>
using namespace Eigen;

ConstraintWithSystem::ConstraintWithSystem(system_interface* system, int state_dim_in, int control_dim_in, int n_steps, double integration_step)
: VectorOfVector()
, _n_steps(n_steps)
, state_dim(state_dim_in)
, control_dim(control_dim_in)
, _integration_step(integration_step)
{
    _system.reset(system);
    start_x = VectorXd::Zero(state_dim_in);
    end_x = VectorXd::Zero(state_dim_in);
}

ConstraintWithSystem::~ConstraintWithSystem()
{
    _system.reset();
}

VectorXd ConstraintWithSystem::operator()(const VectorXd& x) const
{
    /**
    * This returns how much the constraints are violated, i.e. the error for each constraint.
    * For LHS<=RHS: it is LHS - RHS
    * For LHS>=RHS: it is RHS - LHS
    * For LHS==RHS: it is ||LHS-RHS||  (norm can be custom, in the library l1 norm seems to be used)
    * Constraint:
    *   Dynamic constraint (n_steps-1), start state constraint, end state constraint, time min constraint (n_steps-1),
    *     time max constraint (n_steps-1)
    *   Possibly can add other constraints in the future
    * -return:
    *     constraint error:  Dynamic constraint | start constraint | end constraint | time min constraint | time max constraint
    */
    // x: state (n_steps*state_dim) | control ((n_steps-1)*control_dim) | duration (n_steps-1)
    double sum_cost = 0.;
    int control_start = _n_steps*state_dim;
    int duration_start = control_start + (_n_steps-1)*control_dim;
    // number of Dynamic Constraints: n_steps
    VectorXd errs(3*_n_steps-1);  // constraints errors to return
    for (unsigned i=0; i < _n_steps-1; i+=1)
    {
      // eigen::seq returns [a,b]
      // handle Dynamic Constraints
      errs(i) = dynamic_constraint(x.segment(i*state_dim,state_dim),
                                   x.segment(control_start+i*control_dim, control_dim),
                                   x(duration_start+i),
                                   x.segment((i+1)*state_dim,state_dim));
      // handle Time Constraint
      errs(_n_steps+1+i) = time_min_constraint(x(duration_start+i));
      errs(2*_n_steps+i) = time_max_constraint(x(duration_start+i));
    }
    // handle start constraint
    errs(_n_steps-1) = start_constraint(x.segment(0,state_dim));
    // handle terminal constraint
    errs(_n_steps) = term_constraint(x.segment(control_start-state_dim, state_dim));
    return errs;
}

double ConstraintWithSystem::dynamic_constraint(const VectorXd& x, const VectorXd& u, const double dt, const VectorXd& x_) const
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
    // calculate x_dynamics (x(t+1) from dynamics)
    // formula: x_dynamics = x + (x_k1 + 2*x_k2 + 2*x_k3 + x_k4) / 6
    VectorXd x_dynamics = x + (x_k1 + 2*x_k2 + 2*x_k3 + x_k4) / 6;
    // equality constraint between x_dynamics and x_
    //*************************************************************
    /*** Note:
    *   This norm can be an arbitrary norm that the user specifies.
    *   Here we use l2 norm for simplicity
    */
    /** TODO:
    *  May add small disturbance so that we don't strictly enforce the constraint
    */
    delete[] _x_k1;
    delete[] _x_k2;
    delete[] _x_k3;
    delete[] _x_k4;

    return (x_dynamics - x_).squaredNorm();
}
double ConstraintWithSystem::start_constraint(const VectorXd& x) const
{
    // Here we try adding constraints to both ends.
    // This behavior can be modified. For instance, in Bidirectional case
    //*************************************************************
    /*** Note:
    *   This norm can be an arbitrary norm that the user specifies.
    *   Here we use l2 norm for simplicity
    */
    // again, other norms can be used
    return (x - start_x).squaredNorm();
}

double ConstraintWithSystem::term_constraint(const VectorXd& x) const
{
    // Here we try adding constraints to both ends.
    // This behavior can be modified. For instance, in Bidirectional case
    //*************************************************************
    /*** Note:
    *   This norm can be an arbitrary norm that the user specifies.
    *   Here we use l2 norm for simplicity
    */
    // again, other norms can be used
    return (x - end_x).squaredNorm();
}

double ConstraintWithSystem::time_min_constraint(const double dt) const
{
    // constraint: dt>=0
    return -dt;
}

double ConstraintWithSystem::time_max_constraint(const double dt) const
{
    // we don't set max time constraint here, because our cost is total time
    return 0.;
}

void ConstraintWithSystem::set_start_state(const VectorXd& x)
{
    // not sure if there is a more elegant way of copying
    for (unsigned i=0; i < x.size(); i++)
    {
        start_x(i) = x(i);
    }
}
void ConstraintWithSystem::set_end_state(const VectorXd& x)
{
    // not sure if there is a more elegant way of copying
    for (unsigned i=0; i < x.size(); i++)
    {
        end_x(i) = x(i);
    }
}

/** ConstraintWithSystemGoalFree class */
double ConstraintWithSystemGoalFree::term_constraint(const VectorXd& x) const override
{
    return 0.;  // already satisfied
}
