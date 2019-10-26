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
#include <boost/assign/list_of.hpp>
#include <boost/format.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "sco/solver_interface.hpp"
#include "sco/optimizers.hpp"
#include "sco/modeling_utils.hpp"
using namespace sco;
using namespace Eigen;

SQPBVP::SQPBVP(system_interface* system, int state_dim_in, int control_dim_in, int n_steps, double integration_step)
: _n_steps(n_steps)
, state_dim(state_dim_in)
, control_dim(control_dim_in)
, _integration_step(integration_step)
{
    _system.reset(system);
    costPtr.reset(new CostWithSystem(system, state_dim_in, control_dim_in, n_steps, integration_step));
    constraintPtr.reset(new ConstraintWithSystem(system, state_dim_in, control_dim_in, n_steps, integration_step));
}

SQPBVP::~SQPBVP()
{
    _system.reset();
    costPtr.reset();
    constraintPtr.reset();
}

OptResults SQPBVP::solve(const VectorXd& start, const VectorXd& goal, int max_iter) const
{
    /**
    * Solve BVP problem from start to goal by constructing optimization problem.
    */
    // set start and goal states
    costPtr->set_start_state(start);
    costPtr->set_end_state(goal);
    constraintPtr->set_start_state(start);
    constraintPtr->set_end_state(goal);
    // construct optimization problem
    OptProbPtr probPtr(new OptProb());
    vector<string> var_names;
    for (unsigned i=0; i < _n_steps; i++)
    {
        // each state is of length state_dim
        for (unsigned j=0; j < state_dim; j++)
        {
            var_names.push_back( (boost::format("x_%1%_%2%")%i%j).str() );
        }

    }
    for (unsigned i=0; i < _n_steps-1; i++)
    {
        // each control is of length control_dim
        for (unsigned j=0; j < control_dim; j++)
        {
            var_names.push_back( (boost::format("u_%1%_%2%")%i%j).str() );
        }

    }
    for (unsigned i=0; i < _n_steps-1; i++)
    {
        var_names.push_back( (boost::format("dt_%i")%i).str() );
    }
    probPtr->createVariables(var_names);
    probPtr->addCost( CostPtr( new CostFromFunc(ScalarOfVectorPtr(costPtr), probPtr->getVars(), "f") ) );
    // Here ConstraintFromFunc actually specifies Constraint type (inequal or equal)
    // but we specified all constraints using inequal setup (equal becomes some norm <= 0)
    probPtr->addConstraint( (ConstraintPtr(
                               new ConstraintFromFunc(VectorOfVectorPtr(constraintPtr), probPtr->getVars(), VectorXd(), INEQ, "q") )) );
    BasicTrustRegionSQP solver(probPtr);
    // set solver parameters
    solver.max_iter_ = max_iter;
    solver.trust_box_size_ = 1;
    solver.min_trust_box_size_ = 1e-5;
    solver.min_approx_improve_ = 1e-10;
    solver.merit_error_coeff_ = 1;
    // initialize
    DblVec init;
    // state: straight line
    VectorXd dx = (goal - start) / (_n_steps-1);
    for (unsigned i=0; i < _n_steps; i++)
    {
        VectorXd x = start + i*dx;
        // append to init
        init.insert(init.end(), x.data(), x.data() + state_dim);
    }
    // control: 0
    for (unsigned i=0; i < _n_steps-1; i++)
    {
        VectorXd u = VectorXd::Zero(control_dim);
        init.insert(init.end(), u.data(), u.data() + control_dim);
    }
    // time: if there is approximate function, then can calculate, but we don't know
    //       which vars indicate velocity
    // for now we use Euclidean Distance
    //### TODO: add approximate time calculation function
    double T = (goal - start).squaredNorm();
    T = T / (_n_steps-1);
    for (unsigned i=0; i < _n_steps-1; i++)
    {
        init.push_back(T);
    }
    solver.initialize(init);
    OptStatus status = solver.optimize();
    OptResults res(solver.results());
    // delete pointer to clean memory to avoid SEGFAULT
    probPtr.reset();
    return res;
}


/** SQPBVP_forward class */
SQPBVP_forward::SQPBVP_forward(system_interface* system, int state_dim_in, int control_dim_in, int n_steps, double integration_step)
: SQPBVP(system, state_dim_in, control_dim_in, n_steps, integration_step)
{
    costPtr.reset(new CostWithSystemGoal(system, state_dim_in, control_dim_in, n_steps, integration_step));
    constraintPtr.reset(new ConstraintWithSystemGoalFree(system, state_dim_in, control_dim_in, n_steps, integration_step));
}
