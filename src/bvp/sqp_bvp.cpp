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
using namespace sco;
using namespace Eigen;

SQPBVP::SQPBVP(system_interface* system, int state_dim_in, int control_dim_in, int n_steps, double integration_step)
: _system(system)
, _n_steps(n_steps)
, state_dim(state_dim_in)
, control_dim(control_dim_in)
, _integration_step(integration_step)
, costPtr(new CostWithSystem(system, n_steps, integration_step))
, constraintPtr(new ConstraintWithSystem(system, n_steps, integration_step))
{
    probPtr.reset();  // initialize: point to NULL
}

SQPBVP::~SQPBVP()
{
    delete costPtr;
    delete constraintPtr;
    probPtr.reset();
}

std::vector<double> SQPBVP::solve(const VectorXd& start, const VectorXd& goal) const
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
    probPtr.reset(new OptProb());
    vector<string> var_names;
    for (unsigned i=0; i < _n_steps; i++)
    {
        var_names.push_back( (boost::format("x_%i")%i).str() );
    }
    for (unsigned i=0; i < _n_steps-1; i++)
    {
        var_names.push_back( (boost::format("u_%i")%i).str() );
    }
    for (unsigned i=0; i < _n_steps-1; i++)
    {
        var_names.push_back( (boost::format("dt_%i")%i).str() );
    }
    probPtr->createVariables(var_names);
    probPtr->addCost( CostPtr( new CostFromFunc(ScalarOfVectorPtr(costPtr), probPtr->getVars(), 'f') ) );
    // Here ConstraintFromFunc actually specifies Constraint type (inequal or equal)
    // but we specified all constraints using inequal setup (equal becomes some norm <= 0)
    probPtr->addConstraint( (ConstraintPtr(
                               new ConstraintFromFunc(VectorOfVectorPtr(constraintPtr), probPtr->getVars(), VectorXd(), INEQ, 'q') )) );
    BasicTrustRegionSQP solver(probPtr);
    // set solver parameters
    //solver.max_iter_ = 1000;
    //solver.min_trust_box_size_ = 1e-5;
    //solver.min_approx_improve_ = 1e-10;
    //solver.merit_error_coeff_ = 1;
    // initialize
    DblVec init;
    // state: straight line
    VectorXd dx = (goal - start) / (_n_steps-1);
    for (unsigned i=0; i < _n_steps; i++)
    {
        VectorXd x = start + i*dx;
        // append to init
        init.insert(init.end(), x.data().begin(), x.data().end());
    }
    // control: 0
    for (unsigned i=0; i < _n_steps-1; i++)
    {
        init.push_back(0.0);
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
    return solver.x();
}
