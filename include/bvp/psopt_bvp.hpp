/**
* a BVP solver implemented using PSOPT
*/
#ifndef PSOPT_BVP_HPP
#define PSOPT_BVP_HPP
#ifndef PSOPT_H
#define PSOPT_H
#include "psopt.h"
#endif

#include <vector>
#include "bvp/psopt_system.hpp"

class PSOPT_BVP
{
public:
    PSOPT_BVP(psopt_system_t* system_in, int state_n_in, int control_n_in)
    : state_n(state_n_in)
    , control_n(control_n_in)
    , system(system_in)
    , _start(new double[state_n_in])
    , _goal(new double[state_n_in]){}
    ~PSOPT_BVP()
    {
        delete[] _start;
        delete[] _goal;
    };
    void solve(const double* start, const double* goal, int num_steps, int max_iter,
          double tmin, double tmax);

protected:
    int state_n;
    int control_n;
    psopt_system_t* system;
    double* _start;
    double* _goal;
    adouble endpoint_cost(adouble* initial_states, adouble* final_states, adouble* parameters, adouble& t0,
                          adouble& tf, adouble* xad, int iphase, Workspace* workspace);
    adouble integrand_cost(adouble* states, adouble* controls, adouble* parameters, adouble& time, adouble* xad,
                          int iphase, Workspace* workspace);
    void events(adouble* e, adouble* initial_states, adouble* final_states, adouble* parameters, adouble& t0,
                          adouble& tf, adouble* xad, int iphase, Workspace* workspace);
    void linkages(adouble* linkages, adouble* xad, Workspace* workspace);
};

#endif
