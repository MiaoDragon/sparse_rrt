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
typedef void (*dae_f)(adouble*, adouble*, adouble*, adouble*, adouble*, adouble&, double*, int, Workspace*);
typedef adouble (*endpoint_cost_f)(adouble*, adouble*, adouble*, adouble&, adouble&, adouble*, int, Workspace*);
typedef adouble (*integrand_cost_f)(adouble*, adouble*, adouble*, adouble&, adouble*, int, Workspace*);
typedef void (*events_f)(adouble*, adouble*, adouble*, adouble*, adouble&, adouble&, adouble*, int, Workspace*);
typedef void (*linkages_f)(adouble*, adouble*, Workspace*);
class PSOPT_BVP
{
public:
    PSOPT_BVP(const psopt_system_t* system_in, int state_n_in, int control_n_in);

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
    double* _start;
    double* _goal;
    const psopt_system_t* system;
    dae_f dae;
    endpoint_cost_f endpoint_cost;
    integrand_cost_f integrand_cost;
    events_f events;
    linkages_f linkages;
};
#endif
