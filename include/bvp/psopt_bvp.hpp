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

typedef void (*dae_f)(adouble*, adouble*, adouble*, adouble*, adouble*, adouble&, adouble*, int, Workspace*);
typedef adouble (*endpoint_cost_f)(adouble*, adouble*, adouble*, adouble&, adouble&, adouble*, int, Workspace*);
typedef adouble (*integrand_cost_f)(adouble*, adouble*, adouble*, adouble&, adouble*, int, Workspace*);
typedef void (*events_f)(adouble*, adouble*, adouble*, adouble*, adouble&, adouble&, adouble*, int, Workspace*);
typedef void (*linkages_f)(adouble*, adouble*, Workspace*);


struct psopt_result_t
{
    std::vector<std::vector<double>> x;  // (T x X)
    std::vector<std::vector<double>> u;  // (T x U)
    std::vector<double> t;
};

class PSOPT_BVP
{
public:
    PSOPT_BVP(const psopt_system_t* system_in, int state_n_in, int control_n_in);

    psopt_result_t& solve(const double* start, const double* goal, int num_steps, int max_iter,
          double tmin, double tmax);

protected:
    int state_n;
    int control_n;
    const psopt_system_t* system;
    dae_f dae;
    endpoint_cost_f endpoint_cost;
    integrand_cost_f integrand_cost;
    events_f events;
    linkages_f linkages;
};
#endif
