#include "bvp/psopt_bvp.hpp"

void PSOPT_BVP::solve(const double* start, const double* goal, int num_steps, int max_iter,
                 double tmin, double tmax)
{

    // copy to the internal state
    for (unsigned i=0; i < state_n; i++)
    {
        _start[i] = start[i];
        _goal[i] = goal[i];
    }

    Alg algorithm;
    Sol solution;
    Prob problem;

    //problem.name = "Time  Varying state constraint problem";
    //problem.outfilename = "stc1.txt";

    problem.nphases = 1;
    problem.nlinkages = 0;
    psopt_level1_setup(problem);

    problem.phases(1).nstates = state_n;
    problem.phases(1).ncontrols = control_n;
    // events: boundary condition of states
    problem.phases(1).nevents = state_n*2; // boundary condition
    problem.phases(1).npath = 0;  // path constraint
    problem.phases(1).nodes = num_steps;
    //problem.phases(1).nodes = "[20 50]";  // use string as a sequence, and int as a desired number
    psopt_level2_setup(problem, algorithm);

    // obtain boundary from system
    std::vector<std::pair<double, double>> state_bound = system->get_state_bounds();
    for (unsigned i=1; i <= state_n; i+=1)
    {
        // specify the boundary
        problem.phases(1).bounds.lower.states(i) = state_bound[i-1][0];
        problem.phases(1).bounds.upper.states(i) = state_bound[i-1][1];
    }


    // obtain boundary for control
    std::vector<std::pair<double, double>> control_bound = system->get_control_bounds();
    for (unsigned i=1; i <= control_n; i+=1)
    {
        // specify the boundary
        problem.phases(1).bounds.lower.controls(i) = control_bound[i-1][0];
        problem.phases(1).bounds.upper.controls(i) = control_bound[i-1][1];
    }
    for (unsigned i=1; i <= state_n; i+=1)
    {
        // specify the boundary
        problem.phases(1).bounds.lower.events(i) = start[i-1];
        problem.phases(1).bounds.higher.events(i) = start[i-1];
        problem.phases(1).bounds.lower.events(state_n+i) = goal[state_n+i-1];
        problem.phases(1).bounds.higher.events(state_n+i) = goal[state_n+i-1];
    }

    problem.phases(1).bounds.lower.StartTime = 0.0;
    problem.phases(1).bounds.upper.StartTime = 0.0;

    problem.phases(1).bounds.lower.EndTime = tmin;
    problem.phases(1).bounds.upper.EndTime = tmax;


    problem.integrand_cost = &integrand_cost;
    problem.endpoint_cost = &endpoint_cost;
    problem.dae = &(system->dynamics);
    problem.events = &events;
    problem.linkages = &linkages;


    problem.phases(1).guess.controls = zeros(state_n, num_steps);
    problem.phases(1).guess.states = zeros(control_n, num_steps);
    problem.phases(1).guess.time = linspace(0.0, tmax, num_steps);

    algorithm.scaling = "automatic";
    algorithm.derivatives = "automatic";
    algorithm.nlp_iter_max = max_iter;
    algorithm.nlp_tolerance = 1.e-4;
    algorithm.nlp_method = "IPOPT";


    psopt(solution, problem, algorithm);

    DMatrix x = solution.get_states_in_phase(1);
    DMatrix u = solution.get_controls_in_phase(1);
    DMatrix t = solution.get_time_in_phase(1);

    //plot(t,u,"control","time (s)", "u");
    //plot(t,u,"control","time (s)", "u", "u", "pdf", "stc1_control.pdf");

    // printout the solution
    x.Print("bvp_x.txt");
    u.Print("bvp_u.txt");
    t.Print("bvp_t.txt");

}

adouble PSOPT_BVP::endpoint_cost(adouble* initial_states, adouble* final_states, adouble* parameters, adouble& t0,
                                 adouble& tf, adouble* xad, int iphase, Workspace* workspace)
{
    // Since we already set endpoint constraint in events, we don't need it here
    // TODO: maybe we can set one end free, but try to reduce the cost only
    // Here we use the time as endpoint cost for minimum time control
    return tf;
}

adouble PSOPT_BVP::integrand_cost(adouble* states, adouble* controls, adouble* parameters, adouble& time, adouble* xad,
                      int iphase, Workspace* workspace)
{
    adouble retval = 0.0;
    return retval;
}

void PSOPT_BVP::events(adouble* e, adouble* initial_states, adouble* final_states, adouble* parameters, adouble& t0,
            adouble& tf, adouble* xad, int iphase, Workspace* workspace)
{
  for (unsigned i=0; i < state_n; i++)
  {
      e[i] = initial_states[i];
      e[state_n+i] = final_states[i];
  }
}

void PSOPT_BVP::linkages(adouble* linkages, adouble* xad, Workspace* workspace)
{
  // No linkages in this single phase problem
}
