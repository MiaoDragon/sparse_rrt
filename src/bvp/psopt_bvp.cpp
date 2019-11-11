#include "bvp/psopt_bvp.hpp"
#include "bvp/psopt_cart_pole.hpp"
#include "bvp/psopt_system.hpp"

PSOPT_BVP::PSOPT_BVP(const psopt_system_t* system_in, int state_n_in, int control_n_in)
: state_n(state_n_in)
, control_n(control_n_in)
, system(system_in)
{
    // based on the name of the system, register the function for computing cost and so on
    if (system_in->get_name() == "cartpole")
    {
        dae = &(psopt_cart_pole_t::dynamics);
        endpoint_cost = &(psopt_cart_pole_t::endpoint_cost);
        integrand_cost = &(psopt_cart_pole_t::integrand_cost);
        events = &(psopt_cart_pole_t::events);
        linkages = &(psopt_cart_pole_t::linkages);
    }
}

void PSOPT_BVP::solve(psopt_result_t& res, const double* start, const double* goal, int num_steps,
                                int max_iter, double tmin, double tmax)
{

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
        problem.phases(1).bounds.lower.states(i) = state_bound[i-1].first;
        problem.phases(1).bounds.upper.states(i) = state_bound[i-1].second;
    }


    // obtain boundary for control
    std::vector<std::pair<double, double>> control_bound = system->get_control_bounds();
    for (unsigned i=1; i <= control_n; i+=1)
    {
        // specify the boundary
        problem.phases(1).bounds.lower.controls(i) = control_bound[i-1].first;
        problem.phases(1).bounds.upper.controls(i) = control_bound[i-1].second;
    }
    for (unsigned i=1; i <= state_n; i+=1)
    {
        // specify the boundary
        problem.phases(1).bounds.lower.events(i) = start[i-1];
        problem.phases(1).bounds.upper.events(i) = start[i-1];
        problem.phases(1).bounds.lower.events(state_n+i) = goal[i-1];
        problem.phases(1).bounds.upper.events(state_n+i) = goal[i-1];
    }

    problem.phases(1).bounds.lower.StartTime = 0.0;
    problem.phases(1).bounds.upper.StartTime = 0.0;

    problem.phases(1).bounds.lower.EndTime = tmin;
    problem.phases(1).bounds.upper.EndTime = tmax;


    problem.integrand_cost = integrand_cost;
    problem.endpoint_cost = endpoint_cost;
    problem.dae = dae;
    problem.events = events;
    problem.linkages = linkages;


    problem.phases(1).guess.controls = zeros(control_n, num_steps);
    problem.phases(1).guess.states = zeros(state_n, num_steps);
    // DMatrix index starts from 1
    DMatrix states(state_n, num_steps);
    for (unsigned i=0; i < state_n; i+=1)
    {
        DMatrix row(linspace(start[i], goal[i], num_steps));
        row.Transpose();
        states.SetRow(row, i+1);
    }
    states.Save("state_init.txt");
    problem.phases(1).guess.time = linspace(0.0, tmax, num_steps);

    algorithm.scaling = "automatic";
    algorithm.derivatives = "automatic";
    algorithm.hessian = "exact";
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
    x.Save("bvp_x.txt");
    u.Save("bvp_u.txt");
    t.Save("bvp_t.txt");
    // DMatrix -> double vector
    std::cout << "[";
    for (unsigned i=0; i < num_steps; i+=1)
    {
        std::vector<double> x_t;
        std::vector<double> u_t;
        std::cout << "[";
        for (unsigned j=0; j < state_n; j+=1)
        {
            x_t.push_back(x(j+1,i+1));
            std::cout << x(j+1,i+1) << ", ";
        }
        std::cout << "], " << std::endl;
        for (unsigned j=0; j < control_n; j+=1)
        {
            u_t.push_back(u(j+1,i+1));
        }
        res.x.push_back(x_t);
        res.u.push_back(u_t);
        res.t.push_back(t(1,i+1));
    }
    std::cout << "]" << std::endl;
}
