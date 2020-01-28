#include "bvp/psopt_bvp.hpp"
#include "bvp/psopt_point.hpp"
#include "bvp/psopt_cart_pole.hpp"
#include "bvp/psopt_acrobot.hpp"
#include "bvp/psopt_pendulum.hpp"
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
        dist_calculator = new euclidean_distance(system_in->is_circular_topology());
    }
    else if (system_in->get_name() == "pendulum")
    {
        dae = &(psopt_pendulum_t::dynamics);
        endpoint_cost = &(psopt_pendulum_t::endpoint_cost);
        integrand_cost = &(psopt_pendulum_t::integrand_cost);
        events = &(psopt_pendulum_t::events);
        linkages = &(psopt_pendulum_t::linkages);
        dist_calculator = new euclidean_distance(system_in->is_circular_topology());
    }
    else if (system_in->get_name() == "point")
    {
        dae = &(psopt_point_t::dynamics);
        endpoint_cost = &(psopt_point_t::endpoint_cost);
        integrand_cost = &(psopt_point_t::integrand_cost);
        events = &(psopt_point_t::events);
        linkages = &(psopt_point_t::linkages);
        dist_calculator = new euclidean_distance(system_in->is_circular_topology());
    }
    else if (system_in->get_name() == "acrobot")
    {
        dae = &(psopt_acrobot_t::dynamics);
        endpoint_cost = &(psopt_acrobot_t::endpoint_cost);
        integrand_cost = &(psopt_acrobot_t::integrand_cost);
        events = &(psopt_acrobot_t::events);
        linkages = &(psopt_acrobot_t::linkages);
        dist_calculator = new euclidean_distance(system_in->is_circular_topology());
        //TODO: use the distance function in the acrobot
    }
}

PSOPT_BVP::~PSOPT_BVP()
{
    delete dist_calculator;
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
        /**
        if (system->is_circular_topology()[i])
        {
            problem.phases(1).bounds.lower.events(state_n+i) = goal[i-1]-2*M_PI*floor(goal[i-1]/2/M_PI);
            problem.phases(1).bounds.upper.events(state_n+i) = goal[i-1]-2*M_PI*floor(goal[i-1]/2/M_PI);
        }
        else
        {
            problem.phases(1).bounds.lower.events(state_n+i) = goal[i-1];
            problem.phases(1).bounds.upper.events(state_n+i) = goal[i-1];
        }
        */
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
        // if this state is an angle, then map to -pi~pi
        if (system->is_circular_topology()[i])
        {
            double dif = goal[i] - start[i];
            dif = dif - 2*M_PI*ceil(floor(dif/M_PI)/2);
            double wrapped_goal = start[i] + dif;
            DMatrix row(linspace(start[i], wrapped_goal, num_steps));
            row.Transpose();
            states.SetRow(row, i+1);
        }
        else
        {
            DMatrix row(linspace(start[i], goal[i], num_steps));
            row.Transpose();
            states.SetRow(row, i+1);

        }
    }
    //states.Save("state_init.txt");
    // dynamically initialize time based on (l/l_max)^2 * (t_max-t_min) + t_min
    double l = dist_calculator->distance(start, goal, state_n);
    double lmax = system->max_distance();
    //double init_time = pow(l/lmax, 1.8/state_n)*(tmax-tmin)+tmin;
    // initialize to be one step forward
    double init_time = 0.002*num_steps;

    std::cout << "start: [" << start[0] << ", " << start[1] << "]" << std::endl;
    std::cout << "goal: [" << goal[0] << ", " << goal[1] << "]" << std::endl;
    std::cout << "distance: " << l << std::endl;
    std::cout << "max_distance: " << lmax << std::endl;
    std::cout << "init_time: " << init_time << std::endl;
    std::cout << "tmin: " << tmin << std::endl;
    std::cout << "tmax: " << tmax << std::endl;
    problem.phases(1).guess.time = linspace(0.0, init_time, num_steps);

    algorithm.scaling = "automatic";
    algorithm.derivatives = "automatic";
    algorithm.hessian = "exact";
    algorithm.nlp_iter_max = max_iter;
    algorithm.nlp_tolerance = 1.e-6;
    algorithm.nlp_method = "IPOPT";
    algorithm.print_level = 0;
    psopt(solution, problem, algorithm);

    DMatrix x = solution.get_states_in_phase(1);
    DMatrix u = solution.get_controls_in_phase(1);
    DMatrix t = solution.get_time_in_phase(1);

    //plot(t,u,"control","time (s)", "u");
    //plot(t,u,"control","time (s)", "u", "u", "pdf", "stc1_control.pdf");

    // printout the solution
    //x.Save("bvp_x.txt");
    //u.Save("bvp_u.txt");
    //t.Save("bvp_t.txt");
    // DMatrix -> double vector
    //std::cout << "[";
    for (unsigned i=0; i < num_steps; i+=1)
    {
        std::vector<double> x_t;
        std::vector<double> u_t;
        //std::cout << "[";
        for (unsigned j=0; j < state_n; j+=1)
        {
            x_t.push_back(x(j+1,i+1));
            //std::cout << x(j+1,i+1) << ", ";
        }
        //std::cout << "], " << std::endl;
        for (unsigned j=0; j < control_n; j+=1)
        {
            u_t.push_back(u(j+1,i+1));
        }
        res.x.push_back(x_t);
        res.u.push_back(u_t);
        res.t.push_back(t(1,i+1));
    }
    // check if there is any error
    if (solution.error_flag)
    {
        std::cout << "error occurred" << std::endl;
    }
    //std::cout << "]" << std::endl;
}
