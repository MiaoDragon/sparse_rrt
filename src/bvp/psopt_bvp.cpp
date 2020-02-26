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
    random_generator = new RandomGenerator(0);
}

PSOPT_BVP::~PSOPT_BVP()
{
    delete dist_calculator;
    delete random_generator;
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
    //problem.phases(1).nodes = num_steps;
    char node_string[40];
    sprintf(node_string, "[%d, %d, %d]", 1, 2, 4, num_steps/2, num_steps);
    problem.phases(1).nodes = node_string;
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


    //problem.phases(1).guess.controls = zeros(control_n, num_steps);
    DMatrix controls(control_n, num_steps);
    for (unsigned i=0; i < control_n; i+=1)
    {
        for (unsigned j=0; j < num_steps; j+=1)
        {
            // randomly set control input
            controls(i+1,j+1) = random_generator->uniform_random(control_bound[i].first, control_bound[i].second);
        }
    }
    problem.phases(1).guess.controls = controls;
    DMatrix states(state_n, num_steps);
    //problem.phases(1).guess.states = zeros(state_n, num_steps);
    // DMatrix index starts from 1
    // past initialization method
    for (unsigned i=0; i < state_n; i+=1)
    {
        // if this state is an angle, then map to -pi~pi
        if (system->is_circular_topology()[i])
        {
            double dif = goal[i] - start[i];
            dif = dif - 2*M_PI*ceil(floor(dif/M_PI)/2);
            //std::cout << i << "-th state, difference: " << dif << std::endl;
            double wrapped_goal = start[i] + dif;
            //std::cout << i << "-th state, wrapped_goal: " << wrapped_goal << std::endl;
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
    problem.phases(1).guess.states = states;
    //states.Save("state_init.txt");
    // dynamically initialize time based on (l/l_max)^2 * (t_max-t_min) + t_min
    double l = dist_calculator->distance(start, goal, state_n);
    double lmax = system->max_distance();
    //double init_time = pow(l/lmax, 1.8/state_n)*(tmax-tmin)+tmin;
    // initialize to be one step forward
    double init_time = random_generator->uniform_random(tmin, tmax);
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
    //std::cout << "]" << std::endl;
}


//#define MULTI_PHASE
#ifdef MULTI_PHASE
void PSOPT_BVP::solve(psopt_result_t& res, const double* start, const double* goal, int num_steps,
                                int max_iter, double tmin, double tmax,
                                const std::vector<std::vector<double>> &x_init,
                                const std::vector<std::vector<double>> &u_init,
                                const std::vector<double> &t_init)
{

    Alg algorithm;
    Sol solution;
    Prob problem;

    //problem.name = "Time  Varying state constraint problem";
    //problem.outfilename = "stc1.txt";

    problem.nphases = num_steps;
    problem.nlinkages = 0;
    psopt_level1_setup(problem);
    for (unsigned phase_i=1; phase_i<=num_steps; phase_i++)
    {
        problem.phases(phase_i).nstates = state_n;
        problem.phases(phase_i).ncontrols = control_n;
        // events: boundary condition of states
        //problem.phases(phase_i).nevents = state_n*2; // boundary condition
        problem.phases(phase_i).npath = 0;  // path constraint
        char node_string[40];
        sprintf(node_string, "[%d, %d, %d, %d]", 2, 4, num_steps/2, num_steps);
        std::cout << node_string << std::endl;
        problem.phases([phase_i]).nodes = node_string;
    }
    // boundary condition
    problem.phases(1).nevents = state_n;
    problem.phases(num_steps).nevents = state_n;

    psopt_level2_setup(problem, algorithm);

    // obtain boundary from system
    std::vector<std::pair<double, double>> state_bound = system->get_state_bounds();
    std::vector<std::pair<double, double>> control_bound = system->get_control_bounds();
    for (unsigned i=1; i <= state_n; i+=1)
    {
        // specify the boundary
        problem.phases(1).bounds.lower.events(i) = start[i-1];
        problem.phases(1).bounds.upper.events(i) = start[i-1];
        problem.phases(num_steps).bounds.lower.events(i) = goal[i-1];
        problem.phases(num_steps).bounds.upper.events(i) = goal[i-1];
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
    for (unsigned phase_i=1; phase_i<=num_steps; phase_i++)
    {
        for (unsigned i=1; i <= state_n; i+=1)
        {
            // specify the boundary
            problem.phases(phase_i).bounds.lower.states(i) = state_bound[i-1].first;
            problem.phases(phase_i).bounds.upper.states(i) = state_bound[i-1].second;
        }

        // obtain boundary for control
        for (unsigned i=1; i <= control_n; i+=1)
        {
            // specify the boundary
            problem.phases(phase_i).bounds.lower.controls(i) = control_bound[i-1].first;
            problem.phases(phase_i).bounds.upper.controls(i) = control_bound[i-1].second;
        }
        problem.phases(phase_i).bounds.lower.StartTime = 0.0;
        problem.phases(phase_i).bounds.upper.StartTime = 0.0;

        problem.phases(phase_i).bounds.lower.EndTime = tmin;
        problem.phases(phase_i).bounds.upper.EndTime = tmax;
    }


    //problem.integrand_cost = integrand_cost;  // register if there is integrand cost
    problem.endpoint_cost = endpoint_cost;
    problem.dae = dae;
    problem.events = events;
    problem.linkages = linkages;


    //problem.phases(1).guess.controls = zeros(control_n, num_steps);
    DMatrix controls(control_n, num_steps);
    for (unsigned i=0; i < control_n; i+=1)
    {
        for (unsigned j=0; j < num_steps; j+=1)
        {
            // randomly set control input
            controls(i+1,j+1) = u_init[j][i];
        }
    }
    /**
    for (unsigned i=0; i < control_n; i+=1)
    {
        for (unsigned j=0; j < num_steps; j+=1)
        {
            // randomly set control input
            controls(i+1,j+1) = random_generator->uniform_random(control_bound[i].first, control_bound[i].second);
        }
    }
    */
    problem.phases(1).guess.controls = controls;
    //problem.phases(1).guess.states = zeros(state_n, num_steps);
    // DMatrix index starts from 1
    DMatrix states(state_n, num_steps);
    for (unsigned i=0; i < state_n; i+=1)
    {
        for (unsigned j=0; j < num_steps; j+=1)
        {
            states(i+1,j+1) = x_init[j][i];
        }
    }
    /**
    // past initialization method
    for (unsigned i=0; i < state_n; i+=1)
    {
        // if this state is an angle, then map to -pi~pi
        if (system->is_circular_topology()[i])
        {
            double dif = goal[i] - start[i];
            dif = dif - 2*M_PI*ceil(floor(dif/M_PI)/2);
            //std::cout << i << "-th state, difference: " << dif << std::endl;
            double wrapped_goal = start[i] + dif;
            //std::cout << i << "-th state, wrapped_goal: " << wrapped_goal << std::endl;
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
    */
    problem.phases(1).guess.states = states;
    //states.Save("state_init.txt");
    // dynamically initialize time based on (l/l_max)^2 * (t_max-t_min) + t_min
    double l = dist_calculator->distance(start, goal, state_n);
    double lmax = system->max_distance();
    //double init_time = pow(l/lmax, 1.8/state_n)*(tmax-tmin)+tmin;
    // initialize to be one step forward
    DMatrix init_time(num_steps);
    for (unsigned i=0; i < num_steps; i++)
    {
        init_time(i+1) = t_init[i];
    }
    problem.phases(1).guess.time = init_time;

    algorithm.scaling = "automatic";
    algorithm.derivatives = "automatic";
    algorithm.hessian = "exact";
    algorithm.nlp_iter_max = max_iter;
    algorithm.nlp_tolerance = 1.e-6;  // default: 1e-6
    algorithm.ode_tolerance = 1.e-3;  // default: 1e-3
    algorithm.nlp_method = "IPOPT";
    algorithm.print_level = 0;
    //algorithm.collocation_method = "trapezoidal";
    algorithm.diff_matrix = "standard";  // options: "standard", "reduced-roundoff", "central-differences"
    algorithm.nsteps_error_integration = 5;
    //algorithm.mesh_refinement = "automatic";
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



#else
void PSOPT_BVP::solve(psopt_result_t& res, const double* start, const double* goal, int num_steps,
                                int max_iter, double tmin, double tmax,
                                const std::vector<std::vector<double>> &x_init,
                                const std::vector<std::vector<double>> &u_init,
                                const std::vector<double> &t_init)
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
    char node_string[40];
    sprintf(node_string, "[%d, %d]", num_steps/2, num_steps);
    std::cout << node_string << std::endl;
    problem.phases(1).nodes = node_string;
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


    //problem.integrand_cost = integrand_cost;  // register if there is integrand cost
    problem.endpoint_cost = endpoint_cost;
    problem.dae = dae;
    problem.events = events;
    problem.linkages = linkages;


    //problem.phases(1).guess.controls = zeros(control_n, num_steps);
    DMatrix controls(control_n, num_steps);
    for (unsigned i=0; i < control_n; i+=1)
    {
        for (unsigned j=0; j < num_steps; j+=1)
        {
            // randomly set control input
            controls(i+1,j+1) = u_init[j][i];
        }
    }
    /**
    for (unsigned i=0; i < control_n; i+=1)
    {
        for (unsigned j=0; j < num_steps; j+=1)
        {
            // randomly set control input
            controls(i+1,j+1) = random_generator->uniform_random(control_bound[i].first, control_bound[i].second);
        }
    }
    */
    problem.phases(1).guess.controls = controls;
    //problem.phases(1).guess.states = zeros(state_n, num_steps);
    // DMatrix index starts from 1
    DMatrix states(state_n, num_steps);
    for (unsigned i=0; i < state_n; i+=1)
    {
        for (unsigned j=0; j < num_steps; j+=1)
        {
            states(i+1,j+1) = x_init[j][i];
        }
    }
    /**
    // past initialization method
    for (unsigned i=0; i < state_n; i+=1)
    {
        // if this state is an angle, then map to -pi~pi
        if (system->is_circular_topology()[i])
        {
            double dif = goal[i] - start[i];
            dif = dif - 2*M_PI*ceil(floor(dif/M_PI)/2);
            //std::cout << i << "-th state, difference: " << dif << std::endl;
            double wrapped_goal = start[i] + dif;
            //std::cout << i << "-th state, wrapped_goal: " << wrapped_goal << std::endl;
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
    */
    problem.phases(1).guess.states = states;
    //states.Save("state_init.txt");
    // dynamically initialize time based on (l/l_max)^2 * (t_max-t_min) + t_min
    double l = dist_calculator->distance(start, goal, state_n);
    double lmax = system->max_distance();
    //double init_time = pow(l/lmax, 1.8/state_n)*(tmax-tmin)+tmin;
    // initialize to be one step forward
    DMatrix init_time(num_steps);
    for (unsigned i=0; i < num_steps; i++)
    {
        init_time(i+1) = t_init[i];
    }
    problem.phases(1).guess.time = init_time;

    algorithm.scaling = "automatic";
    algorithm.derivatives = "automatic";
    algorithm.hessian = "exact";
    algorithm.nlp_iter_max = max_iter;
    algorithm.nlp_tolerance = 1.e-6;  // default: 1e-6
    algorithm.ode_tolerance = 1.e-3;  // default: 1e-3
    algorithm.nlp_method = "IPOPT";
    algorithm.print_level = 0;
    //algorithm.collocation_method = "trapezoidal";
    algorithm.diff_matrix = "standard";  // options: "standard", "reduced-roundoff", "central-differences"
    algorithm.nsteps_error_integration = 5;
    //algorithm.mesh_refinement = "automatic";
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
    //std::cout << "]" << std::endl;
}
#endif
