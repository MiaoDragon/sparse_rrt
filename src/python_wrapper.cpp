/**
 * @file python_wrapper.cpp
 *
 * @copyright Software License Agreement (BSD License)
 * Original work Copyright 2017 Oleg Y. Sinyavskiy
 * All Rights Reserved.
 * For a full description see the file named LICENSE.
 *
 * Original authors: Oleg Y. Sinyavskiy
 *
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <iostream>
#include <assert.h>

#include "systems/point.hpp"
#include "systems/car.hpp"
#include "systems/cart_pole.hpp"
#include "systems/cart_pole_obs.hpp"
#include "systems/pendulum.hpp"
#include "systems/rally_car.hpp"
#include "systems/two_link_acrobot.hpp"

#include "motion_planners/sst.hpp"
#include "motion_planners/rrt.hpp"

#include "bvp/psopt_bvp.hpp"
#include "bvp/psopt_system.hpp"
#include "bvp/psopt_pendulum.hpp"
#include "bvp/psopt_cart_pole.hpp"
#include "bvp/psopt_point.hpp"

#include "image_creation/planner_visualization.hpp"
#include "systems/distance_functions.h"

#include "utilities/random.hpp"

namespace pybind11 {
    template <typename T>
    using safe_array = typename pybind11::array_t<T, pybind11::array::c_style>;
}

namespace py = pybind11;
using namespace pybind11::literals;


/**
 * @brief Python trampoline for abstract distance_t
 * @details Python trampoline for abstract distance_t to enable python classes override distance_t functions
 *
 */
class py_distance_interface : public distance_t
{
public:

	/**
	 * @copydoc distance_t::distance()
	 */
    double distance(const double* point1, const double* point2, unsigned int state_dimension) const override
    {
        // Copy cpp points to numpy arrays
        py::safe_array<double> point1_array{{state_dimension}};
        std::copy(point1, point1 + state_dimension, point1_array.mutable_data(0));

        py::safe_array<double> point2_array{{state_dimension}};
        std::copy(point2, point2 + state_dimension, point2_array.mutable_data(0));

        // Call python function
        py::gil_scoped_acquire gil;
        py::function overload = py::get_overload(static_cast<const distance_t *>(this), "distance");
        if (!overload) {
            pybind11::pybind11_fail("Tried to call pure virtual function distance");
            return false;
        }

        auto result = overload(point1_array, point2_array);
        // Extract double result
        return py::detail::cast_safe<double>(std::move(result));;
    }
};


/**
 * @brief Create cpp implementation of euclidean_distance
 * @details Create cpp implementation of euclidean_distance to use it from python
 *
 * @param is_circular_topology_array numpy array that indicates whether state coordinates have circular topology
 *
 * @return euclidean_distance object
 */
euclidean_distance create_euclidean_distance(
    const py::safe_array<bool> &is_circular_topology_array)
{
    auto is_circular_topology = is_circular_topology_array.unchecked<1>();
    std::vector<bool> is_circular_topology_v;
    for (int i = 0; i < is_circular_topology_array.shape()[0]; i++) {
        is_circular_topology_v.push_back(is_circular_topology(i));
    }
    return euclidean_distance(is_circular_topology_v);
}


/**
 * @brief Python wrapper for planner_t class
 * @details Python wrapper for planner_t class that handles numpy arguments and passes them to cpp functions
 *
 */
class PlannerWrapper
{
public:

    /**
	 * @copydoc planner_t::step()
	 */
   py::safe_array<double> step_with_sample(psopt_system_t& system, py::safe_array<double>& sample_state_py, int min_time_steps, int max_time_steps, double integration_step)
   {
        auto init_sample_state = sample_state_py.unchecked<1>(); // need to be one dimension vector
        // create a sample variable that holds the initial value from the one passed
        long unsigned int size = init_sample_state.shape(0);
        double* sample_state = new double[size];
        double* new_state = new double[size];
        for (int i = 0; i < size; i++) {
          sample_state[i] = init_sample_state(i);
        }
        planner->step_with_sample(&system, sample_state, new_state, min_time_steps, max_time_steps, integration_step);
        // return the new sample
        py::safe_array<double> new_state_py({size});
        auto new_state_ref = new_state_py.mutable_unchecked<1>();
        for (unsigned int i = 0; i < size; i++) {
          new_state_ref(i) = new_state[i];
        }
        delete[] new_state;
        delete[] sample_state;
        return new_state_py;
    }

    void step(system_interface& system, int min_time_steps, int max_time_steps, double integration_step) {
        planner->step(&system, min_time_steps, max_time_steps, integration_step);
    }
    void step_bvp(psopt_system_t& system, int min_time_steps, int max_time_steps, double integration_step) {
        planner->step_bvp(&system, min_time_steps, max_time_steps, integration_step);
    }

    /**
     * @brief Generate SVG visualization of the planning tree
     * @details Generate SVG visualization of the planning tree
     *
     * @param system an instance of the system to plan for
     * @param image_width Width of the drawing.
     * @param image_height Height of the drawing.
     * @param solution_node_diameter Diameter of a node.
     * @param solution_line_width Width of a planning solution path.
     * @param tree_line_width Width of the edges in the planning tree.
     *
     * @return string with SVG xml description
     */
    std::string visualize_tree_wrapper(
        system_interface& system,
        int image_width,
        int image_height,
        double solution_node_diameter,
        double solution_line_width,
        double tree_line_width)
    {
        std::vector<std::vector<double>> solution_path;
        std::vector<std::vector<double>> controls;
        std::vector<double> costs;
        planner->get_solution(solution_path, controls, costs);

        using namespace std::placeholders;
        std::string document_body = visualize_tree(
            planner->get_root(), solution_path,
            std::bind(&system_t::visualize_point, &system, _1, planner->get_state_dimension()),
            planner->get_start_state(), planner->get_goal_state(),
            image_width, image_height, solution_node_diameter, solution_line_width, tree_line_width);

        return std::move(document_body);
    }

    /**
     * @brief Generate SVG visualization of the nodes in the planning tree
     * @details Generate SVG visualization of the nodes in the planning tree
     *
     * @param system an instance of the system to plan for
     * @param image_width Width of the drawing.
     * @param image_height Height of the drawing.
     * @param node_diameter Diameter of nodes
     * @param solution_node_diameter Diameter of nodes that belong to the planning solution
     *
     * @return string with SVG xml description
     */
    std::string visualize_nodes_wrapper(
        system_interface& system,
        int image_width,
        int image_height,
        double node_diameter,
        double solution_node_diameter)
    {
        std::vector<std::vector<double>> solution_path;
        std::vector<std::vector<double>> controls;
        std::vector<double> costs;
        planner->get_solution(solution_path, controls, costs);

        using namespace std::placeholders;
        std::string document_body = visualize_nodes(
            planner->get_root(), solution_path,
            std::bind(&system_t::visualize_point, &system, _1, planner->get_state_dimension()),
            planner->get_start_state(),
            planner->get_goal_state(),
            image_width, image_height, node_diameter, solution_node_diameter);

        return std::move(document_body);
    }

    /**
	 * @copydoc planner_t::get_solution()
	 */
    py::object get_solution() {
        std::vector<std::vector<double>> solution_path;
        std::vector<std::vector<double>> controls;
        std::vector<double> costs;
        planner->get_solution(solution_path, controls, costs);

        if (controls.size() == 0) {
            return py::none();
        }

        py::safe_array<double> controls_array({controls.size(), controls[0].size()});
        py::safe_array<double> costs_array({costs.size()});
        auto controls_ref = controls_array.mutable_unchecked<2>();
        auto costs_ref = costs_array.mutable_unchecked<1>();
        for (unsigned int i = 0; i < controls.size(); ++i) {
            for (unsigned int j = 0; j < controls[0].size(); ++j) {
                controls_ref(i, j) = controls[i][j];
            }
            costs_ref(i) = costs[i];
        }

        py::safe_array<double> state_array({solution_path.size(), solution_path[0].size()});
        auto state_ref = state_array.mutable_unchecked<2>();
        for (unsigned int i = 0; i < solution_path.size(); ++i) {
            for (unsigned int j = 0; j < solution_path[0].size(); ++j) {
                state_ref(i, j) = solution_path[i][j];
            }
        }
        return py::cast(std::tuple<py::safe_array<double>, py::safe_array<double>, py::safe_array<double>>
            (state_array, controls_array, costs_array));
    }

    /**
	 * @copydoc planner_t::get_number_of_nodes()
	 */
    unsigned int get_number_of_nodes() {
        return this->planner->get_number_of_nodes();
    }

protected:
	/**
	 * @brief Created planner object
	 */
    std::unique_ptr<planner_t> planner;
};



/**
 * @brief Python wrapper for SST planner
 * @details Python wrapper for SST planner that handles numpy arguments and passes them to cpp functions
 *
 */
class __attribute__ ((visibility ("hidden"))) SSTWrapper : public PlannerWrapper{
public:

	/**
	 * @brief Python wrapper of SST planner Constructor
	 * @details Python wrapper of SST planner Constructor
	 *
	 * @param state_bounds_array numpy array (N x 2) with boundaries of the state space (min and max)
	 * @param control_bounds_array numpy array (N x 2) with boundaries of the control space (min and max)
	 * @param distance_computer_py Python wrapper of distance_t implementation
	 * @param start_state_array The start state (numpy array)
	 * @param goal_state_array The goal state  (numpy array)
	 * @param goal_radius The radial size of the goal region centered at in_goal.
	 * @param random_seed The seed for the random generator
	 * @param sst_delta_near Near distance threshold for SST
	 * @param sst_delta_drain Drain distance threshold for SST
	 */
    SSTWrapper(
            const py::safe_array<double> &state_bounds_array,
            const py::safe_array<double> &control_bounds_array,
            py::object distance_computer_py,
            const py::safe_array<double> &start_state_array,
            const py::safe_array<double> &goal_state_array,
            double goal_radius,
            unsigned int random_seed,
            double sst_delta_near,
            double sst_delta_drain
    )
        : _distance_computer_py(distance_computer_py)  // capture distance computer to avoid segfaults because we use a raw pointer from it
    {
        if (state_bounds_array.shape()[0] != start_state_array.shape()[0]) {
            throw std::domain_error("State bounds and start state arrays have to be equal size");
        }

        if (state_bounds_array.shape()[0] != goal_state_array.shape()[0]) {
            throw std::domain_error("State bounds and goal state arrays have to be equal size");
        }

        distance_t* distance_computer = distance_computer_py.cast<distance_t*>();

        auto state_bounds = state_bounds_array.unchecked<2>();
        auto control_bounds = control_bounds_array.unchecked<2>();
        auto start_state = start_state_array.unchecked<1>();
        auto goal_state = goal_state_array.unchecked<1>();

        typedef std::pair<double, double> bounds_t;
        std::vector<bounds_t> state_bounds_v;

        for (unsigned int i = 0; i < state_bounds_array.shape()[0]; i++) {
            state_bounds_v.push_back(bounds_t(state_bounds(i, 0), state_bounds(i, 1)));
        }

        std::vector<bounds_t> control_bounds_v;
        for (unsigned int i = 0; i < control_bounds_array.shape()[0]; i++) {
            control_bounds_v.push_back(bounds_t(control_bounds(i, 0), control_bounds(i, 1)));
        }

        std::function<double(const double*, const double*, unsigned int)>  distance_f =
            [distance_computer] (const double* p0, const double* p1, unsigned int dims) {
                return distance_computer->distance(p0, p1, dims);
            };

        planner.reset(
                new sst_t(
                        &start_state(0), &goal_state(0), goal_radius,
                        state_bounds_v, control_bounds_v,
                        distance_f,
                        random_seed,
                        sst_delta_near, sst_delta_drain)
        );
    }
private:

	/**
	 * @brief Captured distance computer python object to prevent its premature death
	 */
    py::object  _distance_computer_py;
};


class __attribute__ ((visibility ("hidden"))) RRTWrapper : public PlannerWrapper{
public:

	/**
	 * @brief Python wrapper of RRT planner constructor
	 * @details Python wrapper of RRT planner constructor
	 *
	 * @param state_bounds_array numpy array (N x 2) with boundaries of the state space (min and max)
	 * @param control_bounds_array numpy array (N x 2) with boundaries of the control space (min and max)
	 * @param distance_computer_py Python wrapper of distance_t implementation
	 * @param start_state_array The start state (numpy array)
	 * @param goal_state_array The goal state  (numpy array)
	 * @param goal_radius The radial size of the goal region centered at in_goal.
	 * @param random_seed The seed for the random generator
	 */
    RRTWrapper(
            const py::safe_array<double> &state_bounds_array,
            const py::safe_array<double> &control_bounds_array,
            py::object distance_computer_py,
            const py::safe_array<double> &start_state_array,
            const py::safe_array<double> &goal_state_array,
            double goal_radius,
            unsigned int random_seed
    ) : _distance_computer_py(distance_computer_py)
    {
        if (state_bounds_array.shape()[0] != start_state_array.shape()[0]) {
            throw std::runtime_error("State bounds and start state arrays have to be equal size");
        }

        if (state_bounds_array.shape()[0] != goal_state_array.shape()[0]) {
            throw std::runtime_error("State bounds and goal state arrays have to be equal size");
        }

        auto state_bounds = state_bounds_array.unchecked<2>();
        auto control_bounds = control_bounds_array.unchecked<2>();
        auto start_state = start_state_array.unchecked<1>();
        auto goal_state = goal_state_array.unchecked<1>();

        typedef std::pair<double, double> bounds_t;
        std::vector<bounds_t> state_bounds_v;
        for (unsigned int i = 0; i < state_bounds_array.shape()[0]; i++) {
            state_bounds_v.push_back(bounds_t(state_bounds(i, 0), state_bounds(i, 1)));
        }

        std::vector<bounds_t> control_bounds_v;
        for (unsigned int i = 0; i < control_bounds_array.shape()[0]; i++) {
            control_bounds_v.push_back(bounds_t(control_bounds(i, 0), control_bounds(i, 1)));
        }

        distance_t* distance_computer = distance_computer_py.cast<distance_t*>();
        std::function<double(const double*, const double*, unsigned int)>  distance_f =
            [distance_computer] (const double* p0, const double* p1, unsigned int dims) {
                return distance_computer->distance(p0, p1, dims);
            };

        planner.reset(
                new rrt_t(
                        &start_state(0), &goal_state(0), goal_radius,
                        state_bounds_v, control_bounds_v,
                        distance_f,
                        random_seed)
        );
    }
private:

	/**
	 * @brief Captured distance computer python object to prevent its premature death
	 */
    py::object  _distance_computer_py;
};


/**
 * @brief Python trampoline for system_interface distance_t
 * @details Python trampoline for system_interface distance_t to enable python classes override system_interface functions
 * and be able to create python systems
 *
 */
class py_system_interface : public system_interface
{
public:

	/**
	 * @copydoc system_interface::propagate()
	 */
    bool propagate(
        const double* start_state, unsigned int state_dimension,
        const double* control, unsigned int control_dimension,
        int num_steps,
        double* result_state, double integration_step) override
    {
        py::safe_array<double> start_state_array{{state_dimension}};
        std::copy(start_state, start_state + state_dimension, start_state_array.mutable_data(0));

        py::safe_array<double> control_array{{control_dimension}};
        std::copy(control, control + control_dimension, control_array.mutable_data(0));

        py::gil_scoped_acquire gil;
        py::function overload = py::get_overload(static_cast<const system_interface *>(this), "propagate");
        if (!overload) {
            pybind11::pybind11_fail("Tried to call pure virtual function propagate");
            return false;
        }

        auto result = overload(start_state_array, control_array, num_steps, integration_step);
        if (py::isinstance<py::none>(result)) {
            return false;
        } else {
            auto result_state_array = py::detail::cast_safe<py::safe_array<double>>(std::move(result));
            std::copy(result_state_array.data(0), result_state_array.data(0) + state_dimension, result_state);
            return true;
        }
    }

	/**
	 * @copydoc system_interface::visualize_point()
	 */
    std::tuple<double, double> visualize_point(const double* state, unsigned int state_dimension) const override {
        py::safe_array<double> state_array{{state_dimension}};
        std::copy(state, state + state_dimension, state_array.mutable_data(0));

        py::gil_scoped_acquire gil;
        py::function overload = py::get_overload(static_cast<const system_interface *>(this), "visualize_point");
        if (!overload) {
            pybind11::pybind11_fail("Tried to call pure virtual function visualize_point");
            return std::make_tuple(-1., -1.);
        }
        auto result = overload(state_array);
        return py::detail::cast_safe<std::tuple<double, double>>(std::move(result));
    }

	/**
	 * @copydoc system_interface::visualize_obstacles()
	 */
    std::string visualize_obstacles(int image_width, int image_height) const override
    {
    	PYBIND11_OVERLOAD(
            std::string,                /* Return type */
            system_interface,           /* Parent class */
            visualize_obstacles,        /* Name of function in C++ (must match Python name) */
            image_width, image_height   /* Argument(s) */
        );
    }
};


/**
 * @brief CartPole with Obstacle system Wrapper
 * @details python interface using C++ implementation
 *
 */
 class __attribute__ ((visibility ("hidden"))) CartPoleObsWrapper : public system_t
 {
 public:

 	/**
 	 * @brief Python wrapper of CartPoleObs constructor
 	 * @details Python wrapper of CartPoleObs constructor
 	 *
 	 * @param _obs_list: numpy array (N x 2) representing the middle point of the obstacles
     * @param width: width of the rectangle obstacle
 	 */
     CartPoleObsWrapper(
             const py::safe_array<double> &_obs_list,
             double width
      )
     {
         if (_obs_list.shape()[0] == 0) {
             throw std::runtime_error("Should contain at least one obstacles.");
         }
         if (_obs_list.shape()[1] != 2) {
             throw std::runtime_error("Shape of the obstacle input should be (N,2).");
         }
         if (width <= 0.) {
             throw std::runtime_error("obstacle width should be non-negative.");
         }
         auto py_obs_list = _obs_list.unchecked<2>();
         // initialize the array
         std::vector<std::vector<double>> obs_list(_obs_list.shape()[0], std::vector<double>(2, 0.0));
         // copy from python array to this array
         for (unsigned int i = 0; i < obs_list.size(); i++) {
             obs_list[i][0] = py_obs_list(i, 0);
             obs_list[i][1] = py_obs_list(i, 1);
         }
         cart_pole_obs.reset(
                 new cart_pole_obs_t(obs_list, width)
         );
     }

    bool propagate(
             const double* start_state, unsigned int state_dimension,
             const double* control, unsigned int control_dimension,
     	    int num_steps, double* result_state, double integration_step)
    {
        return cart_pole_obs->propagate(start_state, state_dimension, control, control_dimension,
                                    num_steps, result_state, integration_step);
    }

    void enforce_bounds()
    {
        cart_pole_obs->enforce_bounds();
    }

    bool valid_state()
    {
        return cart_pole_obs->valid_state();
    }
    std::tuple<double, double> visualize_point(const double* state, unsigned int state_dimension) const override
    {
        return cart_pole_obs->visualize_point(state, state_dimension);
    }
    std::vector<std::pair<double, double>> get_state_bounds() const override
    {
        return cart_pole_obs->get_state_bounds();
    }
    std::vector<std::pair<double, double>> get_control_bounds() const override
    {
        return cart_pole_obs->get_control_bounds();
    }

    std::vector<bool> is_circular_topology() const override
    {
        return cart_pole_obs->is_circular_topology();
    }

 protected:
 	/**
 	 * @brief Created planner object
 	 */
     std::unique_ptr<cart_pole_obs_t> cart_pole_obs;
 };



class PSOPTBVPWrapper
{
public:
    PSOPTBVPWrapper(psopt_system_t& system, int state_dim_in, int control_dim_in, int random_seed)
    : state_dim(state_dim_in)
    , control_dim(control_dim_in)
    , random_generator(random_seed)
    {
        // create a new system
        _system.reset(&system);
        // _system = new system_interface(&system);
        bvp_solver.reset(new PSOPT_BVP(&system, state_dim_in, control_dim_in));
    }
    ~PSOPTBVPWrapper()
    {
        _system.reset();
        bvp_solver.reset();
    }

    py::object solve(py::safe_array<double>& start_py, py::safe_array<double>& goal_py, int max_iter, int min_time_steps, int max_time_steps, double integration_step)
    {
        auto start_data_py = start_py.unchecked<1>(); // need to be one dimension vector
        auto goal_data_py = goal_py.unchecked<1>();
        int size = start_data_py.shape(0);
        double* start = new double[size];
        double* goal = new double[size];
        // copy from input to start and goal
        for (unsigned i=0; i < size; i++)
        {
            start[i] = start_data_py(i);
            goal[i] = goal_data_py(i);
        }
        int num_steps = 6*this->state_dim;
        psopt_result_t res;
        double tmin = integration_step*num_steps;
        double tmax = 50*max_time_steps*integration_step*num_steps;
        bvp_solver->solve(res, start, goal, num_steps, max_iter, tmin, tmax);


        std::vector<std::vector<double>> res_x = res.x;  // optimziation solution
        std::vector<std::vector<double>> res_u = res.u;  // optimziation solution
        std::vector<double> res_t;  // optimziation solution
        //res_t.push_back(res.t[1]/2);
        for (unsigned i=0; i < num_steps-1; i+=1)
        {
            res_t.push_back(res.t[i+1] - res.t[i]);
            //res_t.push_back((res.t[i+1]-res.t[i-1])/2);
        }
        //res_t.push_back((res.t[num_steps-1]-res.t[num_steps-2])/2);

        py::safe_array<double> state_array({res_x.size(), res_x[0].size()});
        py::safe_array<double> control_array({res_u.size(), res_u[0].size()});
        py::safe_array<double> time_array({res_t.size()});
        auto state_ref = state_array.mutable_unchecked<2>();
        for (unsigned int i = 0; i < res_x.size(); ++i) {
            for (unsigned int j = 0; j < res_x[0].size(); ++j) {
                state_ref(i, j) = res_x[i][j];
            }
        }
        auto control_ref = control_array.mutable_unchecked<2>();
        for (unsigned int i = 0; i < res_u.size(); ++i) {
            for (unsigned int j = 0; j < res_u[0].size(); ++j) {
                control_ref(i, j) = res_u[i][j];
            }
        }
        auto time_ref = time_array.mutable_unchecked<1>();
        for (unsigned int i = 0; i < res_t.size(); ++i) {
            time_ref(i) = res_t[i];
        }

        delete[] start;
        delete[] goal;
        // return flag, available flags, states, controls, time
        return py::cast(std::tuple<py::safe_array<double>, py::safe_array<double>, py::safe_array<double>>
            (state_array, control_array, time_array));
    }

    py::object steerTo(py::safe_array<double>& start_py, py::safe_array<double>& goal_py, int max_iter, int min_time_steps, int max_time_steps, double integration_step)
    {
        auto start_data_py = start_py.unchecked<1>(); // need to be one dimension vector
        auto goal_data_py = goal_py.unchecked<1>();
        int size = start_data_py.shape(0);
        double* start = new double[size];
        double* goal = new double[size];
        // copy from input to start and goal
        for (unsigned i=0; i < size; i++)
        {
            start[i] = start_data_py(i);
            goal[i] = goal_data_py(i);
        }
        int num_steps = 6*this->state_dim;
        psopt_result_t res;
        double tmin = integration_step*num_steps;
        double tmax = 50*max_time_steps*integration_step*num_steps;
        bvp_solver->solve(res, start, goal, num_steps, max_iter, tmin, tmax);


        std::vector<std::vector<double>> x_traj = res.x;  // optimziation solution
        std::vector<std::vector<double>> u_traj = res.u;  // optimziation solution
        std::vector<double> t_traj;  // optimziation solution
        for (unsigned i=0; i < num_steps-1; i+=1)
        {
            t_traj.push_back(res.t[i+1] - res.t[i]);
        }
        // variables to return
        std::vector<std::vector<double>> res_x;
        std::vector<std::vector<double>> res_u;
        std::vector<double> res_t;
        // add the start state to x_traj first
        res_x.push_back(x_traj[0]);

        for (unsigned i=0; i < num_steps-1; i++)
        {
            if (t_traj[i] < integration_step / 2)
            {
                // the time step is too small, ignore this action
                continue;
            }
            int num_dis = std::round(t_traj[i] / integration_step);
            double* control_ptr = u_traj[i].data();
            int num_steps = this->random_generator.uniform_int_random(min_time_steps, max_time_steps);
            int num_j = num_dis / num_steps + 1;
            bool val = true;
            //std::cout << "num_j: " << num_j << std::endl;
            for (unsigned j=0; j < num_j; j++)
            {
                int time_step = num_steps;
                if (j == num_j-1)
                {
                    time_step = num_dis % num_steps;
                }
                if (time_step == 0)
                {
                    // when we don't need to propagate anymore, break
                    break;
                }
                val = _system->propagate(start, this->state_dim, control_ptr, this->control_dim,
                                 time_step, goal, integration_step);
                // copy the new state to start
                for (unsigned k=0; k < this->state_dim; k++)
                {
                    start[k] = goal[k];
                }
                 //std::cout << "after propagation... val: " << val << std::endl;
                // add the new state to tree
                if (!val)
                {
                    // not valid state, no point going further, not adding to tree, stop right here
                    break;
                }
                // add the path to return
                std::vector<double> x_vec;
                std::vector<double> u_vec;
                for (unsigned k=0; k < this->state_dim; k++)
                {
                    x_vec.push_back(start[k]);
                }
                for (unsigned k=0; k < this->control_dim; k++)
                {
                    u_vec.push_back(control_ptr[k]);
                }
                res_x.push_back(x_vec);
                res_u.push_back(u_vec);
                res_t.push_back(time_step*integration_step);
            }
            if (!val)
            {
                break;
            }
        }



        // from solution we can obtain the trajectory: state traj | action traj | time traj
        py::safe_array<double> state_array({res_x.size(), res_x[0].size()});
        py::safe_array<double> control_array({res_u.size(), res_u[0].size()});
        py::safe_array<double> time_array({res_t.size()});
        auto state_ref = state_array.mutable_unchecked<2>();
        for (unsigned int i = 0; i < res_x.size(); ++i) {
            for (unsigned int j = 0; j < res_x[0].size(); ++j) {
                state_ref(i, j) = res_x[i][j];
            }
        }
        auto control_ref = control_array.mutable_unchecked<2>();
        for (unsigned int i = 0; i < res_u.size(); ++i) {
            for (unsigned int j = 0; j < res_u[0].size(); ++j) {
                control_ref(i, j) = res_u[i][j];
            }
        }
        auto time_ref = time_array.mutable_unchecked<1>();
        for (unsigned int i = 0; i < res_t.size(); ++i) {
            time_ref(i) = res_t[i];
        }

        delete[] start;
        delete[] goal;
        // return flag, available flags, states, controls, time
        return py::cast(std::tuple<py::safe_array<double>, py::safe_array<double>, py::safe_array<double>>
            (state_array, control_array, time_array));
    }

protected:
    std::shared_ptr<PSOPT_BVP> bvp_solver;
    std::shared_ptr<psopt_system_t> _system;
    int state_dim, control_dim;
	RandomGenerator random_generator;
};


/**
 * @brief pybind module
 * @details pybind module for all planners, systems and interfaces
 *
 */
PYBIND11_MODULE(_sst_module, m) {
   m.doc() = "Python wrapper for SST planners";

   // Classes and interfaces for distance computation
   py::class_<distance_t, py_distance_interface> distance_interface_var(m, "IDistance");
   distance_interface_var
        .def(py::init<>());
   py::class_<euclidean_distance, distance_t>(m, "EuclideanDistance");
   py::class_<two_link_acrobot_distance, distance_t>(m, "TwoLinkAcrobotDistance").def(py::init<>());
   m.def("euclidean_distance", &create_euclidean_distance, "is_circular_topology"_a.noconvert());

   // Classes and interfaces for systems
   py::class_<system_interface, py_system_interface> system_interface_var(m, "ISystem");
   system_interface_var
        .def(py::init<>())
        .def("propagate", &system_interface::propagate)
        .def("visualize_point", &system_interface::visualize_point)
        .def("visualize_obstacles", &system_interface::visualize_obstacles);

   py::class_<system_t> system(m, "System", system_interface_var);
   system
        .def("get_state_bounds", &system_t::get_state_bounds)
        .def("get_control_bounds", &system_t::get_control_bounds)
        .def("is_circular_topology", &system_t::is_circular_topology)
   ;

   py::class_<car_t>(m, "Car", system).def(py::init<>());
   py::class_<cart_pole_t>(m, "CartPole", system).def(py::init<>());
   // newly added cart_pole_obs
   // TODO: add init parameters
   py::class_<CartPoleObsWrapper>(m, "CartPoleObs", system)
        .def(py::init<const py::safe_array<double> &,
                      double>(),
            "obstacle_list"_a,
            "obstacle_width"_a
        );

   py::class_<pendulum_t>(m, "Pendulum", system).def(py::init<>());
   py::class_<point_t>(m, "Point", system)
       .def(py::init<int>(),
            "number_of_obstacles"_a=5
       );
   py::class_<rally_car_t>(m, "RallyCar", system).def(py::init<>());
   py::class_<two_link_acrobot_t>(m, "TwoLinkAcrobot", system).def(py::init<>());

   // Classes and interfaces for planners
   py::class_<PlannerWrapper> planner(m, "PlannerWrapper");
   planner
        .def("step_with_sample", &PlannerWrapper::step_with_sample)
        .def("step", &PlannerWrapper::step)
        .def("step_bvp", &PlannerWrapper::step_bvp)
        .def("visualize_tree", &PlannerWrapper::visualize_tree_wrapper,
            "system"_a,
            "image_width"_a=500,
            "image_height"_a=500,
            "solution_node_diameter"_a=4.,
            "solution_line_width"_a=3,
            "tree_line_width"_a=0.5
            )
        .def("visualize_nodes", &PlannerWrapper::visualize_nodes_wrapper,
            "system"_a,
            "image_width"_a=500,
            "image_height"_a=500,
            "node_diameter"_a=5,
            "solution_node_diameter"_a=4
            )
        .def("get_solution", &PlannerWrapper::get_solution)
        .def("get_number_of_nodes", &PlannerWrapper::get_number_of_nodes)
   ;

   py::class_<RRTWrapper>(m, "RRTWrapper", planner)
        .def(py::init<const py::safe_array<double>&,
                      const py::safe_array<double>&,
                      py::object,
                      const py::safe_array<double>&,
                      const py::safe_array<double>&,
                      double,
                      unsigned int>(),
            "state_bounds"_a,
            "control_bounds"_a,
            "distance"_a,
            "start_state"_a,
            "goal_state"_a,
            "goal_radius"_a,
            "random_seed"_a
        )
    ;

   py::class_<SSTWrapper>(m, "SSTWrapper", planner)
        .def(py::init<const py::safe_array<double>&,
                      const py::safe_array<double>&,
                      py::object,
                      const py::safe_array<double>&,
                      const py::safe_array<double>&,
                      double,
                      unsigned int,
                      double,
                      double>(),
            "state_bounds"_a,
            "control_bounds"_a,
            "distance"_a,
            "start_state"_a,
            "goal_state"_a,
            "goal_radius"_a,
            "random_seed"_a,
            "sst_delta_near"_a,
            "sst_delta_drain"_a
        )
   ;

    py::class_<PSOPTBVPWrapper>(m, "PSOPTBVPWrapper")
         .def(py::init<psopt_system_t&,
                       int,
                       int,
                       int>(),
             "system"_a,
             "state_dim"_a,
             "control_dim"_a,
             "random_seed"_a
         )
         .def("steerTo", &PSOPTBVPWrapper::steerTo,
             "start"_a,
             "goal"_a,
             "max_iter"_a,
             "min_time_steps"_a,
             "max_time_steps"_a,
             "integration_step"_a
         )
         .def("solve", &PSOPTBVPWrapper::solve,
             "start"_a,
             "goal"_a,
             "max_iter"_a,
             "min_time_steps"_a,
             "max_time_steps"_a,
             "integration_step"_a
         )
     ;
     py::class_<psopt_system_t> psopt_system(m, "PSOPTSystem", system);
     system
          .def("get_state_bounds", &psopt_system_t::get_state_bounds)
          .def("get_control_bounds", &psopt_system_t::get_control_bounds)
          .def("is_circular_topology", &psopt_system_t::is_circular_topology)
     ;

     py::class_<psopt_cart_pole_t>(m, "PSOPTCartPole", psopt_system).def(py::init<>());
     py::class_<psopt_pendulum_t>(m, "PSOPTPendulum", psopt_system).def(py::init<>());
     py::class_<psopt_point_t>(m, "PSOPTPoint", psopt_system).def(py::init<>());
}
