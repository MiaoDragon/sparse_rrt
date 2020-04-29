/**
 * @file rrt.cpp
 *
 * @copyright Software License Agreement (BSD License)
 * Original work Copyright (c) 2014, Rutgers the State University of New Jersey, New Brunswick
 * Modified work Copyright 2017 Oleg Y. Sinyavskiy
 * All Rights Reserved.
 * For a full description see the file named LICENSE.
 *
 * Original authors: Zakary Littlefield, Kostas Bekris
 * Modifications by: Oleg Y. Sinyavskiy
 *
 */

#include "motion_planners/rrt.hpp"
#include "nearest_neighbors/graph_nearest_neighbors.hpp"
#include "bvp/psopt_bvp.hpp"
#include "bvp/psopt_system.hpp"

#include <iostream>
#include <deque>


rrt_node_t::rrt_node_t(double* point, unsigned int state_dimension, rrt_node_t* a_parent, tree_edge_t&& a_parent_edge, double a_cost)
	    : tree_node_t(point, state_dimension, std::move(a_parent_edge), a_cost)
	    , parent(a_parent)
{
}

rrt_node_t::~rrt_node_t() {

}


void rrt_t::get_solution(std::vector<std::vector<double>>& solution_path, std::vector<std::vector<double>>& controls, std::vector<double>& costs)
{
    std::vector<proximity_node_t*> close_nodes = metric.find_delta_close_and_closest(goal_state, goal_radius);

    double length = std::numeric_limits<double>::max();;
    for(unsigned i=0;i<close_nodes.size();i++)
    {
        rrt_node_t* v = (rrt_node_t*)(close_nodes[i]->get_state());
        double temp = v->get_cost() ;
        if( temp < length)
        {
            length = temp;
            nearest = v;
        }
    }
    //now nearest should be the closest node to the goal state
    if(this->distance(goal_state,nearest->get_point(), this->state_dimension) < goal_radius)
    {
        std::deque<const rrt_node_t*> path;
        while(nearest->get_parent()!=NULL)
        {
            path.push_front(nearest);
            nearest = nearest->get_parent();
        }

        std::vector<double> root_state;
        for (unsigned c=0; c<this->state_dimension; c++) {
            root_state.push_back(root->get_point()[c]);
        }
        solution_path.push_back(root_state);

        for(unsigned i=0;i<path.size();i++)
        {
            std::vector<double> current_state;
            for (unsigned c=0; c<this->state_dimension; c++) {
                current_state.push_back(path[i]->get_point()[c]);
            }
            solution_path.push_back(current_state);

            std::vector<double> current_control;
            for (unsigned c=0; c<this->control_dimension; c++) {
                current_control.push_back(path[i]->get_parent_edge().get_control()[c]);
            }
            controls.push_back(current_control);
            costs.push_back(path[i]->get_parent_edge().get_duration());
        }
    }
}

void rrt_t::nearest_state(const double* state, std::vector<double> &res_state)
{
    // find the nearest node in the tree to the state, and copy to res_state
    rrt_node_t* nearest = nearest_vertex(state);
    const double* nearest_state = nearest->get_point();
    for (unsigned i=0; i < this->state_dimension; i++)
    {
        res_state[i] = nearest_state[i];
    }
}

void rrt_t::step_with_sample(system_interface* system, double* sample_state, double* from_state, double* new_state, double* new_control, double& new_time, int min_time_steps, int max_time_steps, double integration_step)
{
    /* @Author: Yinglong Miao
     * Given the random sample from some sampler
     * Find the closest existing node
     * Generate random control
     * Propagate for random time with constant random control from the closest node
     * If resulting state is valid, add a resulting state into the tree and perform sst-specific graph manipulations
     */

  this->random_control(new_control);

  nearest = nearest_vertex(sample_state);
  for (unsigned i=0; i<this->state_dimension; i++)
  {
      from_state[i] = nearest->get_point()[i];
  }
  int num_steps = this->random_generator.uniform_int_random(min_time_steps, max_time_steps);
  new_time = num_steps*integration_step;
  if(system->propagate(
	  nearest->get_point(), this->state_dimension, new_control, this->control_dimension,
	  num_steps, new_state, integration_step))
  {
	  //create a new tree node
	  rrt_node_t* new_node = static_cast<rrt_node_t*>(nearest->add_child(new rrt_node_t(
		  new_state, this->state_dimension, nearest,
		  tree_edge_t(new_control, this->control_dimension, new_time),
		  nearest->get_cost() + new_time)
	  ));
	  metric.add_node(new_node);
	  number_of_nodes++;
  }
}


void rrt_t::step(system_interface* system, int min_time_steps, int max_time_steps, double integration_step)
{
    double* sample_state = new double[this->state_dimension];
    double* sample_control = new double[this->control_dimension];

    this->random_state(sample_state);
    this->random_control(sample_control);

    nearest = nearest_vertex(sample_state);
    int num_steps = this->random_generator.uniform_int_random(min_time_steps, max_time_steps);
    double duration = num_steps*integration_step;
    if(system->propagate(
        nearest->get_point(), this->state_dimension, sample_control, this->control_dimension,
        num_steps, sample_state, integration_step))
    {
        //create a new tree node
        rrt_node_t* new_node = static_cast<rrt_node_t*>(nearest->add_child(new rrt_node_t(
            sample_state, this->state_dimension, nearest,
            tree_edge_t(sample_control, this->control_dimension, duration),
            nearest->get_cost() + duration)
        ));
        metric.add_node(new_node);
        number_of_nodes++;
    }
    delete sample_state;
    delete sample_control;
}



void rrt_t::step_bvp(system_interface* propagate_system, psopt_system_t* bvp_system, psopt_result_t& step_res, const double* start_state, const double* goal_state, int num_iters, int num_steps, double step_sz,
    std::vector<std::vector<double>> &x_init,
    std::vector<std::vector<double>> &u_init,
    std::vector<double> &t_init)
{
    /**
    * solve BVP(x_start, x_goal, x_init, u_init, t_init) -> xs, us, ts
    * propagate and add to tree
    **/
    rrt_node_t* nearest = nearest_vertex(start_state);
    if (bvp_solver == NULL)
    {
        bvp_solver = new PSOPT_BVP(bvp_system, this->state_dimension, this->control_dimension);
    }
    psopt_result_t res;
    bvp_solver->solve(res, start_state, goal_state, num_steps, num_iters, step_sz, step_sz*(num_steps-1), \
                      x_init, u_init, t_init);
    std::vector<std::vector<double>> x_traj = res.x;
    std::vector<std::vector<double>> u_traj = res.u;
    std::vector<double> t_traj;
    for (unsigned i=0; i < num_steps-1; i+=1)
    {
        t_traj.push_back(res.t[i+1] - res.t[i]);
    }

    // try to connect from nearest to input_sample_state
    // convert from double array to VectorXd
    rrt_node_t* x_tree = nearest;
    //double* x_traj_i = new double[this->state_dimension];
    double* state_t = new double[this->state_dimension];
    double* end_state = new double[this->state_dimension];
    double* u_traj_i = new double[this->control_dimension];

    std::vector<double> res_x_i;
    for (unsigned k=0; k < this->state_dimension; k++)
    {
        res_x_i.push_back(x_tree->get_point()[k]);
        state_t[k] = start_state[k];
    }
    step_res.x.push_back(res_x_i);
    bool val = true;
    double res_t;
    for (unsigned i=0; i < num_steps-1; i++)
    {
        int num_dis = std::floor(t_traj[i] / step_sz);
        res_t = t_traj[i] - num_dis * step_sz;
        for (unsigned j=0; j < this->control_dimension; j++)
        {
            u_traj_i[j] = u_traj[i][j];
        }

        for (unsigned j=0; j < num_dis; j++)
        {
            val = propagate_system->propagate(state_t, this->state_dimension, u_traj_i, this->control_dimension,
					  1, end_state, step_sz);
            // add the new state to tree
            if (!val)
            {
                // not valid state, no point going further, not adding to tree, stop right here
                break;
            }
            std::vector<double> res_x_i;
            std::vector<double> res_u_i;
            for (unsigned k=0; k < this->state_dimension; k++)
            {
                res_x_i.push_back(end_state[k]);
            }
            for (unsigned k=0; k < this->control_dimension; k++)
            {
                res_u_i.push_back(u_traj_i[k]);
            }
            step_res.x.push_back(res_x_i);
            step_res.u.push_back(res_u_i);
            step_res.t.push_back(step_sz);
            for (unsigned k=0; k < this->state_dimension; k++)
            {
                state_t[k] = end_state[k];
            }
        }
        if (!val)
        {
            break;
        }
        val = propagate_system->propagate(state_t, this->state_dimension, u_traj_i, this->control_dimension,
                  1, end_state, res_t);
        if (!val)
        {
            break;
        }
        std::vector<double> res_x_i;
        std::vector<double> res_u_i;
        for (unsigned k=0; k < this->state_dimension; k++)
        {
            res_x_i.push_back(end_state[k]);
            state_t[k] = end_state[k];
        }
        for (unsigned k=0; k < this->control_dimension; k++)
        {
            res_u_i.push_back(u_traj_i[k]);
        }
        step_res.x.push_back(res_x_i);
        step_res.u.push_back(res_u_i);
        step_res.t.push_back(res_t);

    }
    // add the last valid node to tree
    //sst_node_t* new_x_tree = add_to_tree(state_t, u_traj_i, x_tree, res_t);
	//create a new tree node
	rrt_node_t* new_x_tree = static_cast<rrt_node_t*>(x_tree->add_child(new rrt_node_t(
		state_t, this->state_dimension, x_tree,
		tree_edge_t(u_traj_i, this->control_dimension, res_t),
		x_tree->get_cost() + res_t)
	));
	metric.add_node(new_x_tree);
    x_tree = new_x_tree;

    delete u_traj_i;
    delete end_state;
}

void rrt_t::step_bvp(psopt_system_t* system, int min_time_steps, int max_time_steps, double integration_step)
{
	/**
    * generate random sample
    **/
    double* sample_state = new double[this->state_dimension];
	this->random_state(sample_state);
	nearest = nearest_vertex(sample_state);

    // try to connect from nearest to input_sample_state
    // convert from double array to VectorXd
    const double* start_x = nearest->get_point();
    double* end_x = sample_state;
    int num_steps = 6*this->state_dimension;
    //int num_steps = 6*this->state_dimension;
    // initialize bvp pointer if it is nullptr
    if (bvp_solver == NULL)
    {
        bvp_solver = new PSOPT_BVP(system, this->state_dimension, this->control_dimension);
    }

    //OptResults res = bvp_solver->solve(start_x, end_x, 100);
    psopt_result_t res;
    bvp_solver->solve(res, start_x, end_x, num_steps, 100, integration_step*num_steps, 50*max_time_steps*integration_step*num_steps);
    std::vector<std::vector<double>> x_traj = res.x;
    std::vector<std::vector<double>> u_traj = res.u;
    std::vector<double> t_traj;
    for (unsigned i=0; i < num_steps-1; i+=1)
    {
        t_traj.push_back(res.t[i+1] - res.t[i]);
  	  //std::cout << "t_traj[" << i << "]: " << res.t[i+1] - res.t[i] << std::endl;
    }
    //TODO: do something with the trajectories
    // simulate forward using the action trajectory, regardless if the traj opt is successful or not
    rrt_node_t* x_tree = nearest;
    // double* result_x = new double[this->state_dimension];
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

            // todo: we can also use larger step for adding
            bool val = system->propagate(x_tree->get_point(), this->state_dimension, control_ptr, this->control_dimension,
                             time_step, sample_state, integration_step);
             //std::cout << "after propagation... val: " << val << std::endl;
            // add the new state to tree
            if (!val)
            {
                // not valid state, no point going further, not adding to tree, stop right here
                x_tree = NULL;
                break;
            }
  		  {
  	          //create a new tree node
  	          rrt_node_t* new_node = static_cast<rrt_node_t*>(x_tree->add_child(new rrt_node_t(
  	              sample_state, this->state_dimension, x_tree,
  	              tree_edge_t(control_ptr, this->control_dimension, time_step*integration_step),
  	              x_tree->get_cost() + time_step*integration_step)
  	          ));
  	          metric.add_node(new_node);
  			  x_tree = new_node;
  	          number_of_nodes++;
  	      }
            if (!x_tree)
            {
                break;
            }

        }
        if (!x_tree)
        {
            break;
        }

    }
    //std::cout << "after creating new nodes" << std::endl;


	delete[] sample_state;
}



rrt_node_t* rrt_t::nearest_vertex(const double* state) const
{
    double distance;
    return (rrt_node_t*)(metric.find_closest(state, &distance)->get_state());
}
