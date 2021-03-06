/**
 * @file sst.cpp
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

//## TODO
//  add eigen dependency to the package
//#define DEBUG 1
//#define DEBUG_BRANCH_BOUND 1
#include "motion_planners/sst.hpp"
#include "nearest_neighbors/graph_nearest_neighbors.hpp"
#include "bvp/psopt_bvp.hpp"
#include "bvp/psopt_system.hpp"

#include <iostream>
#include <deque>
#include <time.h>
sst_node_t::sst_node_t(const double* point, unsigned int state_dimension, sst_node_t* a_parent, tree_edge_t&& a_parent_edge, double a_cost)
    : tree_node_t(point, state_dimension, std::move(a_parent_edge), a_cost)
    , parent(a_parent)
    , active(true)
    , witness(NULL)
{

}

sst_node_t::~sst_node_t() {

}


sample_node_t::sample_node_t(
    sst_node_t* const representative,
    const double* a_point, unsigned int state_dimension)
    : state_point_t(a_point, state_dimension)
    , rep(representative)
{

}

sample_node_t::~sample_node_t()
{

}

sst_t::sst_t(
    const double* in_start, const double* in_goal,
    double in_radius,
    const std::vector<std::pair<double, double> >& a_state_bounds,
    const std::vector<std::pair<double, double> >& a_control_bounds,
    std::function<double(const double*, const double*, unsigned int)> a_distance_function,
    unsigned int random_seed,
    double delta_near, double delta_drain)
    : planner_t(in_start, in_goal, in_radius,
                a_state_bounds, a_control_bounds, a_distance_function, random_seed)
    , best_goal(nullptr)
    , sst_delta_near(delta_near)
    , sst_delta_drain(delta_drain)
{
    //initialize the metrics
    unsigned int state_dimensions = this->get_state_dimension();
    std::function<double(const double*, const double*)> raw_distance =
        [state_dimensions, a_distance_function](const double* s0, const double* s1) {
            return a_distance_function(s0, s1, state_dimensions);
        };
    metric.set_distance(raw_distance);

    root = new sst_node_t(in_start, a_state_bounds.size(), nullptr, tree_edge_t(nullptr, 0, -1.), 0.);
    metric.add_node(root);
    number_of_nodes++;

    samples.set_distance(raw_distance);

    sample_node_t* first_witness_sample = new sample_node_t(static_cast<sst_node_t*>(root), start_state, this->state_dimension);
    samples.add_node(first_witness_sample);
    witness_nodes.push_back(first_witness_sample);
    // initialize BVP solver
    bvp_solver = NULL;
    //std::cout << "initialized SST Wrapper" << std::endl;
}

sst_t::~sst_t() {
    delete root;
    for (auto w: this->witness_nodes) {
        delete w;
    }
    if (bvp_solver)
    {
        // if bvp_solver pointer is not NULL
        delete bvp_solver;
    }
}


void sst_t::get_solution(std::vector<std::vector<double>>& solution_path, std::vector<std::vector<double>>& controls, std::vector<double>& costs)
{
	if(best_goal==NULL)
    {
        //std::cout << "haven't found a path yet." << std::endl;
        //std::cout << "number of nodes: " << number_of_nodes << std::endl;
    	return;
    }
	sst_node_t* nearest_path_node = best_goal;

	//now nearest_path_node should be the closest node to the goal state
	std::deque<sst_node_t*> path;
	while(nearest_path_node->get_parent()!=NULL)
	{
		path.push_front(nearest_path_node);
        nearest_path_node = nearest_path_node->get_parent();
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

int sst_t::add_to_tree_public(system_interface* system, const double* sample_state, const double* sample_control, int num_steps, double integration_step)
{
    sst_node_t* nearest = nearest_vertex(sample_state);
    double* new_state = new double[this->state_dimension];
    std::cout << "sample state:" << std::endl;
    std::cout << sample_state[0] << ", "<< sample_state[1] << ", "<< sample_state[2] << ", "<< sample_state[3] << std::endl;

    std::cout << "nearest_state:" << std::endl;
    std::cout << nearest->get_point()[0] << ", "<< nearest->get_point()[1] << ", "<< nearest->get_point()[2] << ", "<< nearest->get_point()[3] << std::endl;


    if(system->propagate(
        nearest->get_point(), this->state_dimension, sample_control, this->control_dimension,
        num_steps, new_state, integration_step))
    {
        std::cout << "propagte ok!" << std::endl;
        std::cout << "new state:" << std::endl;
        std::cout << new_state[0] << ", "<< new_state[1] << ", "<< new_state[2] << ", "<< new_state[3] << std::endl;
        sst_node_t* res = add_to_tree(new_state, sample_control, nearest, num_steps*integration_step);
        if (res != NULL)
        {
            std::cout << "add_to_tree ok" << std::endl;
            delete new_state;
            return 1;
        }
    }
    delete new_state;
    return 0;
}

void sst_t::step_with_sample(system_interface* system, double* sample_state, double* from_state, double* new_state, double* new_control, double& new_time, int min_time_steps, int max_time_steps, double integration_step)
{
    /* @Author: Yinglong Miao
     * Given the random sample from some sampler
     * Find the closest existing node
     * Generate random control
     * Propagate for random time with constant random control from the closest node
     * If resulting state is valid, add a resulting state into the tree and perform sst-specific graph manipulations
     */


	//this->random_state(sample_state);
  // sample a bunch of controls, and choose the one with the minimum distance to the sample_state
  // remember the sample state by a temperate Variable
  //std::cout << "inside step_with_sample" << std::endl;

  this->random_control(new_control);
  sst_node_t* nearest = nearest_vertex(sample_state);
  for (unsigned i=0; i<this->state_dimension; i++)
  {
      from_state[i] = nearest->get_point()[i];
  }
  int num_steps = this->random_generator.uniform_int_random(min_time_steps, max_time_steps);
  new_time = 0.;
  //std::cout << "before propagating in C++" << std::endl;

  ///**
  //#### below is the previous working case: if collision then throw the entire trajectory
  if(system->propagate(
      nearest->get_point(), this->state_dimension, new_control, this->control_dimension,
      num_steps, new_state, integration_step))
  {
      new_time = num_steps*integration_step;
      add_to_tree(new_state, new_control, nearest, new_time);
  }
  else
  {
      new_time = 0.; // not added to the tree
  }
  return;
  //#####
  //*/





 // below propagate every step until collision happens

 double* past_valid_state = new double[this->state_dimension];
 for (unsigned i=0; i<this->state_dimension; i++)
 {
     past_valid_state[i] = nearest->get_point()[i]; // starting point
 }
 int propagated_step = 0;
 for (unsigned t=0; t<num_steps; t++)
 {
     // obtain the propagation result
     bool val = system->propagate(
         past_valid_state, this->state_dimension, new_control, this->control_dimension,
         1, new_state, integration_step);
    //std::cout << "propagation step: " << t << std::endl;
    if (t==0 && !val)
    {
        // if iteration is 0 and the propagation fails (didn't step at all)
        // return failure
        delete past_valid_state;
        //std::cout << "after step_with_sample, failed" << std::endl;

        return;
    }
    if (!val)
    {
        // if the propagation is not valid, then add the last valid point to tree
        new_time = propagated_step*integration_step;
        add_to_tree(past_valid_state, new_control, nearest, new_time);
        // set the past_valid_state to new_state
        for (unsigned i=0; i<this->state_dimension; i++)
        {
            new_state[i] = past_valid_state[i];
        }
        // return success
        //std::cout << "step_with_sample return: new_time: " << new_time << std::endl;
        delete past_valid_state;
        //std::cout << "after step_with_sample" << std::endl;

        return;
    }
    // otherwise update the past_valid_state
    for (unsigned i=0; i<this->state_dimension; i++)
    {
        past_valid_state[i] = new_state[i];
    }
    propagated_step += 1;  // valid propagation +1
 }
 // success after all propagation
 // then add the last valid point to tree
 new_time = propagated_step*integration_step;
 add_to_tree(past_valid_state, new_control, nearest, new_time);
 // set the past_valid_state to new_state
 for (unsigned i=0; i<this->state_dimension; i++)
 {
     new_state[i] = past_valid_state[i];
 }
 // return success
 delete past_valid_state;
 //std::cout << "after step_with_sample" << std::endl;
 return;



  //std::cout << "after step in C++" << std::endl;
}


void sst_t::step(system_interface* system, int min_time_steps, int max_time_steps, double integration_step)
{
    /*
     * Generate a random sample
     * Find the closest existing node
     * Generate random control
     * Propagate for random time with constant random control from the closest node
     * If resulting state is valid, add a resulting state into the tree and perform sst-specific graph manipulations
     */
    //std::cout << "start of step  in C++" << std::endl;
    double* sample_state = new double[this->state_dimension];
    double* sample_control = new double[this->control_dimension];
	this->random_state(sample_state);
	this->random_control(sample_control);
    sst_node_t* nearest = nearest_vertex(sample_state);
	int num_steps = this->random_generator.uniform_int_random(min_time_steps, max_time_steps);
    double duration = num_steps*integration_step;
    //std::cout << "before propagating in C++" << std::endl;
	if(system->propagate(
	    nearest->get_point(), this->state_dimension, sample_control, this->control_dimension,
	    num_steps, sample_state, integration_step))
	{
		add_to_tree(sample_state, sample_control, nearest, duration);
	}
    //std::cout << "after step in C++" << std::endl;
    delete sample_state;
    delete sample_control;
}

void sst_t::step_with_output(system_interface* system, int min_time_steps, int max_time_steps, double integration_step, double* steer_start, double* steer_goal)
{
    /*
     * Generate a random sample
     * Find the closest existing node
     * Generate random control
     * Propagate for random time with constant random control from the closest node
     * If resulting state is valid, add a resulting state into the tree and perform sst-specific graph manipulations
     */
    double* sample_state = new double[this->state_dimension];
    double* sample_control = new double[this->control_dimension];
	this->random_state(sample_state);
	this->random_control(sample_control);
    sst_node_t* nearest = nearest_vertex(sample_state);
	int num_steps = this->random_generator.uniform_int_random(min_time_steps, max_time_steps);
    double duration = num_steps*integration_step;
	if(system->propagate(
	    nearest->get_point(), this->state_dimension, sample_control, this->control_dimension,
	    num_steps, sample_state, integration_step))
	{
		add_to_tree(sample_state, sample_control, nearest, duration);
	}
    // copy to output
    for (unsigned i=0; i <this->state_dimension; i++)
    {
        steer_start[i] = nearest->get_point()[i];
        steer_goal[i] = sample_state[i];
    }

    delete sample_state;
    delete sample_control;
}


void sst_t::nearest_state(const double* state, std::vector<double> &res_state)
{
    // find the nearest node in the tree to the state, and copy to res_state
    sst_node_t* nearest = nearest_vertex(state);
    //std::cout << "in nearest_state: " << std::endl;
    //std::cout << "state=" << "[" << state[0] << ", " << state[1] << ", " << state[2] << ", " << state[3]<< "]"  << std::endl;
    const double* nearest_state = nearest->get_point();
    for (unsigned i=0; i < this->state_dimension; i++)
    {
        res_state[i] = nearest_state[i];
    }
}



void sst_t::step_bvp(system_interface* propagate_system, psopt_system_t* bvp_system, psopt_result_t& step_res, const double* start_state, const double* goal_state, int psopt_num_iters, int psopt_num_steps, double psopt_step_sz,
    double step_sz,
    std::vector<std::vector<double>> &x_init,
    std::vector<std::vector<double>> &u_init,
    std::vector<double> &t_init)
{
    /**
    * solve BVP(x_start, x_goal, x_init, u_init, t_init) -> xs, us, ts
    * propagate and add to tree
    **/
    //std::cout << "inside sst: step_bvp" << std::endl;
    //std::cout << "system->state_dim: " << bvp_system->get_state_dimension() << std::endl;

    // start_state: input sampled state, may not be in the stored tree
    // here we find the nearest node to the start_state, and solve the bvp with the nearest_node
    // try to connect from nearest to sampled goal state
    // convert from double array to VectorXd

    sst_node_t* nearest = nearest_vertex(start_state);
    sst_node_t* x_tree = nearest;
    //double* x_traj_i = new double[this->state_dimension];
    double* state_t = new double[this->state_dimension];
    // copy the nearest => state_t as our bvp starting point
    for (unsigned i=0; i < this->state_dimension; i++)
    {
        state_t[i] = x_tree->get_point()[i];
    }

    if (bvp_solver == NULL)
    {
        bvp_solver = new PSOPT_BVP(bvp_system, this->state_dimension, this->control_dimension);
    }
    psopt_result_t res;
    //std::cout << "sst: before solve... "<< std::endl;
    clock_t begin_time;
    begin_time = clock();
    //std::cout << "bvp goal_state= [" << goal_state[0] << ", " << goal_state[1] << ", " << goal_state[2] << ", " << goal_state[3] <<"]" << std::endl;

    bvp_solver->solve(res, state_t, goal_state, psopt_num_steps, psopt_num_iters, psopt_step_sz, psopt_step_sz*(psopt_num_steps-1), \
                      x_init, u_init, t_init);
    std::cout << "psopt_num_iters: " << psopt_num_iters << std::endl;
    //std::cout << "step_bvp: solve time: " << float( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;

    std::cout << "sst: after solve. "<< std::endl;

    std::vector<std::vector<double>> x_traj = res.x;
    std::vector<std::vector<double>> u_traj = res.u;
    std::vector<double> t_traj;
    #ifdef DEBUG
    std::cout << "solution of bvp solver: t_traj" << std::endl;
    #endif

    for (unsigned i=0; i < psopt_num_steps-1; i+=1)
    {
        t_traj.push_back(res.t[i+1] - res.t[i]);
        #ifdef DEBUG
        std::cout << "res.t[" << i << "]: " << res.t[i] << std::endl;
        std::cout << "t_traj[" << i << "]: " << t_traj[i] << std::endl;
        #endif
        std::cout << "x_traj:" << std::endl;
        for (unsigned j=0; j < 4; j++)
        {
            std::cout << x_traj[i][j] << std::endl;

        }

        std::cout << "u_traj:" << std::endl;
        std::cout << u_traj[i][0] << std::endl;
        std::cout <<  "t_traj:" << std::endl;
        std::cout << t_traj[i] << std::endl;
    }




    double* end_state = new double[this->state_dimension];
    double* u_traj_i = new double[this->control_dimension];


    // propagating to obtain the trajectory that follows the dynamics
    // should use starting point as the nearest tree, i.e. state_t
    std::vector<double> res_x_i;
    for (unsigned k=0; k < this->state_dimension; k++)
    {
        res_x_i.push_back(state_t[k]);
    }
    step_res.x.push_back(res_x_i);
    bool val = true; // if valid propagation or not
    double res_t;
    double total_t = 0.;
    //std::cout << "sst: after copying res "<< std::endl;
    //std::cout << "number of nodes: " << number_of_nodes << std::endl;

    for (unsigned i=0; i < psopt_num_steps-1; i++)
    {
        std::cout << "i = " << i << std::endl;
        int num_dis = std::round(t_traj[i] / step_sz);
        std::cout << "num_dis: " << num_dis << std::endl;
        #ifdef DEBUG
        std::cout << "step_bvp propagating..." << std::endl;
        std::cout << "i=" << i << std::endl;
        std::cout << "num_dis=" << num_dis << ", t_traj[i]=" << t_traj[i] << std::endl;
        std::cout << "res_t=" << res_t << std::endl;
        #endif
        for (unsigned j=0; j < this->control_dimension; j++)
        {
            u_traj_i[j] = u_traj[i][j];
        }
        int num_valid_steps = 0;
        for (unsigned int j=0; j < num_dis; j++)
        {
            val = propagate_system->propagate(state_t, this->state_dimension, u_traj_i, this->control_dimension,
					  1, end_state, step_sz);
            // add the new state to tree
            if (!val)
            {
                // not valid state, no point going further, not adding to tree, stop right here
                #ifdef DEBUG
                std::cout << "invalid propagation in step_bvp" << std::endl;
                std::cout << "j=" << j << std::endl;
                #endif
                break;
            }
            #ifdef DEBUG
            std::cout << "after propagation in step_bvp" << std::endl;
            std::cout << "num_dis=" << num_dis << "," <<  "j=" << j << std::endl;
            std::cout << "start_state=" << "[" << state_t[0] << ", " << state_t[1] << ", " << state_t[2] << ", " << state_t[3]<< "]"  << std::endl;
            std::cout << "goal_state=" << "[" << end_state[0] << ", " << end_state[1] << ", " << end_state[2] << ", " << end_state[3]<< "]"  << std::endl;
            #endif
            num_valid_steps += 1;
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
        std::cout << "num_valid_steps: " << num_valid_steps << std::endl;
        if (num_valid_steps > 0)
        {
            // add the last valid node to tree, with the same control
            sst_node_t* new_x_tree = bvp_add_to_tree_without_opt(state_t, u_traj_i, x_tree, num_valid_steps*step_sz);
            total_t += num_dis*step_sz;
            x_tree = new_x_tree;
        }
        if (!val)
        {
            break;
        }


    }
    // add the last valid node to tree, with the same control for t_traj[i] time
    if (total_t > 0.)
    {
        // if at least move on step further, add to representative states
        bvp_make_representative(state_t, x_tree);
        //metric.add_node(x_tree);

    }


    if (total_t == 0.)
    {
        // random propagate
        double* sample_state = new double[this->state_dimension];
        double* sample_control = new double[this->control_dimension];
        this->random_control(sample_control);
        int sst_num_steps = this->random_generator.uniform_int_random(5, 100);
        double duration = sst_num_steps*step_sz;
        //std::cout << "before propagating in C++" << std::endl;
        if(propagate_system->propagate(
            nearest->get_point(), this->state_dimension, sample_control, this->control_dimension,
            sst_num_steps, sample_state, step_sz))
        {
            add_to_tree(sample_state, sample_control, nearest, duration);

            std::vector<double> res_x_i;
            std::vector<double> res_u_i;
            for (unsigned k=0; k < this->state_dimension; k++)
            {
                res_x_i.push_back(sample_state[k]);
            }
            for (unsigned k=0; k < this->control_dimension; k++)
            {
                res_u_i.push_back(sample_control[k]);
            }
            step_res.x.push_back(res_x_i);
            step_res.u.push_back(res_u_i);
            step_res.t.push_back(duration);

        }
        //std::cout << "after step in C++" << std::endl;
        delete sample_state;
        delete sample_control;

    }
    delete u_traj_i;
    delete end_state;
    //std::cout << "after sst: step_bvp" << std::endl;
}


sst_node_t* sst_t::nearest_vertex(const double* sample_state)
{
	//performs the best near query
    //std::cout << "sst: nearest_vertex" << std::endl;
    //std::cout << "sample_state =" << "[" << sample_state[0] << ", " << sample_state[1] << ", " << sample_state[2] << ", " << sample_state[3]<< "]"  << std::endl;

    //std::cout << "sst_delta_near: " << sst_delta_near << std::endl;

    std::vector<proximity_node_t*> close_nodes = metric.find_delta_close_and_closest(sample_state, this->sst_delta_near);
    //std::cout << "close_nodes len: " << close_nodes.size() << std::endl;
    double length = std::numeric_limits<double>::max();;
    sst_node_t* nearest = nullptr;
    for(unsigned i=0;i<close_nodes.size();i++)
    {
        tree_node_t* v = (tree_node_t*)(close_nodes[i]->get_state());
        double temp = v->get_cost() ;
        //std::cout << "nearest_vertex[ " << i << "] =" << "[" << v->get_point()[0] << ", " << v->get_point()[1] << ", " << v->get_point()[2] << ", " << v->get_point()[3]<< "]"  << std::endl;
        //std::cout << "cost: " << temp << std::endl;

        if( temp < length)
        {
            length = temp;
            nearest = (sst_node_t*)v;
        }
    }
    assert (nearest != nullptr);
    return nearest;
}

sst_node_t* sst_t::add_to_tree(const double* sample_state, const double* sample_control, sst_node_t* nearest, double duration)
{
	//check to see if a sample exists within the vicinity of the new node
    sample_node_t* witness_sample = find_witness(sample_state);

    sst_node_t* representative = witness_sample->get_representative();
	if(representative==NULL || representative->get_cost() > nearest->get_cost() + duration)
	{
		if(best_goal==NULL || nearest->get_cost() + duration <= best_goal->get_cost())
		{
			//create a new tree node
			//set parent's child
			sst_node_t* new_node = static_cast<sst_node_t*>(nearest->add_child(
			    new sst_node_t(
                    sample_state, this->state_dimension,
                    nearest,
                    tree_edge_t(sample_control, this->control_dimension, duration),
                    nearest->get_cost() + duration)
            ));
			number_of_nodes++;

            //std::cout << "node1: [" << new_node->get_point()[0] << ", " << new_node->get_point()[1] << ", " << new_node->get_point()[2] << ", " << new_node->get_point()[3] << "]" << std::endl;
            //std::cout << "node2: [" << goal_state[0] << ", " << goal_state[1] << ", " << goal_state[2] << ", " << goal_state[3] << "]" << std::endl;
            //std::cout << "distance (node1, node2): " << this->distance(new_node->get_point(), goal_state, this->state_dimension) << std::endl;

	        if(best_goal==NULL && this->distance(new_node->get_point(), goal_state, this->state_dimension)<goal_radius)
	        {
	        	best_goal = new_node;
	        	branch_and_bound((sst_node_t*)root);
	        }
	        else if(best_goal!=NULL && best_goal->get_cost() > new_node->get_cost() &&
	                this->distance(new_node->get_point(), goal_state, this->state_dimension)<goal_radius)
	        {
	        	best_goal = new_node;
	        	branch_and_bound((sst_node_t*)root);
	        }

            // Acquire representative again - it can be different
            representative = witness_sample->get_representative();
			if(representative!=NULL)
			{
				//optimization for sparsity
				if(representative->is_active())
				{
					metric.remove_node(representative);
					representative->make_inactive();
				}

	            sst_node_t* iter = representative;
	            while( is_leaf(iter) && !iter->is_active() && !is_best_goal(iter))
	            {
	                sst_node_t* next = (sst_node_t*)iter->get_parent();
	                remove_leaf(iter);
	                iter = next;
	            }

			}
			witness_sample->set_representative(new_node);
			new_node->set_witness(witness_sample);
			metric.add_node(new_node);

            // return the pointer to the new node
            return new_node;
		}
	}
    return NULL;
}

sst_node_t* sst_t::bvp_add_to_tree_without_opt(const double* sample_state, const double* sample_control, sst_node_t* nearest, double duration)
{
    sst_node_t* new_node = static_cast<sst_node_t*>(nearest->add_child(
        new sst_node_t(
            sample_state, this->state_dimension,
            nearest,
            tree_edge_t(sample_control, this->control_dimension, duration),
            nearest->get_cost() + duration)
    ));
    new_node->make_inactive(); // make_inactive because they are only intermediate nodes
    number_of_nodes++;
    /**
    // we don't check if the itermermediate nodes are goal or not
    if(best_goal==NULL && this->distance(new_node->get_point(), goal_state, this->state_dimension)<goal_radius)
    {
        best_goal = new_node;
        branch_and_bound((sst_node_t*)root);
    }
    else if(best_goal!=NULL && best_goal->get_cost() > new_node->get_cost() &&
            this->distance(new_node->get_point(), goal_state, this->state_dimension)<goal_radius)
    {
        best_goal = new_node;
        branch_and_bound((sst_node_t*)root);
    }
    */
    return new_node;
}

void sst_t::bvp_make_representative(const double* sample_state, sst_node_t* node)
{
    // check if the node is valid ()
    //check to see if a sample exists within the vicinity of the new node
    sample_node_t* witness_sample = find_witness(sample_state);

    sst_node_t* representative = witness_sample->get_representative();
    //std::cout << "distance to goal: " << this->distance(node->get_point(), goal_state, this->state_dimension) << std::endl;
    if (representative == NULL)
    {
    //    std::cout << "sst_make_representative: representative is NULL" << std::endl;
    }
    else
    {
        //std::cout << "representative node: [" << representative->get_point()[0] << ", " << representative->get_point()[1] << ", " << representative->get_point()[2] << ", " << representative->get_point()[3] <<"]" << std::endl;
        //std::cout << "node: [" << node->get_point()[0] << ", " << node->get_point()[1] << ", " << node->get_point()[2] << ", " << node->get_point()[3] <<"]" << std::endl;
        //std::cout << "representative cost: " << representative->get_cost() << std::endl;
        //std::cout << "node cost: " << node->get_cost() << std::endl;

    }
	if(representative==NULL || representative->get_cost() > node->get_cost())
	{
		if(best_goal==NULL || node->get_cost() <= best_goal->get_cost())
		{
			//passed the test
            node->make_active();
	        if(best_goal==NULL && this->distance(node->get_point(), goal_state, this->state_dimension)<goal_radius)
	        {
	        	best_goal = node;
	        	branch_and_bound((sst_node_t*)root);
	        }
	        else if(best_goal!=NULL && best_goal->get_cost() > node->get_cost() &&
	                this->distance(node->get_point(), goal_state, this->state_dimension)<goal_radius)
	        {
	        	best_goal = node;
	        	branch_and_bound((sst_node_t*)root);
	        }

            // Acquire representative again - it can be different
            representative = witness_sample->get_representative();
			if(representative!=NULL)
			{
                //std::cout << "sst_make_representative: second witness, representative is not NULL" << std::endl;
				//optimization for sparsity
				if(representative->is_active())
				{
                    //std::cout << "removing nodes from metric" << std::endl;
					metric.remove_node(representative);
					representative->make_inactive();
				}

	            sst_node_t* iter = representative;
	            while( is_leaf(iter) && !iter->is_active() && !is_best_goal(iter))
	            {
	                sst_node_t* next = (sst_node_t*)iter->get_parent();
	                remove_leaf(iter);
	                iter = next;
	            }

			}
            // adding new node to nearest_neighbors
            //std::cout << "adding new node to nearest_neighbors" << std::endl;
			witness_sample->set_representative(node);
			node->set_witness(witness_sample);
			metric.add_node(node);
            //std::cout << "node state: [" << node->get_point()[0] << ", " << node->get_point()[1] << ", " << node->get_point()[2] << ", " << node->get_point()[3] <<"]" << std::endl;

		}
	}
    else
    {
        //std::cout << "making the node inactive because representative invalid" << std::endl;
        // failed the prior test, then remove it
        //optimization for sparsity
        node->make_inactive();
        sst_node_t* iter = node;
        while( is_leaf(iter) && !iter->is_active() && !is_best_goal(iter))
        {
            sst_node_t* next = (sst_node_t*)iter->get_parent();
            remove_leaf(iter);
            iter = next;
        }
    }
}



sample_node_t* sst_t::find_witness(const double* sample_state)
{
	double distance;
    sample_node_t* witness_sample = (sample_node_t*)samples.find_closest(sample_state, &distance)->get_state();
	if(distance > this->sst_delta_drain)
	{
		//create a new sample
		witness_sample = new sample_node_t(NULL, sample_state, this->state_dimension);
		samples.add_node(witness_sample);
		witness_nodes.push_back(witness_sample);
	}
    return witness_sample;
}

void sst_t::branch_and_bound(sst_node_t* node)
{
    // Copy children becuase apparently, they are going to be modified
    std::list<tree_node_t*> children = node->get_children();
    for (std::list<tree_node_t*>::const_iterator iter = children.begin(); iter != children.end(); ++iter)
    {
    	branch_and_bound((sst_node_t*)(*iter));
    }
    if(is_leaf(node) && node->get_cost() > best_goal->get_cost())
    {
    	if(node->is_active())
    	{
            #ifdef DEBUG_BRANCH_BOUND
            // print node point
            std::cout << "branch_and_bound state=" << "[" << node->get_point()[0] << ", " << node->get_point()[1] << ", " << node->get_point()[2] << ", " << node->get_point()[3]<< "]"  << std::endl;
            if (node->get_witness() == NULL)
            {
                std::cout << "witness is NULL" << std::endl;
            }
            #endif
	    	node->get_witness()->set_representative(NULL);
	    	metric.remove_node(node);
	    }
    	remove_leaf(node);
    }
}

bool sst_t::is_leaf(tree_node_t* node)
{
	return node->is_leaf();
}

void sst_t::remove_leaf(sst_node_t* node)
{
	if(node->get_parent() != NULL)
	{
		node->get_parent_edge();
		node->get_parent()->remove_child(node);
		number_of_nodes--;
		delete node;
	}
}

bool sst_t::is_best_goal(tree_node_t* v)
{
	if(best_goal==NULL)
		return false;
    sst_node_t* new_v = best_goal;

    while(new_v->get_parent()!=NULL)
    {
        if(new_v == v)
            return true;

        new_v = new_v->get_parent();
    }
    return false;

}
