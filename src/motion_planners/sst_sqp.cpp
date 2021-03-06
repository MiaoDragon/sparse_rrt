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
#include "motion_planners/sst.hpp"
#include "nearest_neighbors/graph_nearest_neighbors.hpp"
#include "bvp/sqp_bvp.hpp"
#include <sco/optimizers.hpp>

#include <Eigen/Dense>

#include <iostream>
#include <deque>

using namespace Eigen;

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
		return;
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


void sst_t::step_with_sample(system_interface* system, double* sample_state, double* new_state, int min_time_steps, int max_time_steps, double integration_step)
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
  sst_node_t* nearest = nearest_vertex(sample_state);

  // try to connect from nearest to input_sample_state
  // convert from double array to VectorXd
  VectorXd start_x(this->state_dimension);
  VectorXd end_x(this->state_dimension);
  for (unsigned i=0; i < this->state_dimension; i++)
  {
      start_x(i) = nearest->get_point()[i];
      end_x(i) = sample_state[i];
  }
  //std::vector<std::vector<double>>
  int num_steps = 3*this->state_dimension;
  //int num_steps = 6*this->state_dimension;
  // initialize bvp pointer if it is nullptr
  if (bvp_solver == NULL)
  {
      bvp_solver = new SQPBVP_forward(system, this->state_dimension, this->control_dimension, num_steps, integration_step);
  }

  OptResults res = bvp_solver->solve(start_x, end_x, 100);
  std::vector<double> solution(res.x);
  std::cout << "after creating solution variable" << std::endl;
  // from solution we can obtain the trajectory: state traj | action traj | time traj
  std::vector<std::vector<double>> x_traj;
  std::vector<std::vector<double>> u_traj;
  std::vector<double> t_traj;
  int control_start = num_steps*this->state_dimension;
  int duration_start = control_start + (num_steps-1)*this->control_dimension;
  for (unsigned i=0; i < num_steps-1; i++)
  {
      // states
      int begin_idx = i*this->state_dimension;
      int end_idx = (i+1)*this->state_dimension;
      std::vector<double> x(solution.begin()+begin_idx, solution.begin()+end_idx);
      x_traj.push_back(x);
      // controls
      begin_idx = i*this->control_dimension+control_start;
      end_idx = (i+1)*this->control_dimension+control_start;
      std::vector<double> u(solution.begin()+begin_idx, solution.begin()+end_idx);
      u_traj.push_back(u);
      // time
      t_traj.push_back(solution[duration_start+i]);
  }
  std::cout << "after inserting into trajectory." << std::endl;
  //TODO: do something with the trajectories
  // simulate forward using the action trajectory, regardless if the traj opt is successful or not
  sst_node_t* x_tree = nearest;
  // double* result_x = new double[this->state_dimension];
  for (unsigned i=0; i < num_steps-1; i++)
  {
      if (t_traj[i] < integration_step / 2)
      {
          // the time step is too small, ignore this action
          continue;
      }
      int num_steps = std::round(t_traj[i] / integration_step);
      double* control_ptr = u_traj[i].data();
      system->propagate(x_tree->get_point(), this->state_dimension, control_ptr, this->control_dimension,
                       num_steps, new_state, integration_step);
        std::cout << "after propagation..." << std::endl;
       // add the new state to tree
       sst_node_t* new_x_tree = add_to_tree(new_state, control_ptr, x_tree, num_steps*integration_step);
       std::cout << "after adding into tree" << std::endl;
       x_tree = new_x_tree;
       // if the created tree node is nullptr, stop right there
       if (!x_tree)
       {
           break;
       }

  }
  std::cout << "after creating new nodes" << std::endl;
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
    delete sample_state;
    delete sample_control;
}

sst_node_t* sst_t::nearest_vertex(const double* sample_state)
{
	//performs the best near query
    std::vector<proximity_node_t*> close_nodes = metric.find_delta_close_and_closest(sample_state, this->sst_delta_near);

    double length = std::numeric_limits<double>::max();;
    sst_node_t* nearest = nullptr;
    for(unsigned i=0;i<close_nodes.size();i++)
    {
        tree_node_t* v = (tree_node_t*)(close_nodes[i]->get_state());
        double temp = v->get_cost() ;
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
