/**
* Define the system propagate function in psopt format (adouble)
*/

#ifndef PSOPT_TWO_LINK_ACROBOT_HPP
#define PSOPT_TWO_LINK_ACROBOT_HPP

#ifndef PSOPT_H
#define PSOPT_H
#include "psopt.h"
#endif

#include "bvp/psopt_system.hpp"

class psopt_acrobot_t : public psopt_system_t
{
public:
	psopt_acrobot_t()
        : psopt_system_t()
	{
		state_dimension = 4;
		control_dimension = 1;
		temp_state = new double[state_dimension];
		deriv = new double[state_dimension];
	}
	virtual ~psopt_acrobot_t(){
	    delete[] temp_state;
	    delete[] deriv;
	}
	static double distance(const double* point1, const double* point2, unsigned int);

	std::string get_name() const override;
	double max_distance() const override;
	/**
	 * @copydoc system_t::propagate(const double*, const double*, int, int, double*, double& )
	 */
	virtual bool propagate(
	    const double* start_state, unsigned int state_dimension,
        const double* control, unsigned int control_dimension,
	    int num_steps, double* result_state, double integration_step);

    static void dynamics(adouble* derivatives, adouble* path, adouble* states, adouble* controls, adouble* parameters,
             adouble& time, adouble* xad, int iphase, Workspace* workspace);
 	static adouble endpoint_cost(adouble* initial_states, adouble* final_states, adouble* parameters, adouble& t0,
             adouble& tf, adouble* xad, int iphase, Workspace* workspace);
	static adouble integrand_cost(adouble* states, adouble* controls, adouble* parameters, adouble& time, adouble* xad,
	         int iphase, Workspace* workspace);
	static void events(adouble* e, adouble* initial_states, adouble* final_states, adouble* parameters, adouble& t0,
	         adouble& tf, adouble* xad, int iphase, Workspace* workspace);
	static void linkages(adouble* linkages, adouble* xad, Workspace* workspace);

	/**
	 * @copydoc system_t::enforce_bounds()
	 */
	virtual void enforce_bounds();

	/**
	 * @copydoc system_t::valid_state()
	 */
	virtual bool valid_state();

	/**
	 * @copydoc system_t::visualize_point(double*, svg::Dimensions)
	 */
	std::tuple<double, double> visualize_point(const double* state, unsigned int state_dimension) const override;

	/**
	 * @copydoc system_t::get_state_bounds()
	 */
    std::vector<std::pair<double, double>> get_state_bounds() const override;

    /**
	 * @copydoc system_t::get_control_bounds()
	 */
    std::vector<std::pair<double, double>> get_control_bounds() const override;

    /**
	 * @copydoc system_t::is_circular_topology()
	 */
    std::vector<bool> is_circular_topology() const override;

protected:
	double* deriv;
	void update_derivative(const double* control);
};





#endif
