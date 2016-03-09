#include <mismix/include.h>
#include <mismix/equation_data.h>
#include <mismix/assembly_copydata.h>
#include <mismix/class.h>
#include <mismix/parameter.h>

  template <int dim>
  void
  UBC_mis_mixing<dim>::solve_fluid_equation(double increase_dimless_vel)
  {
    diffusion_step (increase_dimless_vel);
    
    if (std::abs(parameters.coeff_relax_div_velocity) > 1e-8)
      relaxation_div_velocity_step ();
  
    projection_step ();
    pressure_rot_step ();
    extrapolation_step ();    
  }
  

  template <int dim>
  void
  UBC_mis_mixing<dim>::solution_update ()
  {
      vel_n_minus_minus_1 = vel_n_minus_1;
      vel_n_minus_1 = vel_n;
      vel_n = vel_n_plus_1;

      pre_n_minus_1 = pre_n;
      pre_n = pre_n_plus_1;

      aux_n_minus_1 = aux_n;
      aux_n = aux_n_plus_1;
  }
    

// Explicit instantiations
// template class UBC_mis_mixing<1>;
template class UBC_mis_mixing<2>;
template class UBC_mis_mixing<3>;
