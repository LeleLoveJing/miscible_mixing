#include <mismix/include.h>
#include <mismix/equation_data.h>
#include <mismix/assembly_copydata.h>
#include <mismix/class.h>
#include <mismix/parameter.h>
  
  template <int dim>
  UBC_mis_mixing<dim>::UBC_mis_mixing (Parameters &parameters_)
    :
    parameters (parameters_),
    pcout (std::cout,
           (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
            == 0)),
    triangulation (MPI_COMM_WORLD,
      typename Triangulation<dim>::MeshSmoothing
      (Triangulation<dim>::smoothing_on_refinement |
       Triangulation<dim>::smoothing_on_coarsening)),
    fe_velocity (FE_Q<dim>(parameters.degree_of_velocity), dim),
    fe_pressure (FE_Q<dim>(parameters.degree_of_pressure)),
    fe_auxilary (FE_Q<dim>(parameters.degree_of_pressure)),
    concentr_fe (FE_Q<dim>(parameters.degree_of_concentr)),
    fe_error    (FE_DGQ<dim>(parameters.degree_of_concentr)),
    dof_handler_velocity (triangulation),
    dof_handler_pressure (triangulation),
    dof_handler_auxilary (triangulation),
    concentr_dof_handler (triangulation),
    dof_handler_error    (triangulation),
    time_step (0.0),
    old_time_step (0.0),
    computed_time_step (0.0),
    timestep_number (0),
    max_entropy_viscosity (std::numeric_limits<double>::min()),
    rebuild_concentr_matrices (true),
    rebuild_concentr_preconditioner (true),
    computing_timer (pcout,
                     TimerOutput::summary,
                     TimerOutput::wall_times)
  {}
  
// Explicit instantiations
// template class UBC_mis_mixing<1>;
template class UBC_mis_mixing<2>;
template class UBC_mis_mixing<3>;
