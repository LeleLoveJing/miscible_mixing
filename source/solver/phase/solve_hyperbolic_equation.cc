#include <mismix/include.h>
#include <mismix/equation_data.h>
#include <mismix/assembly_copydata.h>
#include <mismix/class.h>
#include <mismix/parameter.h>



  //------------ Solve concentr Part ---------------------------
  
  template <int dim>
  void UBC_mis_mixing<dim>::
  local_assemble_concentr_matrix (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                  Assembly::Scratch::concentrMatrix<dim> &scratch,
                                  Assembly::CopyData::concentrMatrix<dim> &data)
  {
    const unsigned int dofs_per_cell = scratch.concentr_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = scratch.concentr_fe_values.n_quadrature_points;

    scratch.concentr_fe_values.reinit (cell);
    cell->get_dof_indices (data.local_dof_indices);

    data.local_mass_matrix = 0;
    data.local_stiffness_matrix = 0;

//    double coef_a1 = 1.0;
//    if (timestep_number > 1)
//      coef_a1 = (2*time_step + old_time_step) /
//                (time_step + old_time_step);

    for (unsigned int q=0; q<n_q_points; ++q)
      {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
          {
            scratch.grad_phi_T[k] = scratch.concentr_fe_values.shape_grad (k,q);
            scratch.phi_T[k]      = scratch.concentr_fe_values.shape_value (k, q);
          }

        for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            {
              data.local_mass_matrix(i,j) +=
                                             scratch.phi_T[i]*
                                             scratch.phi_T[j]*
                                             scratch.concentr_fe_values.JxW(q);
            }
      }
  }

  template <int dim>
  void
  UBC_mis_mixing<dim>::
  copy_local_to_global_concentr_matrix (const Assembly::CopyData::concentrMatrix<dim> &data)
  {
    concentr_constraints.distribute_local_to_global (data.local_mass_matrix,
                                                     data.local_dof_indices,
                                                     concentr_mass_matrix);
      
    concentr_constraints.distribute_local_to_global (data.local_stiffness_matrix,
                                                     data.local_dof_indices,
                                                     concentr_stiffness_matrix);
  }


  template <int dim>
  void UBC_mis_mixing<dim>::assemble_concentr_matrix ()
  {
    pcout << "  * Assemble Concentr Matrix.. " << std::endl;
    
    concentr_mass_matrix = 0;
    concentr_stiffness_matrix = 0;

    const QGauss<dim> quadrature_formula(parameters.degree_of_concentr+2);
    MappingQ<dim> mapping (concentr_fe.get_degree());
    
    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    WorkStream::
    run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                     concentr_dof_handler.begin_active()),
         CellFilter (IteratorFilters::LocallyOwnedCell(),
                     concentr_dof_handler.end()),
         std_cxx1x::bind (&UBC_mis_mixing<dim>::
                          local_assemble_concentr_matrix,
                          this,
                          std_cxx1x::_1,
                          std_cxx1x::_2,
                          std_cxx1x::_3),
         std_cxx1x::bind (&UBC_mis_mixing<dim>::
                          copy_local_to_global_concentr_matrix,
                          this,
                          std_cxx1x::_1),
         Assembly::Scratch::
         concentrMatrix<dim> (concentr_fe, mapping, quadrature_formula),
         Assembly::CopyData::
         concentrMatrix<dim> (concentr_fe));

    concentr_mass_matrix.compress(VectorOperation::add);
    concentr_stiffness_matrix.compress(VectorOperation::add);

    rebuild_concentr_matrices = true;
    rebuild_concentr_preconditioner = true;
    
    concentr_matrix = 0;
    
    const bool use_bdf2_scheme = (timestep_number != 0);
    
    if (use_bdf2_scheme == true)
    {
      concentr_matrix.copy_from (concentr_mass_matrix);
      concentr_matrix *= (2*time_step + old_time_step) /
                           (time_step + old_time_step);
      concentr_matrix.add (time_step, concentr_stiffness_matrix);
    }
    else
    {
      concentr_matrix.copy_from (concentr_mass_matrix);
      concentr_matrix.add (time_step, concentr_stiffness_matrix);
    }


    T_preconditioner.reset (new TrilinosWrappers::PreconditionJacobi());
    T_preconditioner->initialize (concentr_matrix);
    
  }

  template <int dim>
  void UBC_mis_mixing<dim>::
  local_assemble_concentr_rhs (   const std::pair<double,double>        global_T_range,
                                  const double                          global_max_velocity,
                                  const double                          global_entropy_variation,
                                  const typename DoFHandler<dim>::active_cell_iterator &cell,
                                  Assembly::Scratch::concentrRHS<dim>                  &scratch,
                                  Assembly::CopyData::concentrRHS<dim>                 &data)
  {    
    const bool use_bdf2_scheme = (timestep_number != 0);

    const unsigned int dofs_per_cell = scratch.concentr_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = scratch.concentr_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector velocities (0);

    data.local_rhs = 0;
    data.matrix_for_bc = 0;
    cell->get_dof_indices (data.local_dof_indices);

    scratch.concentr_fe_values.reinit (cell);

    typename DoFHandler<dim>::active_cell_iterator
    vel_cell (&triangulation,
              cell->level(),
              cell->index(),
              &dof_handler_velocity);
    scratch.fe_velocity_values.reinit (vel_cell);

    scratch.concentr_fe_values.get_function_values (old_concentr_solution,
                                                    scratch.old_concentr_values);
    scratch.concentr_fe_values.get_function_values (old_old_concentr_solution,
                                                    scratch.old_old_concentr_values);
    scratch.concentr_fe_values.get_function_gradients (old_concentr_solution,
                                                       scratch.old_concentr_grads);
    scratch.concentr_fe_values.get_function_gradients (old_old_concentr_solution,
                                                       scratch.old_old_concentr_grads);
    scratch.concentr_fe_values.get_function_laplacians (old_concentr_solution,
                                                        scratch.old_concentr_laplacians);
    scratch.concentr_fe_values.get_function_laplacians (old_old_concentr_solution,
                                                        scratch.old_old_concentr_laplacians);

    scratch.fe_velocity_values[velocities].get_function_values (vel_star,
                                                               scratch.old_velocity_values);
    scratch.fe_velocity_values[velocities].get_function_values (vel_star_old,
                                                               scratch.old_old_velocity_values);
    scratch.fe_velocity_values[velocities].get_function_symmetric_gradients (vel_star,
                                                                             scratch.old_strain_rates);
    scratch.fe_velocity_values[velocities].get_function_symmetric_gradients (vel_star_old,
                                                                             scratch.old_old_strain_rates);

    double nu
      = compute_entropy_viscosity_for_hyperbolic (scratch.old_concentr_values,
                                                  scratch.old_old_concentr_values,
                                                  scratch.old_concentr_grads,
                                                  scratch.old_old_concentr_grads,
                                                  scratch.old_concentr_laplacians,
                                                  scratch.old_old_concentr_laplacians,
                                                  scratch.old_velocity_values,
                                                  scratch.old_old_velocity_values,
                                                  scratch.old_strain_rates,
                                                  scratch.old_old_strain_rates,
                                                  global_max_velocity,
                                                  global_T_range.second - global_T_range.first,
                                                  0.5 * (global_T_range.second + global_T_range.first),
                                                  global_entropy_variation,
                                                  cell->diameter());
//
//    max_entropy_viscosity = std::max(max_entropy_viscosity, nu);
//    max_entropy_viscosity = Utilities::MPI::max (max_entropy_viscosity, MPI_COMM_WORLD);
//    nu = 0.001;
    
    for (unsigned int q=0; q<n_q_points; ++q)
      {
        for (unsigned int k=0; k<dofs_per_cell; ++k)
          {
            scratch.phi_T[k]      = scratch.concentr_fe_values.shape_value (k, q);
            scratch.grad_phi_T[k] = scratch.concentr_fe_values.shape_grad (k, q);
          }


        const double T_term_for_rhs
          = (use_bdf2_scheme ?
             (scratch.old_concentr_values[q] *
              (1 + time_step/old_time_step)
              -
              scratch.old_old_concentr_values[q] *
              (time_step * time_step) /
              (old_time_step * (time_step + old_time_step)))
             :
             scratch.old_concentr_values[q]);

        const double ext_T
          = (use_bdf2_scheme ?
             (scratch.old_concentr_values[q] *
              (1 + time_step/old_time_step)
              -
              scratch.old_old_concentr_values[q] *
              time_step/old_time_step)
             :
             scratch.old_concentr_values[q]);

        const Tensor<1,dim> ext_grad_T
          = (use_bdf2_scheme ?
             (scratch.old_concentr_grads[q] *
              (1 + time_step/old_time_step)
              -
              scratch.old_old_concentr_grads[q] *
              time_step/old_time_step)
             :
             scratch.old_concentr_grads[q]);

        const Tensor<1,dim> extrapolated_u
          = (use_bdf2_scheme ?
             (scratch.old_velocity_values[q] *
              (1 + time_step/old_time_step)
              -
              scratch.old_old_velocity_values[q] *
              time_step/old_time_step)
             :
             scratch.old_velocity_values[q]);

        const SymmetricTensor<2,dim> extrapolated_strain_rate
          = (use_bdf2_scheme ?
             (scratch.old_strain_rates[q] *
              (1 + time_step/old_time_step)
              -
              scratch.old_old_strain_rates[q] *
              time_step/old_time_step)
             :
             scratch.old_strain_rates[q]);

        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            data.local_rhs(i) += (T_term_for_rhs * scratch.phi_T[i]
                                  -
                                  time_step *
                                  extrapolated_u * ext_grad_T * scratch.phi_T[i]
                                  -
                                  time_step *
                                  nu * ext_grad_T * scratch.grad_phi_T[i])*
                                  scratch.concentr_fe_values.JxW(q);

//            if (concentr_constraints.is_inhomogeneously_constrained(data.local_dof_indices[i]))
              {
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                  data.matrix_for_bc(j,i) += (scratch.phi_T[i] * scratch.phi_T[j] *
                                             (use_bdf2_scheme ?
                                             ((2*time_step + old_time_step) /
                                             (time_step + old_time_step)) : 1.))*
                                             scratch.concentr_fe_values.JxW(q);
              }
          }
      }
  }


  template <int dim>
  void
  UBC_mis_mixing<dim>::
  copy_local_to_global_concentr_rhs (const Assembly::CopyData::concentrRHS<dim> &data)
  {
    concentr_constraints.distribute_local_to_global (data.local_rhs,
                                                     data.local_dof_indices,
                                                     concentr_rhs,
                                                     data.matrix_for_bc);
  }

  template <int dim>
  void UBC_mis_mixing<dim>::assemble_concentr_system (const double maximal_velocity)
  {
 pcout << "  * Assemble Concentr System.. ";

    const bool use_bdf2_scheme = (timestep_number != 0);
      
    concentr_rhs = 0;

    const QGauss<dim> quadrature_formula(parameters.degree_of_concentr+2);
    const std::pair<double,double>
    global_T_range = get_extrapolated_concentr_range();

    const double average_concentr = 0.5 * (global_T_range.first +
                                           global_T_range.second);
    const double global_entropy_variation = get_entropy_variation (average_concentr);

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator> CellFilter;

    MappingQ<dim> mapping (concentr_fe.get_degree());
      
    WorkStream::
    run (
         CellFilter (IteratorFilters::LocallyOwnedCell(),
                     concentr_dof_handler.begin_active()),
         CellFilter (IteratorFilters::LocallyOwnedCell(),
                     concentr_dof_handler.end()),
         std_cxx1x::bind (&UBC_mis_mixing<dim>::
                          local_assemble_concentr_rhs,
                          this,
                          global_T_range,
                          maximal_velocity,
                          global_entropy_variation,
                          std_cxx1x::_1,
                          std_cxx1x::_2,
                          std_cxx1x::_3),
         std_cxx1x::bind (&UBC_mis_mixing<dim>::
                          copy_local_to_global_concentr_rhs,
                          this,
                          std_cxx1x::_1),
         Assembly::Scratch::concentrRHS<dim>  (concentr_fe, fe_velocity, mapping, quadrature_formula),
         Assembly::CopyData::concentrRHS<dim> (concentr_fe)
         );
        
    concentr_rhs.compress(VectorOperation::add);
  }

  template <int dim>
  void
  UBC_mis_mixing<dim>::solve_concentr_equation (const double maximal_velocity)
  {

     old_old_concentr_solution = old_concentr_solution;
     old_concentr_solution     = concentr_solution;
      
     assemble_concentr_system (maximal_velocity);
      
     SolverControl solver_control (concentr_matrix.m(),
                                   1e-10*concentr_rhs.l2_norm());
     SolverCG<TrilinosWrappers::MPI::Vector>   cg (solver_control);

     TrilinosWrappers::MPI::Vector
         distributed_concentr_solution (concentr_rhs);
     distributed_concentr_solution = concentr_solution;

     cg.solve (concentr_matrix, distributed_concentr_solution,
               concentr_rhs, *T_preconditioner);

     concentr_constraints.distribute (distributed_concentr_solution);

     std::pair<unsigned int, unsigned int> range_i = distributed_concentr_solution.local_range();
   
//     for (unsigned int i=0 ; i<distributed_concentr_solution.size(); ++i)
//     {
//       if (distributed_concentr_solution.in_local_range(i))
//       {
//         double temp_value = distributed_concentr_solution[i];
//         if (temp_value > 1.000000) distributed_concentr_solution[i] = 1.0;
//         if (temp_value < -0.00000) distributed_concentr_solution[i] = 0.0;
//       }
//     }

     concentr_solution = distributed_concentr_solution;

//        TrilinosWrappers::MPI::Vector distr_solution (concentr_rhs);
//        distr_solution = concentr_solution;
//        TrilinosWrappers::MPI::Vector distr_old_solution (concentr_rhs);
//        distr_old_solution = old_old_concentr_solution;
//        distr_solution .sadd (1.+time_step/old_time_step, -time_step/old_time_step,
//                              distr_old_solution);
//        concentr_solution = distr_solution;

     pcout << solver_control.last_step() << std::endl;

//     pcout << "  * Max. Entropy Viscosity  = " << max_entropy_viscosity << std::endl;



  }
// Explicit instantiations
// template class UBC_mis_mixing<1>;
template class UBC_mis_mixing<2>;
template class UBC_mis_mixing<3>;
