#include <mismix/include.h>
#include <mismix/equation_data.h>
#include <mismix/assembly_copydata.h>
#include <mismix/class.h>
#include <mismix/parameter.h>

//------------ relaxation_div_velocity ---------------------------
  
  template <int dim>
  void 
  UBC_mis_mixing<dim>::relaxation_div_velocity_step ()
  {
    pcout << "  * Relaxation Div Velocity Step.. ";
        
    matrix_auxilary = 0;
    rhs_auxilary = 0;
    
    const QGauss<dim> quadrature_formula(parameters.degree_of_pressure+1);
    
    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    MappingQ<dim> pressure_mapping (parameters.degree_of_pressure);
    MappingQ<dim> velocity_mapping (parameters.degree_of_velocity);
    MappingQ<dim> concentr_mapping (parameters.degree_of_concentr);
    
    WorkStream::
    run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                     dof_handler_auxilary.begin_active()),
         CellFilter (IteratorFilters::LocallyOwnedCell(),
                     dof_handler_auxilary.end()),
         std_cxx1x::bind (&UBC_mis_mixing<dim>::
                          local_assemble_relaxation_div_velocity_step,
                          this,
                          std_cxx1x::_1,
                          std_cxx1x::_2,
                          std_cxx1x::_3),
         std_cxx1x::bind (&UBC_mis_mixing<dim>::
                          copy_local_to_global_relaxation_div_velocity_step,
                          this,
                          std_cxx1x::_1),
         Assembly::Scratch::
         relaxation_div_velocity_step<dim> (fe_auxilary,
                                            pressure_mapping,
                                            quadrature_formula,
                                            (update_values      |
                                             update_quadrature_points |
                                             update_JxW_values   |
                                             update_gradients),
                                            fe_velocity,
                                            velocity_mapping,
                                            (update_values      |
                                             update_quadrature_points |
                                             update_gradients),
                                            concentr_fe,
                                            concentr_mapping,
                                            (update_values      |
                                            update_quadrature_points)),
         Assembly::CopyData::relaxation_div_velocity_step<dim> (fe_auxilary));

    matrix_auxilary.compress(VectorOperation::add);
    rhs_auxilary.compress(VectorOperation::add);
    
    TrilinosWrappers::MPI::Vector distributed_sol (rhs_auxilary);
    distributed_sol = aux_n_plus_1;
      
    SolverControl solver_control (matrix_auxilary.m(), parameters.eps_ns*rhs_auxilary.l2_norm ());

    SolverCG<TrilinosWrappers::MPI::Vector> cg (solver_control);

    TrilinosWrappers::PreconditionAMG preconditioner;

    TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
    Amg_data.elliptic = true;
//    if (parameters.degree_of_pressure > 1)
    Amg_data.higher_order_elements = true;
    Amg_data.smoother_sweeps = 2;
    Amg_data.aggregation_threshold = 0.02;

    preconditioner.initialize (matrix_auxilary, Amg_data);
    cg.solve (matrix_auxilary, distributed_sol, rhs_auxilary, preconditioner);
    constraints_auxilary.distribute (distributed_sol);
    aux_n_plus_1 = distributed_sol;
    
    pcout << solver_control.last_step() << std::endl;
  }
  
  template <int dim>
  void UBC_mis_mixing<dim>::
  local_assemble_relaxation_div_velocity_step (const typename DoFHandler<dim>::active_cell_iterator &cell,
             Assembly::Scratch::relaxation_div_velocity_step <dim> &scratch,
          Assembly::CopyData::relaxation_div_velocity_step <dim> &data)
  { 
    const unsigned int dofs_per_cell = scratch.fe_auxilary_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points = scratch.fe_auxilary_values.n_quadrature_points;

    scratch.fe_auxilary_values.reinit (cell);

    const FEValuesExtractors::Vector velocities (0);
    
    typename DoFHandler<dim>::active_cell_iterator
    velocity_cell (&triangulation,
                   cell->level(),
                   cell->index(),
                   &dof_handler_velocity);
    scratch.fe_velocity_values.reinit (velocity_cell);

    scratch.fe_velocity_values[velocities].get_function_gradients (vel_n_plus_1, scratch.grad_vel_n_plus_1_values);

    cell->get_dof_indices (data.local_dof_indices);
    
    data.local_matrix = 0; data.local_rhs = 0;
 
    typename DoFHandler<dim>::active_cell_iterator
    concentr_cell (&triangulation,
                   cell->level(),
                   cell->index(),
                   &concentr_dof_handler);
    scratch.concentr_fe_values.reinit (concentr_cell);
    
    scratch.concentr_fe_values.get_function_values (concentr_solution, scratch.concentr_values);
 
    std::pair<double, double> discont_variables
                              = compute_discont_variable_on_cell (n_q_points, scratch.concentr_values);
   
    for (unsigned int q=0; q<n_q_points; ++q)
    {
      double theta = 1.0 - 2.0*discont_variables.first;
      if (parameters.is_density_stable_flow == true)
        theta = 2.0*discont_variables.first - 1.0;
        
      double coeff_with_adv_term = 1.0 + theta*parameters.Atwood_number;
      double inv_coeff_with_adv_term = 1.0/coeff_with_adv_term;

      for (unsigned int k=0; k<dofs_per_cell; ++k)
      {
       scratch.phi_p[k] = scratch.fe_auxilary_values.shape_value (k,q);
       scratch.grads_phi_p[k] = scratch.fe_auxilary_values.shape_grad (k,q);
      }
      
      double mm1 = 0.0;
      for (unsigned int d=0; d<dim; ++d)
        mm1 += scratch.grad_vel_n_plus_1_values[q][d][d];
      
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
       data.local_rhs(i) += scratch.phi_p[i]*
                            mm1*
                            scratch.fe_auxilary_values.JxW(q);
     
       for (unsigned int j=0; j<dofs_per_cell; ++j)
          {
              data.local_matrix(i,i) += scratch.phi_p[i]*
                                        scratch.phi_p[j]*
                                        scratch.fe_auxilary_values.JxW(q);
           
        data.local_matrix(i,i) += cell->diameter()*
                                  parameters.coeff_relax_div_velocity*
                                  scratch.grads_phi_p[i]*
                                  scratch.grads_phi_p[j]*
                                  scratch.fe_auxilary_values.JxW(q);
          }
      }  
    }
  }
  
  template <int dim>
  void
  UBC_mis_mixing<dim>::
  copy_local_to_global_relaxation_div_velocity_step (const Assembly::CopyData::relaxation_div_velocity_step<dim> &data)
  {
    constraints_auxilary.distribute_local_to_global (data.local_matrix,
                                                     data.local_rhs,
                                                     data.local_dof_indices,
                                                     matrix_auxilary,
                                                     rhs_auxilary);
  }


  //------------ Projection ---------------------------
  
  template <int dim>
  void 
  UBC_mis_mixing<dim>::projection_step ()
  {
    pcout << "  * Projection Step.. ";
        
    matrix_auxilary = 0;
    rhs_auxilary = 0;
    
    const QGauss<dim> quadrature_formula(parameters.degree_of_pressure+1+1);
    
    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    MappingQ<dim> pressure_mapping (parameters.degree_of_pressure);
    MappingQ<dim> velocity_mapping (parameters.degree_of_velocity);
    MappingQ<dim> concentr_mapping (parameters.degree_of_concentr);
    
    WorkStream::
    run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                     dof_handler_auxilary.begin_active()),
         CellFilter (IteratorFilters::LocallyOwnedCell(),
                     dof_handler_auxilary.end()),
         std_cxx1x::bind (&UBC_mis_mixing<dim>::
                          local_assemble_projection_step,
                          this,
                          std_cxx1x::_1,
                          std_cxx1x::_2,
                          std_cxx1x::_3),
         std_cxx1x::bind (&UBC_mis_mixing<dim>::
                          copy_local_to_global_projection_step,
                          this,
                          std_cxx1x::_1),
         Assembly::Scratch::
         projection_step<dim> (fe_auxilary,
                               pressure_mapping,
                               quadrature_formula,
                               (update_values      |
                                update_quadrature_points |
                                update_JxW_values   |
                                update_gradients),
                               fe_velocity,
                               velocity_mapping,
                               (update_values      |
                                update_quadrature_points |
                                update_gradients),
                               concentr_fe,
                               concentr_mapping,
                               (update_values      |
                               update_quadrature_points)),
         Assembly::CopyData::projection_step<dim> (fe_auxilary));

    matrix_auxilary.compress(VectorOperation::add);
    rhs_auxilary.compress(VectorOperation::add);
    
    TrilinosWrappers::MPI::Vector distributed_sol (rhs_auxilary);
    distributed_sol = aux_n_plus_1;
      
    SolverControl solver_control (matrix_auxilary.m(), parameters.eps_ns*rhs_auxilary.l2_norm ());

    SolverCG<TrilinosWrappers::MPI::Vector> cg (solver_control);

    TrilinosWrappers::PreconditionAMG preconditioner;
    preconditioner.initialize (matrix_auxilary);
    cg.solve (matrix_auxilary, distributed_sol, rhs_auxilary, preconditioner);
    constraints_auxilary.distribute (distributed_sol);
    aux_n_plus_1 = distributed_sol;
    
    pcout << solver_control.last_step() << std::endl;
  }
  
  template <int dim>
  void UBC_mis_mixing<dim>::
  local_assemble_projection_step (const typename DoFHandler<dim>::active_cell_iterator &cell,
             Assembly::Scratch::projection_step <dim> &scratch,
          Assembly::CopyData::projection_step <dim> &data)
  { 
    const unsigned int dofs_per_cell = scratch.fe_auxilary_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points = scratch.fe_auxilary_values.n_quadrature_points;

    scratch.fe_auxilary_values.reinit (cell);

    const FEValuesExtractors::Vector velocities (0);
    
    typename DoFHandler<dim>::active_cell_iterator
    velocity_cell (&triangulation,
                   cell->level(),
                   cell->index(),
                   &dof_handler_velocity);
    scratch.fe_velocity_values.reinit (velocity_cell);

    scratch.fe_velocity_values[velocities].get_function_gradients (vel_n_plus_1, scratch.grad_vel_n_plus_1_values);

    cell->get_dof_indices (data.local_dof_indices);
    
    data.local_matrix = 0; data.local_rhs   = 0;
 
    typename DoFHandler<dim>::active_cell_iterator
    concentr_cell (&triangulation,
             cell->level(),
             cell->index(),
             &concentr_dof_handler);
    scratch.concentr_fe_values.reinit (concentr_cell);
    
    scratch.concentr_fe_values.get_function_values (concentr_solution, scratch.concentr_values);
    scratch.fe_auxilary_values.get_function_values (aux_n_plus_1, scratch.div_vel_values);
 
    std::pair<double, double> discont_variables = compute_discont_variable_on_cell (n_q_points,
                                                                                    scratch.concentr_values);
   
    for (unsigned int q=0; q<n_q_points; ++q)
    {
      double theta = 1.0 - 2.0*discont_variables.first;
      if (parameters.is_density_stable_flow == true)
        theta = 2.0*discont_variables.first - 1.0;
        
      double coeff_with_adv_term = 1.0 + theta*parameters.Atwood_number;
      double inv_coeff_with_adv_term = 1.0/coeff_with_adv_term;

      for (unsigned int k=0; k<dofs_per_cell; ++k)
      {
       scratch.phi_p[k] = scratch.fe_auxilary_values.shape_value (k,q);
       scratch.grads_phi_p[k] = scratch.fe_auxilary_values.shape_grad (k,q);
      }
     
      double mm1 = 0.0;
      if (std::abs(parameters.coeff_relax_div_velocity) < 1e-8)
      {
        for (unsigned int d=0; d<dim; ++d)
          mm1 += scratch.grad_vel_n_plus_1_values[q][d][d];
      }
      else {mm1 = scratch.div_vel_values[q];}
     
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
       data.local_rhs(i) += (1.5)*
                               scratch.phi_p[i]*
                      mm1*
                               (1./time_step)*
                      coeff_with_adv_term*
                      scratch.fe_auxilary_values.JxW(q);
     
       for (unsigned int j=0; j<dofs_per_cell; ++j)
        data.local_matrix(i,j) -= scratch.grads_phi_p[i]*
                         scratch.grads_phi_p[j]*
                         scratch.fe_auxilary_values.JxW(q);
      }  
    }
  }
  
  template <int dim>
  void
  UBC_mis_mixing<dim>::
  copy_local_to_global_projection_step (const Assembly::CopyData::projection_step<dim> &data)
  {
    constraints_auxilary.distribute_local_to_global (data.local_matrix,
                                                     data.local_rhs,
                                                     data.local_dof_indices,
                                                     matrix_auxilary,
                                                     rhs_auxilary);
  }

// Explicit instantiations
// template class UBC_mis_mixing<1>;
template class UBC_mis_mixing<2>;
template class UBC_mis_mixing<3>;
