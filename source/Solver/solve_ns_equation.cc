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
  UBC_mis_mixing<dim>::extrapolation_step ()
  {
    pcout << "  * Extrapolation Step.. " << std::endl;

    TrilinosWrappers::MPI::Vector
      dist_vel_n_plus_1 (rhs_velocity),
      dist_vel_n (rhs_velocity), dist_vel_n_minus_1 (rhs_velocity),
      dist_vel_n_minus_minus_1 (rhs_velocity), dist_vel_star (rhs_velocity),
      dist_vel_star_old (rhs_velocity);
   
    dist_vel_n_plus_1 = vel_n_plus_1; dist_vel_n = vel_n; dist_vel_n_minus_1 = vel_n_minus_1;
    dist_vel_star = vel_star, dist_vel_star_old = vel_star_old;

    dist_vel_star.sadd (0.0, 2.0, dist_vel_n_plus_1);
    dist_vel_star.sadd (1.0, -1.0, dist_vel_n);
     
    dist_vel_star_old.sadd (0.0, 2.0, dist_vel_n);
    dist_vel_star_old.sadd (1.0, -1.0, dist_vel_n_minus_1);
    
    vel_star = dist_vel_star;
    vel_star_old = dist_vel_star_old;
    


  }
  
  //------------ Diffusion ---------------------------
  
  template <int dim>
  void 
  UBC_mis_mixing<dim>::diffusion_step (double increase_dimless_vel)
  {
    pcout << "  * Diffusion Step.. Assemble Start.. " << increase_dimless_vel << std::endl;
    
    matrix_velocity = 0;
    rhs_velocity = 0;
    
    const QGauss<dim> quadrature_formula(parameters.degree_of_velocity+1);
    
    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    MappingQ<dim> velocity_mapping (parameters.degree_of_velocity);
    MappingQ<dim> pressure_mapping (parameters.degree_of_pressure);
    MappingQ<dim> concentr_mapping (parameters.degree_of_concentr);
    
    WorkStream::
    run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                     dof_handler_velocity.begin_active()),
         CellFilter (IteratorFilters::LocallyOwnedCell(),
                     dof_handler_velocity.end()),
         std_cxx1x::bind (&UBC_mis_mixing<dim>::
                          local_assemble_diffusion_step,
                          this,
                          std_cxx1x::_1,
                          std_cxx1x::_2,
                          std_cxx1x::_3),
         std_cxx1x::bind (&UBC_mis_mixing<dim>::
                          copy_local_to_global_diffusion_step,
                          this,
                          std_cxx1x::_1),
         Assembly::Scratch::diffusion_step<dim> (fe_velocity, 
                                                 velocity_mapping,
                                                 quadrature_formula,
                                                 (update_values   |
                                                  update_quadrature_points   |
                                                  update_JxW_values   |
                                                  update_gradients    |
                                                  update_hessians),
                                                 fe_pressure,
                                                 pressure_mapping,
                                                 (update_values      |
                                                  update_quadrature_points   |
                                                  update_gradients  |
                                                  update_hessians),
                                                  concentr_fe,
                                                  concentr_mapping,
                                                 (update_values      |
                                                  update_quadrature_points   |
                                                  update_gradients)),
                                                   Assembly::CopyData::diffusion_step<dim> (fe_velocity));

    matrix_velocity.compress(VectorOperation::add);
    rhs_velocity.compress(VectorOperation::add);
 
    pcout << "  * Diffusion Step.. Assemble End.. " << increase_dimless_vel << std::endl;

    std::map<unsigned int,double> boundary_values;
    std::vector<bool> vel_prof(dim, true);

    TrilinosWrappers::MPI::Vector distributed_sol (rhs_velocity);
    distributed_sol = vel_n_plus_1;
      
    VectorTools::interpolate_boundary_values (dof_handler_velocity,
                                              3,
                                              EquationData::
                                              Inflow_Velocity<dim> (increase_dimless_vel, 1),
                                              boundary_values);

    MatrixTools::apply_boundary_values (boundary_values,
                                        matrix_velocity,
                                        distributed_sol,
                                        rhs_velocity,
                                        false);

    pcout << "  * Diffusion Step.. Solve.. ";

    SolverControl solver_control (matrix_velocity.m(), 
                      parameters.eps_ns*rhs_velocity.l2_norm ());

    SolverGMRES<TrilinosWrappers::MPI::Vector>
      gmres (solver_control,SolverGMRES<TrilinosWrappers::MPI::Vector>::
          AdditionalData(parameters.kry_size));

    TrilinosWrappers::PreconditionILU preconditioner;
    preconditioner.initialize (matrix_velocity);
    gmres.solve (matrix_velocity, distributed_sol, rhs_velocity, preconditioner);

    constraints_velocity.distribute (distributed_sol);
    vel_n_plus_1 = distributed_sol;
    
    pcout << solver_control.last_step() << std::endl;
  }
  
  template <int dim>
  void UBC_mis_mixing<dim>::
  local_assemble_diffusion_step (const typename DoFHandler<dim>::active_cell_iterator &cell,
     Assembly::Scratch::diffusion_step <dim> &scratch,
     Assembly::CopyData::diffusion_step <dim> &data)
  { 
    const unsigned int dofs_per_cell = scratch.fe_velocity_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points = scratch.fe_velocity_values.n_quadrature_points;
    const FEValuesExtractors::Vector velocities (0);
    scratch.fe_velocity_values.reinit (cell);
    
    typename DoFHandler<dim>::active_cell_iterator
    pressure_cell (&triangulation,
                   cell->level(),
                   cell->index(),
                   &dof_handler_pressure);
    scratch.fe_pressure_values.reinit (pressure_cell);
    
    typename DoFHandler<dim>::active_cell_iterator
    concentr_cell (&triangulation,
                   cell->level(),
                   cell->index(),
                   &concentr_dof_handler);
    scratch.concentr_fe_values.reinit (concentr_cell);
    
    cell->get_dof_indices (data.local_dof_indices);
    
    typename DoFHandler<dim>::active_cell_iterator
    error_cell (&triangulation,
                cell->level(),
                cell->index(),
                &dof_handler_error);

    std::vector<types::global_dof_index>  error_local_dof_indices (fe_error.dofs_per_cell);
    error_cell->get_dof_indices (error_local_dof_indices);

    data.local_matrix = 0; data.local_rhs = 0;

    scratch.fe_velocity_values[velocities].get_function_values (vel_star, scratch.vel_star_values);
    scratch.fe_velocity_values[velocities].get_function_laplacians (vel_star, scratch.laplacian_vel_star_values);
    scratch.fe_velocity_values[velocities].get_function_gradients (vel_star, scratch.grad_vel_star_values);

    scratch.fe_velocity_values[velocities].get_function_values (vel_n, scratch.vel_n_values);
    scratch.fe_velocity_values[velocities].get_function_values (vel_n_minus_1, scratch.vel_n_minus_1_values);
    
    scratch.fe_pressure_values.get_function_gradients (aux_n, scratch.grad_aux_n_values);
    scratch.fe_pressure_values.get_function_gradients (aux_n_minus_1, scratch.grad_aux_n_minus_1_values);
    scratch.fe_pressure_values.get_function_gradients (pre_n, scratch.grad_pre_n_values);

    scratch.fe_pressure_values.get_function_hessians (aux_n, scratch.grad_grad_aux_n_values);
    scratch.fe_pressure_values.get_function_hessians (aux_n_minus_1, scratch.grad_grad_aux_n_minus_1_values);
    scratch.fe_pressure_values.get_function_hessians (pre_n, scratch.grad_grad_pre_n_values);
   
    scratch.fe_pressure_values.get_function_values (aux_n, scratch.aux_n_values);
    scratch.fe_pressure_values.get_function_values (aux_n_minus_1, scratch.aux_n_minus_1_values);
    scratch.fe_pressure_values.get_function_values (pre_n, scratch.pre_n_values);
   
    scratch.concentr_fe_values.get_function_values (concentr_solution, scratch.concentr_values);
   
//    // For local artificial viscosity
//    std::vector<Tensor<1, dim> > vel_energy_values (n_q_points);
//    std::vector<Tensor<1, dim> > old_vel_energy_values (n_q_points);
//    std::vector<Tensor<2, dim> > grad_vel_energy_values (n_q_points);

    std::pair<double, double> discont_variables = compute_discont_variable_on_cell (n_q_points,
                                                                                    scratch.concentr_values);
 
    std::vector<Tensor<1, dim> > div_vel_values (n_q_points);
    std::vector<Tensor<1, dim> > projected_grad_pressure_values (n_q_points);
   
    for (unsigned int q=0; q<n_q_points; ++q)
    {
      for (unsigned int d=0; d<dim; ++d)
        div_vel_values[q][d] = scratch.grad_vel_star_values [q][d][d];
     
      projected_grad_pressure_values[q] = scratch.grad_pre_n_values[q] +
                                          (4./3.)*scratch.grad_aux_n_values[q] -
                                          (1./3.)*scratch.grad_aux_n_minus_1_values[q];
    }

    double theta = 1.0 - 2.0*discont_variables.first;
    if (parameters.is_density_stable_flow == true)
      theta = 2.0*discont_variables.first - 1.0;

    double coeff_with_adv_term = 1.0 + theta*parameters.Atwood_number;
    double inv_coeff_with_adv_term = 1./coeff_with_adv_term;

    Tensor <1, dim> source_vector;
    for (unsigned int d=0; d<dim; ++d)
      source_vector[d] = parameters.inclined_angle_vector[d];
    double coeff1_for_adv = coeff_with_adv_term;
    double coeff2_for_visco =  discont_variables.second/parameters.Reynolds_number;
    double coeff3_for_source = (theta/(parameters.Froude_number*parameters.Froude_number));

    std::pair<double, double> entropy_pair =
      compute_entropy_viscosity_for_navier_stokes(scratch.vel_n_values,
                                                  scratch.vel_n_minus_1_values,
                                                  scratch.grad_vel_star_values,
                                                  scratch.laplacian_vel_star_values,
                                                  scratch.grad_pre_n_values,
                                                  source_vector,
                                                  coeff1_for_adv,
                                                  coeff2_for_visco,
                                                  coeff3_for_source,
                                                  cell->diameter());

    for (unsigned int k=0; k<fe_error.dofs_per_cell; ++k)
    {
      entropy_viscosity_for_ns [error_local_dof_indices[k]] = entropy_pair.second;
      energy_norm_for_ns       [error_local_dof_indices[k]] = entropy_pair.first;
    }

    for (unsigned int q=0; q<n_q_points; ++q)
    {
      for (unsigned int k=0; k<dofs_per_cell; ++k)
      {
        scratch.phi_u[k] = scratch.fe_velocity_values[velocities].value (k,q);
        scratch.grads_phi_u[k] = scratch.fe_velocity_values[velocities].gradient (k,q);
        scratch.symm_grads_phi_u[k] = scratch.fe_velocity_values[velocities].symmetric_gradient(k,q);
        scratch.divergence_phi_u[k] = scratch.fe_velocity_values[velocities].divergence (k,q);
      }
            
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        unsigned int component_i = fe_velocity.system_to_base_index(i).first.first;
       
        //Time-stepping
        data.local_rhs(i) -= scratch.phi_u[i]*
                             (-2.0*scratch.vel_n_values[q]+
                              0.5*scratch.vel_n_minus_1_values[q])*
                              scratch.fe_velocity_values.JxW(q);

        //Gradient of pressure
        data.local_rhs(i) -= inv_coeff_with_adv_term*
                             time_step*
                             scratch.phi_u[i]*
                             projected_grad_pressure_values[q]*
                             scratch.fe_velocity_values.JxW(q);

        //body-force term : bouyant and surface tension
        data.local_rhs(i) += inv_coeff_with_adv_term*
                             theta*
                             time_step*
                             (1.0/(parameters.Froude_number*parameters.Froude_number))*
                             parameters.inclined_angle_vector*
                             scratch.phi_u[i]*
                             scratch.fe_velocity_values.JxW(q);

        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
         unsigned int component_j = fe_velocity.system_to_base_index(j).first.first;

         //Time-stepping
         data.local_matrix(i,j) += 1.5*
                                   scratch.phi_u[i]*
                                   scratch.phi_u[j]*
                                   scratch.fe_velocity_values.JxW(q);

         //Convective-Term
         {
           data.local_matrix(i,j) += time_step*
                                     scratch.grads_phi_u[j]*
                                     scratch.vel_star_values[q]*
                                     scratch.phi_u[i]*
                                     scratch.fe_velocity_values.JxW(q);

           double mm1 = 0.0;
           for (unsigned int d=0; d<dim; ++d)
             mm1 += scratch.grad_vel_star_values [q][d][d];

           data.local_matrix(i,j) += time_step*
                                     0.5*
                                     scratch.phi_u[i]*
                                     mm1*
                                     scratch.phi_u[j]*
                                     scratch.fe_velocity_values.JxW(q);
         }

         //Viscous term
         data.local_matrix(i,j) += inv_coeff_with_adv_term*
                                   time_step*
                                   (discont_variables.second)*
                                   (2.0/parameters.Reynolds_number)*
                                   scratch.symm_grads_phi_u[i]*
                                   scratch.symm_grads_phi_u[j]*
                                   scratch.fe_velocity_values.JxW(q);

         // Artificial Viscous Term
         data.local_matrix(i,j) += inv_coeff_with_adv_term*
                                   time_step*
                                   entropy_pair.second*
                                   scratch.symm_grads_phi_u[i]*
                                   scratch.symm_grads_phi_u[j]*
                                   scratch.fe_velocity_values.JxW(q);

         // Grad-Div Stablization Term
         // Check the Nan value and need to adaptive coefficient
         data.local_matrix(i,j) += inv_coeff_with_adv_term*
                                   time_step*
                                   parameters.coeff_gamma_grad_div*
                                   scratch.divergence_phi_u[i]*
                                   scratch.divergence_phi_u[j]*
                                   scratch.fe_velocity_values.JxW(q);

        }//j-loop
      }//i-loop
    }//quadrature-loop  
  }
  
  template <int dim>
  void
  UBC_mis_mixing<dim>::
  copy_local_to_global_diffusion_step (const Assembly::CopyData::diffusion_step<dim> &data)
  {
    constraints_velocity.distribute_local_to_global ( data.local_matrix,
                                                      data.local_rhs,
                                                      data.local_dof_indices,
                                                      matrix_velocity,
                                                      rhs_velocity);
  }  
  
  
  //------------ Pressure Correction With Rotation --------------------------- 
  
  
  template <int dim>
  void 
  UBC_mis_mixing<dim>::pressure_rot_step ()
  {
    pcout << "  * Correction Step.. ";
    
    matrix_pressure = 0;
    rhs_pressure = 0;

    const QGauss<dim> quadrature_formula(parameters.degree_of_pressure+1);

    typedef
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
    CellFilter;

    MappingQ<dim> pressure_mapping (parameters.degree_of_pressure);
    MappingQ<dim> velocity_mapping (parameters.degree_of_velocity);
    MappingQ<dim> concentr_mapping (parameters.degree_of_concentr);
    
    WorkStream::
    run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                     dof_handler_pressure.begin_active()),
         CellFilter (IteratorFilters::LocallyOwnedCell(),
                     dof_handler_pressure.end()),
         std_cxx1x::bind (&UBC_mis_mixing<dim>::
                          local_assemble_pressure_rot_step,
                          this,
                          std_cxx1x::_1,
                          std_cxx1x::_2,
                          std_cxx1x::_3),
         std_cxx1x::bind (&UBC_mis_mixing<dim>::
                          copy_local_to_global_pressure_rot_step,
                          this,
                          std_cxx1x::_1),
         Assembly::Scratch::
         pressure_rot_step<dim> (fe_pressure, 
                                 pressure_mapping,
                                 quadrature_formula,
                                 (update_values  |
                                  update_quadrature_points |
                                  update_JxW_values |
                                  update_gradients),
                                 fe_velocity,
                                 velocity_mapping,
                                 (update_values  |
                                  update_quadrature_points |
                                  update_gradients),
                                 concentr_fe,
                                 concentr_mapping,
                                 (update_values      |
                                  update_quadrature_points)),
         Assembly::CopyData::pressure_rot_step<dim> (fe_pressure));

    matrix_pressure.compress(VectorOperation::add);
    rhs_pressure.compress(VectorOperation::add);

    TrilinosWrappers::MPI::Vector distributed_sol (rhs_pressure);
    distributed_sol = pre_n_plus_1;
      
    SolverControl solver_control (matrix_pressure.m(), 
       parameters.eps_ns*rhs_pressure.l2_norm ());

    SolverCG<TrilinosWrappers::MPI::Vector> cg (solver_control);

    TrilinosWrappers::PreconditionAMG preconditioner;
    preconditioner.initialize (matrix_pressure);
    cg.solve (matrix_pressure, distributed_sol, rhs_pressure, preconditioner);
    constraints_pressure.distribute (distributed_sol);
    pre_n_plus_1 = distributed_sol;
    
    pcout << solver_control.last_step() << std::endl;
    
  }

  template <int dim>
  void UBC_mis_mixing<dim>::
  local_assemble_pressure_rot_step (const typename DoFHandler<dim>::active_cell_iterator &cell,
        Assembly::Scratch::pressure_rot_step <dim> &scratch,
        Assembly::CopyData::pressure_rot_step <dim> &data)
  { 
    const unsigned int dofs_per_cell = scratch.fe_pressure_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points = scratch.fe_pressure_values.n_quadrature_points;
    scratch.fe_pressure_values.reinit (cell);
    const FEValuesExtractors::Vector velocities (0);
    
    typename DoFHandler<dim>::active_cell_iterator
    velocity_cell (&triangulation,
                   cell->level(),
                   cell->index(),
                   &dof_handler_velocity);
    scratch.fe_velocity_values.reinit (velocity_cell);
   
    typename DoFHandler<dim>::active_cell_iterator
    concentr_cell (&triangulation,
                   cell->level(),
                   cell->index(),
                   &concentr_dof_handler);
    scratch.concentr_fe_values.reinit (concentr_cell);

    cell->get_dof_indices (data.local_dof_indices);
    
    data.local_matrix = 0;
    data.local_rhs = 0;
    
    scratch.fe_velocity_values[velocities].get_function_gradients (vel_n_plus_1, scratch.grad_vel_sol_values);
    scratch.fe_pressure_values.get_function_values (pre_n, scratch.pre_sol_values);
    scratch.fe_pressure_values.get_function_values (aux_n_plus_1, scratch.aux_sol_values);
    scratch.concentr_fe_values.get_function_values (concentr_solution, scratch.concentr_values);

    std::pair<double, double> discont_variables
                              = compute_discont_variable_on_cell (n_q_points, scratch.concentr_values);
   
    for (unsigned int q=0; q<n_q_points; ++q)
    {

      for (unsigned int k=0; k<dofs_per_cell; ++k)
       scratch.phi_p[k] = scratch.fe_pressure_values.shape_value (k,q);
      
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {      
         data.local_rhs(i) += scratch.phi_p[i]*
                              (
                                scratch.pre_sol_values[q]+
                                scratch.aux_sol_values[q]
                              )*
                              scratch.fe_pressure_values.JxW(q);
  
         double bb = 0.0;
         for (unsigned int d=0; d<dim; ++d)
           bb += scratch.grad_vel_sol_values[q][d][d];

          data.local_rhs(i) -= scratch.phi_p[i]*
                               bb*
                               (discont_variables.second)*
                               (1.0/parameters.Reynolds_number)*
                               scratch.fe_pressure_values.JxW(q);
     
         for (unsigned int j=0; j<dofs_per_cell; ++j)
           data.local_matrix(i,j) += scratch.phi_p[i]*
                                     scratch.phi_p[j]*
                                     scratch.fe_pressure_values.JxW(q);
      }  
    }
  }
  
  template <int dim>
  void
  UBC_mis_mixing<dim>::
  copy_local_to_global_pressure_rot_step 
      (const Assembly::CopyData::pressure_rot_step<dim> &data)
  {
    constraints_pressure.distribute_local_to_global (data.local_matrix,
                                                     data.local_rhs,
                                                     data.local_dof_indices,
                                                     matrix_pressure,
                                                     rhs_pressure);
  }
  
  //------------ For Update Solution In Time ---------------------------

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
