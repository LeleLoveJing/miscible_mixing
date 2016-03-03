#include <mismix/include.h>
#include <mismix/equation_data.h>
#include <mismix/assembly_copydata.h>
#include <mismix/class.h>
#include <mismix/parameter.h>

    template <int dim>
    void UBC_mis_mixing<dim>::initial_refine_mesh ()
    {
      pcout << "* Initial Refine Mesh.." << std::endl;
      for (unsigned int i=0; i<parameters.max_grid_level; ++i)
      {
        for (typename Triangulation<dim>::active_cell_iterator
        cell=triangulation.begin_active();
        cell!=triangulation.end(); ++cell)
        if (cell->is_locally_owned() || cell->is_ghost())
        {
          unsigned int dir = 0;
          if (dim==3) dir = 2;

          double dum = parameters.init_sep_x *
                       parameters.length_of_domain [parameters.flow_direction];

          if (std::abs(cell->center()[dir]-dum) < 1.0)
            cell->set_refine_flag ();
        }
        triangulation.execute_coarsening_and_refinement ();
      }
    }

    template <int dim>
    void UBC_mis_mixing<dim>::loop_over_cell_error_indicator ()
    {
      pcout << "* Compute Err Ind.." << std::endl;
     
      const QIterated<dim> quadrature (QTrapez<1>(), parameters.degree_of_concentr+1);
      const unsigned int n_q_points = quadrature.size();
      FEValues<dim> fe_values (concentr_fe, quadrature,
                               update_values | update_JxW_values);
      std::vector<double> concentr_values(n_q_points);

      typename DoFHandler<dim>::active_cell_iterator
        cell = concentr_dof_handler.begin_active(),
        endc = concentr_dof_handler.end();
      for (; cell!=endc; ++cell)
      if (cell->is_locally_owned() || cell->is_ghost())
      {
        cell->clear_refine_flag ();
        cell->clear_coarsen_flag ();
  
        fe_values.reinit (cell);
        fe_values.get_function_values (concentr_solution, concentr_values);
  
        double mm = 0.0; double kk = 0.0; double yy = 0.0;
        for (unsigned int q=1; q<n_q_points; ++q)
        {
          yy +=  concentr_values[q]*(1.0-concentr_values[q])/cell->diameter();
        }
        yy /= double (n_q_points);

        bool indi_on = false;
        if (yy > parameters.error_threshold/cell->diameter()) indi_on = true;

        if (indi_on == true) cell->set_refine_flag();
        else if  (indi_on == false && cell->level() > 0 ) cell->set_coarsen_flag ();

      }
    }

    template <int dim>
    void UBC_mis_mixing<dim>::loop_over_cell_error_indicator2 ()
    {
      pcout << "* Compute Err Ind2.." << std::endl;
     
      const QIterated<dim> quadrature (QTrapez<1>(), parameters.degree_of_concentr+1);
      const unsigned int n_q_points = quadrature.size();
      FEValues<dim> fe_values (concentr_fe, quadrature,
                               update_values | update_JxW_values);
      std::vector<double> concentr_values(n_q_points);

      typename DoFHandler<dim>::active_cell_iterator
        cell = concentr_dof_handler.begin_active(),
        endc = concentr_dof_handler.end();
      for (; cell!=endc; ++cell)
      if (cell->is_locally_owned() || cell->is_ghost())
      {
        fe_values.reinit (cell);
        fe_values.get_function_values (concentr_solution, concentr_values);
  
        double mm = 0.0; double kk = 0.0; double yy = 0.0;
        for (unsigned int q=1; q<n_q_points; ++q)
        {
          yy +=  concentr_values[q]*(1.0-concentr_values[q])/cell->diameter();
        }
        yy /= double (n_q_points);

        if (yy > parameters.error_threshold/cell->diameter())
        {
          cell->clear_coarsen_flag ();
          cell->set_refine_flag ();
        }
      }
    }
    
    template <int dim>
    void UBC_mis_mixing<dim>::compute_for_SymmTensorFlux ()
    {
      pcout << "* Compute for Symm and Flux.." << std::endl;
     
      post_error_crit1 = 0;
      post_error_crit2 = 0;
     
      const QIterated<dim> quadrature (QTrapez<1>(), parameters.degree_of_concentr+1);
      const unsigned int n_q_points = quadrature.size();
      FEValues<dim> fe_values (concentr_fe, quadrature,
                               update_values |
                               update_JxW_values);
      FEValues<dim> fe_velocity_values (fe_velocity, quadrature,
                                        update_values |
                                        update_gradients |
                                        update_JxW_values);
     
      std::vector<double> concentr_values(n_q_points);
      std::vector<Tensor<1, dim> > velocity_values (n_q_points);
      std::vector<Tensor<2, dim> > grad_velocity_values (n_q_points);
      std::vector<SymmetricTensor<2,dim> > symmTesor_values (n_q_points);
      std::vector<unsigned int>    local_dofs_indices (concentr_fe.dofs_per_cell);
     
      const FEValuesExtractors::Vector velocities (0);
     
      typename DoFHandler<dim>::active_cell_iterator
        cell = concentr_dof_handler.begin_active(),
        endc = concentr_dof_handler.end(),
        vel_cell = dof_handler_velocity.begin_active ();
     
      for (; cell!=endc; ++cell, ++vel_cell)
      if (cell->is_locally_owned() || cell->is_ghost())
      {
        fe_values.reinit (cell);
        cell->get_dof_indices (local_dofs_indices);
       
        fe_velocity_values.reinit (vel_cell);
        fe_values.get_function_values                                   (concentr_solution, concentr_values);
        fe_velocity_values[velocities].get_function_values              (vel_star, velocity_values);
        fe_velocity_values[velocities].get_function_gradients           (vel_star, grad_velocity_values);
        fe_velocity_values[velocities].get_function_symmetric_gradients (vel_star, symmTesor_values);
       
        for (unsigned int i=0; i<concentr_fe.dofs_per_cell; ++i)
        {
          double dum0 = 0.0;
          double dum1 = 0.0;
         
          for (unsigned int q=0; q<n_q_points; ++q)
          {
            double xx =  1.0 - std::abs(1.-2.*concentr_values[q]);
            double yy =  concentr_values[q]*(1.0-concentr_values[q])/cell->diameter();
           
            double theta = 1.0-2.*concentr_values[q];
           
            dum0 += fe_values.shape_value(i, q)*
                    yy*
                    fe_values.JxW(q);
           
            Tensor <1, dim> div_vel_values;
            
            for (unsigned int d=0; d<dim; ++d)
              div_vel_values [d] = grad_velocity_values[q][d][d];
            
            dum1 += fe_values.shape_value(i, q)*
                    div_vel_values.norm()*
                    fe_values.JxW(q);
          }
         
          post_error_crit1 [local_dofs_indices[i]] = dum0;
          post_error_crit2 [local_dofs_indices[i]] = dum1;
        }
      }
    }

    template <int dim>
    void UBC_mis_mixing<dim>::compute_for_post_error (unsigned int no_type, 
                                                      Vector<float> &estimated_error_per_cell)
    {
      pcout << "* Compute Error-Est.. " << no_type << std::endl;
     
      const QIterated<dim> quadrature (QTrapez<1>(), parameters.degree_of_concentr+1);
      const unsigned int n_q_points = quadrature.size();
      FEValues<dim> fe_values (concentr_fe, quadrature,
                               update_values |
                               update_JxW_values);
      FEValues<dim> fe_velocity_values (fe_velocity, quadrature,
                                        update_values |
                                        update_gradients |
                                        update_JxW_values);
     
      std::vector<double> concentr_values(n_q_points);
      std::vector<Tensor<1, dim> > velocity_values (n_q_points);
      std::vector<Tensor<1, dim> > velocity_star_values (n_q_points);
      std::vector<Tensor<2, dim> > grad_velocity_values (n_q_points);
      std::vector<SymmetricTensor<2,dim> > symmTesor_values (n_q_points);
     
      const FEValuesExtractors::Vector velocities (0);
     
      typename DoFHandler<dim>::active_cell_iterator
        cell = concentr_dof_handler.begin_active(),
        endc = concentr_dof_handler.end(),
        vel_cell = dof_handler_velocity.begin_active ();
     
      unsigned int no_cell = 0;
      
      for (; cell!=endc; ++cell, ++vel_cell, ++no_cell)
      if (cell->is_locally_owned() || cell->is_ghost())
      {
        fe_values.reinit (cell);
        fe_velocity_values.reinit (vel_cell);
        
        fe_values.get_function_values                                   (concentr_solution, concentr_values);
        fe_velocity_values[velocities].get_function_values              (vel_star, velocity_star_values);
        fe_velocity_values[velocities].get_function_values              (vel_n_plus_1, velocity_values);
        fe_velocity_values[velocities].get_function_gradients           (vel_star, grad_velocity_values);
        fe_velocity_values[velocities].get_function_symmetric_gradients (vel_star, symmTesor_values);

        double dum = 0.0;
                  
        for (unsigned int q=0; q<n_q_points; ++q)
        {
          if (no_type == 2)
          {
            Tensor<1, dim> div_vel_values;
            for (unsigned int d=0; d<dim; ++d)
              div_vel_values [d] = grad_velocity_values[q][d][d];
            
            dum += div_vel_values.norm();
            
          }
          else if (no_type == 3)
          {
            Tensor<1, dim> vel_diff;
            vel_diff = velocity_star_values[q] - velocity_values[q];
          
            dum += vel_diff.norm();
          }
          
        }
        
        estimated_error_per_cell [no_cell] = dum;
      }
    }
    
    template <int dim>
    void UBC_mis_mixing<dim>::refine_mesh (const unsigned int max_grid_level)
    {
        pcout << "* Refine Mesh.. " << parameters.type_adaptivity_rule << std::endl;
     
        if (parameters.type_adaptivity_rule == 0)
        {
          compute_for_SymmTensorFlux ();
         
          Vector<float> estimated_error_per_cell  (triangulation.n_active_cells());
          Vector<float> estimated_error_per_cell0 (triangulation.n_active_cells());
          Vector<float> estimated_error_per_cell1 (triangulation.n_active_cells());
          Vector<float> estimated_error_per_cell2 (triangulation.n_active_cells());

          KellyErrorEstimator<dim>::estimate (concentr_dof_handler,
                                              QGauss<dim-1>(parameters.degree_of_concentr+1),
                                              typename FunctionMap<dim>::type(),
                                              post_error_crit1,
                                              estimated_error_per_cell0,
                                              ComponentMask(),
                                              0,
                                              0,
                                              triangulation.locally_owned_subdomain());
         
          KellyErrorEstimator<dim>::estimate (concentr_dof_handler,
                                              QGauss<dim-1>(parameters.degree_of_concentr+1),
                                              typename FunctionMap<dim>::type(),
                                              post_error_crit2,
                                              estimated_error_per_cell1,
                                              ComponentMask(),
                                              0,
                                              0,
                                              triangulation.locally_owned_subdomain());
         
          KellyErrorEstimator<dim>::estimate (concentr_dof_handler,
                                              QGauss<dim-1>(parameters.degree_of_concentr+1),
                                              typename FunctionMap<dim>::type(),
                                              concentr_solution,
                                              estimated_error_per_cell2,
                                              ComponentMask(),
                                              0,
                                              0,
                                              triangulation.locally_owned_subdomain());
         
          for (unsigned int i=0; i<triangulation.n_active_cells(); ++i)
            estimated_error_per_cell (i) = estimated_error_per_cell1 (i) *
                                           estimated_error_per_cell2 (i);
         
          parallel::distributed::GridRefinement::
          refine_and_coarsen_fixed_fraction (triangulation,
                                             estimated_error_per_cell,
                                             parameters.ref_crit,
                                             parameters.coar_crit);
         
        }
        else if (parameters.type_adaptivity_rule == 1)
        {
          compute_for_SymmTensorFlux ();
         
          Vector<float> estimated_error_per_cell  (triangulation.n_active_cells());
          Vector<float> estimated_error_per_cell0 (triangulation.n_active_cells());
          Vector<float> estimated_error_per_cell1 (triangulation.n_active_cells());
          Vector<float> estimated_error_per_cell2 (triangulation.n_active_cells());

          KellyErrorEstimator<dim>::estimate (concentr_dof_handler,
                                              QGauss<dim-1>(parameters.degree_of_concentr+1),
                                              typename FunctionMap<dim>::type(),
                                              post_error_crit1,
                                              estimated_error_per_cell0,
                                              ComponentMask(),
                                              0,
                                              0,
                                              triangulation.locally_owned_subdomain());
         
          KellyErrorEstimator<dim>::estimate (concentr_dof_handler,
                                              QGauss<dim-1>(parameters.degree_of_concentr+1),
                                              typename FunctionMap<dim>::type(),
                                              post_error_crit2,
                                              estimated_error_per_cell1,
                                              ComponentMask(),
                                              0,
                                              0,
                                              triangulation.locally_owned_subdomain());
          for (unsigned int i=0; i<triangulation.n_active_cells(); ++i)
            estimated_error_per_cell (i) = estimated_error_per_cell0 (i) *
                                           estimated_error_per_cell1 (i);
         
          parallel::distributed::GridRefinement::
          refine_and_coarsen_fixed_fraction (triangulation,
                                             estimated_error_per_cell,
                                             parameters.ref_crit,
                                             parameters.coar_crit);
        }
        else if (parameters.type_adaptivity_rule == 2)
        {
          compute_for_SymmTensorFlux ();
         
          Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
          compute_for_post_error (2, estimated_error_per_cell);
         
          parallel::distributed::GridRefinement::
          refine_and_coarsen_fixed_fraction (triangulation,
                                             estimated_error_per_cell,
                                             parameters.ref_crit,
                                             parameters.coar_crit);
        }
        else if (parameters.type_adaptivity_rule == 3)
        {
          Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
          compute_for_post_error (3, estimated_error_per_cell);
          parallel::distributed::GridRefinement::
          refine_and_coarsen_fixed_fraction (triangulation,
                                             estimated_error_per_cell,
                                             parameters.ref_crit,
                                             parameters.coar_crit);
        }
        else if (parameters.type_adaptivity_rule == 4)
        {
          compute_for_SymmTensorFlux ();
         
          Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
          compute_for_post_error (2, estimated_error_per_cell);
         
          parallel::distributed::GridRefinement::
          refine_and_coarsen_fixed_fraction (triangulation,
                                             estimated_error_per_cell,
                                             parameters.ref_crit,
                                             parameters.coar_crit);

          loop_over_cell_error_indicator2 ();
        }
        else if (parameters.type_adaptivity_rule == 5)
        {
          Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
          compute_for_post_error (3, estimated_error_per_cell);
          parallel::distributed::GridRefinement::
          refine_and_coarsen_fixed_fraction (triangulation,
                                             estimated_error_per_cell,
                                             parameters.ref_crit,
                                             parameters.coar_crit);
          
          loop_over_cell_error_indicator2 ();
        }
        else if (parameters.type_adaptivity_rule == 6)
        {
          loop_over_cell_error_indicator ();
        }
        else if (parameters.type_adaptivity_rule == 7)
        {
          Vector<float> estimated_error_per_cell  (triangulation.n_active_cells());

          KellyErrorEstimator<dim>::estimate (dof_handler_pressure,
                                              QGauss<dim-1>(parameters.degree_of_pressure+1),
                                              typename FunctionMap<dim>::type(),
                                              pre_n_plus_1,
                                              estimated_error_per_cell,
                                              ComponentMask(),
                                              0,
                                              0,
                                              triangulation.locally_owned_subdomain());

          parallel::distributed::GridRefinement::
          refine_and_coarsen_fixed_fraction (triangulation,
                                             estimated_error_per_cell,
                                             parameters.ref_crit,
                                             parameters.coar_crit);
        }
        else if (parameters.type_adaptivity_rule == 8)
        {
          Vector<float> estimated_error_per_cell  (triangulation.n_active_cells());

          KellyErrorEstimator<dim>::estimate (dof_handler_velocity,
                                              QGauss<dim-1>(parameters.degree_of_velocity+1),
                                              typename FunctionMap<dim>::type(),
                                              vel_star,
                                              estimated_error_per_cell,
                                              ComponentMask(),
                                              0,
                                              0,
                                              triangulation.locally_owned_subdomain());

          parallel::distributed::GridRefinement::
          refine_and_coarsen_fixed_fraction (triangulation,
                                             estimated_error_per_cell,
                                             parameters.ref_crit,
                                             parameters.coar_crit);
        }
        
        if (triangulation.n_levels() > parameters.max_grid_level)
          for (typename Triangulation<dim>::active_cell_iterator
            cell = triangulation.begin_active(parameters.max_grid_level);
            cell != triangulation.end(); ++cell)
              cell->clear_refine_flag ();
     
        {
          std::vector<const TrilinosWrappers::MPI::Vector *> vel_system (3);
            vel_system[0] = &vel_n_plus_1;
            vel_system[1] = &vel_n;
            vel_system[2] = &vel_n_minus_1;
          parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector>
            vel_trans(dof_handler_velocity);
    
          std::vector<const TrilinosWrappers::MPI::Vector *> pre_system (4);
            pre_system[0] = &pre_star;
            pre_system[1] = &pre_n_plus_1;
            pre_system[2] = &pre_n;
            pre_system[3] = &pre_n_minus_1;
          parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector>
            pre_trans(dof_handler_pressure);
            
          std::vector<const TrilinosWrappers::MPI::Vector *> aux_system (3);
            aux_system[0] = &aux_n_plus_1;
            aux_system[1] = &aux_n;
            aux_system[2] = &aux_n_minus_1;
          parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector>
            aux_trans(dof_handler_auxilary);

          std::vector<const TrilinosWrappers::MPI::Vector *> con_system (3);
            con_system[0] = &concentr_solution;
            con_system[1] = &old_concentr_solution;
            con_system[2] = &old_old_concentr_solution;
          parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector>
            con_trans(concentr_dof_handler);
            
          triangulation.prepare_coarsening_and_refinement();
          
          vel_trans.prepare_for_coarsening_and_refinement(vel_system);
          pre_trans.prepare_for_coarsening_and_refinement(pre_system);
          aux_trans.prepare_for_coarsening_and_refinement(aux_system);
          con_trans.prepare_for_coarsening_and_refinement(con_system);
          
          triangulation.execute_coarsening_and_refinement ();

          {
            setup_dofs_error ();
          }
          

          {
            setup_dofs_velocity ();
            TrilinosWrappers::MPI::Vector d0_system (rhs_velocity),
                                          d1_system (rhs_velocity),
                                          d2_system (rhs_velocity);

            std::vector<TrilinosWrappers::MPI::Vector *> temp_sol (3);
              temp_sol[0] = & (d0_system);
              temp_sol[1] = & (d1_system);
              temp_sol[2] = & (d2_system);

            vel_trans.interpolate (temp_sol);
          
            vel_n_plus_1  = d0_system;
            vel_n         = d1_system;
            vel_n_minus_1 = d2_system;
          }
          
          {
            setup_dofs_pressure ();
            TrilinosWrappers::MPI::Vector d0_system (rhs_pressure),
                                          d1_system (rhs_pressure),
                                          d2_system (rhs_pressure),
                                          d3_system (rhs_pressure);
  
            std::vector<TrilinosWrappers::MPI::Vector *> temp_sol (4);
              temp_sol[0] = & (d0_system);
              temp_sol[1] = & (d1_system);
              temp_sol[2] = & (d2_system);
              temp_sol[3] = & (d3_system);
    
            pre_trans.interpolate (temp_sol);
      
            pre_star       = d0_system;
            pre_n_plus_1   = d1_system;
            pre_n          = d2_system;
            pre_n_minus_1  = d3_system;
          
          }

          {
            setup_dofs_auxilary ();
            TrilinosWrappers::MPI::Vector d0_system (rhs_auxilary),
                                          d1_system (rhs_auxilary),
                                          d2_system (rhs_auxilary);
  
            std::vector<TrilinosWrappers::MPI::Vector *> temp_sol (3);
              temp_sol[0] = & (d0_system);
              temp_sol[1] = & (d1_system);
              temp_sol[2] = & (d2_system);
    
            aux_trans.interpolate (temp_sol);
          
            aux_n_plus_1  = d0_system;
            aux_n         = d1_system;
            aux_n_minus_1 = d2_system;
          }
          
          {
            setup_dofs_concentr ();
             TrilinosWrappers::MPI::Vector d0_system (concentr_rhs),
                                           d1_system (concentr_rhs),
                                           d2_system (concentr_rhs);

             std::vector<TrilinosWrappers::MPI::Vector *> temp_sol (3);
               temp_sol[0] = & (d0_system);
               temp_sol[1] = & (d1_system);
               temp_sol[2] = & (d2_system);

             con_trans.interpolate (temp_sol);

             concentr_solution         = d0_system;
             old_concentr_solution     = d1_system;
             old_old_concentr_solution = d2_system;
          }
        }
    }

// Explicit instantiations
// template class UBC_mis_mixing<1>;
template class UBC_mis_mixing<2>;
template class UBC_mis_mixing<3>;
