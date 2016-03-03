#include <mismix/include.h>
#include <mismix/equation_data.h>
#include <mismix/assembly_copydata.h>
#include <mismix/class.h>
#include <mismix/parameter.h>

  template <int dim>
  void UBC_mis_mixing<dim>::compute_for_avrage_quantities_surfIntgl (std::ofstream &out_avr_c,
                                                                     std::ofstream &out_avr_vFlow_c,
                                                                     std::ofstream &out_avr_vvFlow_c,
                                                                     std::ofstream &out_avr_vvLati_c,
                                                                     std::ofstream &out_avr_vvDept_c)
  {
    pcout << "* Compute the Average Quantities by Surface Integral.." << std::endl;
   
    unsigned int number_slices = parameters.number_slices_coarse_mesh + 1;
    double interval_flow_dir_coor = parameters.length_of_domain [parameters.flow_direction]/
                                    double (number_slices-1);
   
    std::vector<double> integral_area_over_face (number_slices),
                        intgral_of_concentr_over_face (number_slices),
                        intgral_of_AxialVC_over_face (number_slices),
                        intgral_of_AxialVVC_over_face (number_slices),
                        intgral_of_LatitudeVVC_over_face (number_slices),
                        intgral_of_DepthVVC_over_face (number_slices),
                        coor_pick_up (number_slices);
   
    for (unsigned int i=0; i<number_slices; ++i)
     coor_pick_up[i] = double(i)*interval_flow_dir_coor;
   
    const QIterated<dim-1> face_quadrature_formula (QTrapez<1>(), parameters.degree_of_concentr+1);
    const unsigned int n_q_points_face = face_quadrature_formula.size();
    FEFaceValues<dim> fe_face_values_concentr (concentr_fe,
                                               face_quadrature_formula,
                                               update_values   |
                             update_quadrature_points |
                             update_JxW_values);

    FEFaceValues<dim> fe_face_values_velocity (fe_velocity,
                                               face_quadrature_formula,
                                               update_values);
   
    std::vector<double> concentr_face_values        (n_q_points_face);
    std::vector<Tensor<1, dim> > velocity_face_values (n_q_points_face);
   
    typename DoFHandler<dim>::active_cell_iterator
      cell = concentr_dof_handler.begin_active(),
      endc = concentr_dof_handler.end();
    typename DoFHandler<dim>::active_cell_iterator
      vel_cell = dof_handler_velocity.begin_active();
   
    const FEValuesExtractors::Vector velocities (0);
   
    for (; cell!=endc; ++cell, ++vel_cell)
    if (cell->is_locally_owned())
    {
      for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no)
      {
        fe_face_values_concentr.reinit (cell, face_no);
        fe_face_values_velocity.reinit (vel_cell, face_no);
       
        typename DoFHandler<dim>::face_iterator face_concentr = cell->face(face_no);
        typename DoFHandler<dim>::face_iterator face_velocity = vel_cell->face(face_no);
       
        bool is_integ_face = false;
        unsigned int which_append_zaxis = std::numeric_limits<double>::max();
        
        for (unsigned int i=0; i<number_slices; ++i)
        if (is_integ_face == false)
        if (std::abs(face_concentr->center()[parameters.flow_direction] - coor_pick_up[i]) < 1e-5 )
        {
          is_integ_face = true;
          which_append_zaxis = i;
         
          if (is_integ_face && which_append_zaxis < number_slices)
          {
          
            fe_face_values_concentr.get_function_values (concentr_solution, concentr_face_values);
            fe_face_values_velocity[velocities].get_function_values (vel_star, velocity_face_values);
           
            for (unsigned int q=0; q<n_q_points_face; ++q)
            {
                       
              double quanty_c = 1.0 - concentr_face_values [q];
            
              integral_area_over_face [i]       += fe_face_values_concentr.JxW(q);
              intgral_of_concentr_over_face[i]  += quanty_c*fe_face_values_concentr.JxW(q);
              intgral_of_AxialVC_over_face[i]   += quanty_c*
                                                   velocity_face_values[q][parameters.flow_direction]*
                                                   fe_face_values_concentr.JxW(q);
              intgral_of_AxialVVC_over_face[i]  += quanty_c*
                                                   velocity_face_values[q][parameters.flow_direction]*
                                                   velocity_face_values[q][parameters.flow_direction]*
                                                   fe_face_values_concentr.JxW(q);
              intgral_of_LatitudeVVC_over_face[i]  += quanty_c*
                                                      velocity_face_values[q][parameters.latitude_direction]*
                                                      velocity_face_values[q][parameters.latitude_direction]*
                                                      fe_face_values_concentr.JxW(q);
              if (dim == 3)
              intgral_of_DepthVVC_over_face[i]  += quanty_c*
                                                   velocity_face_values[q][parameters.depth_direction]*
                                                   velocity_face_values[q][parameters.depth_direction]*
                                                   fe_face_values_concentr.JxW(q);
            }
          }
        }
      }
    }
   
    std::vector<double> global_integral_area_over_face (number_slices),
                        global_intgral_of_concentr_over_face (number_slices),
                        global_intgral_of_AxialVC_over_face (number_slices),
                        global_intgral_of_AxialVVC_over_face (number_slices),
                        global_intgral_of_LatitudeVVC_over_face (number_slices),
                        global_intgral_of_DepthVVC_over_face (number_slices);
   
    Utilities::MPI::sum (integral_area_over_face,       MPI_COMM_WORLD, global_integral_area_over_face);
    Utilities::MPI::sum (intgral_of_concentr_over_face, MPI_COMM_WORLD, global_intgral_of_concentr_over_face);
    Utilities::MPI::sum (intgral_of_AxialVC_over_face, MPI_COMM_WORLD, global_intgral_of_AxialVC_over_face);
    Utilities::MPI::sum (intgral_of_AxialVVC_over_face, MPI_COMM_WORLD, global_intgral_of_AxialVVC_over_face);
    Utilities::MPI::sum (intgral_of_LatitudeVVC_over_face, MPI_COMM_WORLD, global_intgral_of_LatitudeVVC_over_face);
    Utilities::MPI::sum (intgral_of_DepthVVC_over_face, MPI_COMM_WORLD, global_intgral_of_DepthVVC_over_face);
   
    out_avr_c << timestep_number;
    out_avr_vFlow_c << timestep_number;
    out_avr_vvFlow_c << timestep_number;
    out_avr_vvLati_c << timestep_number;
    out_avr_vvDept_c << timestep_number;
   
    double min_length_mixing = 0.0;
    double max_length_mixing = 1.0;

    for (unsigned int i=0; i<number_slices; ++i)
    {
      out_avr_c << " " << global_intgral_of_concentr_over_face[i]/global_integral_area_over_face[i];
      out_avr_vFlow_c << " " << global_intgral_of_AxialVC_over_face[i]/global_integral_area_over_face[i];
      out_avr_vvFlow_c << " " << global_intgral_of_AxialVVC_over_face[i]/global_integral_area_over_face[i];
      out_avr_vvLati_c << " " << global_intgral_of_LatitudeVVC_over_face[i]/global_integral_area_over_face[i];
      out_avr_vvDept_c << " " << global_intgral_of_DepthVVC_over_face[i]/global_integral_area_over_face[i];

      double val = global_intgral_of_concentr_over_face[i]/global_integral_area_over_face[i];

      if (std::abs(val) < 0.01000001 && min_length_mixing < val)
      {
        mixing_min_max.second = i;
        min_length_mixing = val;
      }

      if (std::abs(val) > 0.98999999 && max_length_mixing > val)
      {
        mixing_min_max.first = i;
        max_length_mixing = val;
      }

    }
   
    out_avr_c << std::endl;
    out_avr_vFlow_c << std::endl;
    out_avr_vvFlow_c << std::endl;
    out_avr_vvLati_c << std::endl;
    out_avr_vvDept_c << std::endl;
  }

  template <int dim>
  void UBC_mis_mixing<dim>::compute_global_error_norm (std::ofstream &out_errors)
  {
    pcout << "* Compute Global Error Norm.." << std::endl;

    unsigned int n_e = dof_handler_error.n_dofs();
    IndexSet partitioning (n_e), relevant_partitioning (n_e);
    partitioning = dof_handler_error.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs (dof_handler_error,
                                             relevant_partitioning);

    out_errors  << timestep_number << " "
                << triangulation.n_global_active_cells () << " ";

    TrilinosWrappers::MPI::Vector dist_vel_diff_norm (rhs_velocity),
                                  dist_pre_diff_norm (rhs_pressure),
                                  dist_vel (rhs_velocity),
                                  dist_vel_old (rhs_velocity),
                                  dist_pre_n_plus_1 (rhs_pressure),
                                  dist_pre_n (rhs_pressure),
                                  dist_concentr (concentr_rhs);

    TrilinosWrappers::MPI::Vector dist_grad_vel_diff_norm (partitioning, MPI_COMM_WORLD),
                                  dist_grad_pre_diff_norm (partitioning, MPI_COMM_WORLD),
                                  dist_grad_vel_curr_norm (partitioning, MPI_COMM_WORLD),
                                  dist_grad_pre_curr_norm (partitioning, MPI_COMM_WORLD),
                                  div_vel_norm            (partitioning, MPI_COMM_WORLD),
                                  dist_grad_concentr_flow_norm (partitioning, MPI_COMM_WORLD),
                                  dist_grad_concentr_all_norm  (partitioning, MPI_COMM_WORLD);
    dist_vel = vel_n;
    dist_vel_old = vel_n_minus_1;
    dist_pre_n_plus_1 = pre_n;
    dist_pre_n = pre_n_minus_1;

    dist_vel_diff_norm.sadd (0.0, +1.0, dist_vel);
    dist_vel_diff_norm.sadd (1.0, -1.0, dist_vel_old);
    dist_pre_diff_norm.sadd (0.0, +1.0, dist_pre_n_plus_1);
    dist_pre_diff_norm.sadd (1.0, -1.0, dist_pre_n);

    dist_concentr = concentr_solution;

    const QIterated<dim> quadrature (QTrapez<1>(), parameters.degree_of_concentr+1);
    const unsigned int n_q_points = quadrature.size();

    FEValues<dim> fe_values (fe_error, quadrature,
                             update_values |
                             update_gradients |
                             update_JxW_values);

    FEValues<dim> fe_velocity_values (fe_velocity, quadrature,
                                      update_values |
                                      update_gradients);

    FEValues<dim> fe_pressure_values (fe_pressure, quadrature,
                                      update_values |
                                      update_gradients);

    FEValues<dim> fe_concentr_values (concentr_fe, quadrature,
                                      update_values |
                                      update_gradients);

    std::vector<Tensor<2, dim> > grad_velocity_values (n_q_points);
    std::vector<Tensor<2, dim> > grad_old_velocity_values (n_q_points);
    std::vector<Tensor<1, dim> > grad_pre_values (n_q_points);
    std::vector<Tensor<1, dim> > grad_pre_old_values (n_q_points);
    std::vector<Tensor<1, dim> > grad_concentr_values (n_q_points);
    std::vector<Tensor<1, dim> > grad_old_concentr_values (n_q_points);

    std::vector<unsigned int>    local_dofs_indices (fe_error.dofs_per_cell);

    const FEValuesExtractors::Vector velocities (0);

    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_error.begin_active(),
      endc = dof_handler_error.end(),
      vel_cell = dof_handler_velocity.begin_active (),
      pre_cell = dof_handler_pressure.begin_active (),
      con_cell = concentr_dof_handler.begin_active();

    for (; cell!=endc; ++cell, ++vel_cell, ++pre_cell, ++con_cell)
    if (cell->is_locally_owned())
    {
      fe_values.reinit (cell);
      fe_velocity_values.reinit (vel_cell);
      fe_pressure_values.reinit (pre_cell);
      fe_concentr_values.reinit (con_cell);

      cell->get_dof_indices (local_dofs_indices);

      fe_velocity_values[velocities].get_function_gradients (vel_n, grad_velocity_values);
      fe_velocity_values[velocities].get_function_gradients (vel_n_minus_1, grad_old_velocity_values);

      fe_pressure_values.get_function_gradients (pre_n, grad_pre_values);
      fe_pressure_values.get_function_gradients (pre_n_minus_1, grad_pre_old_values);

      fe_concentr_values.get_function_gradients (concentr_solution, grad_concentr_values);
      fe_concentr_values.get_function_gradients (old_concentr_solution, grad_old_concentr_values);

      for (unsigned int i=0; i<fe_error.dofs_per_cell; ++i)
      {
        double dum0 = 0.0;
        double dum1 = 0.0;
        double dum2 = 0.0;
        double dum3 = 0.0;
        double dum4 = 0.0;
        double dum5 = 0.0;
        double dum6 = 0.0;

        for (unsigned int q=0; q<n_q_points; ++q)
        {
          for (unsigned int d=0; d<dim; ++d)
            dum0 += fe_values.shape_value (i, q)*
                    grad_velocity_values[q][d][d]*
                    fe_values.JxW(q);

          {
            Tensor<1, dim> nn;
            for (unsigned int d=0; d<dim; ++d)
              nn[d] = grad_pre_values[q][d] - grad_pre_old_values[q][d];

            dum1 += fe_values.shape_value (i, q)*
                    nn.norm()*
                    fe_values.JxW(q);
          }

          {
            Tensor<2, dim> nn;
            for (unsigned int d1=0; d1<dim; ++d1)
              for (unsigned int d2=0; d2<dim; ++d2)
                nn[d1][d2] = grad_velocity_values[q][d1][d2] - grad_old_velocity_values[q][d1][d2];

            dum2 += fe_values.shape_value (i, q)*
                    nn.norm()*
                    fe_values.JxW(q);
          }

          {
            Tensor<1, dim> nn;
              nn[0] = grad_concentr_values[q][parameters.flow_direction];

            dum3 += fe_values.shape_value (i, q)*
                    nn.norm()*
                    fe_values.JxW(q);
          }

          {
            Tensor<1, dim> nn;
            for (unsigned int d=0; d<dim; ++d)
              nn[d] = grad_concentr_values[q][d];

            dum4 += fe_values.shape_value (i, q)*
                    nn.norm()*
                    fe_values.JxW(q);
          }

          {
            Tensor<2, dim> nn = grad_velocity_values[q];
            dum5 += fe_values.shape_value (i, q)*
                    nn.norm()*
                    fe_values.JxW(q);
          }

          {
            Tensor<1, dim> nn = grad_pre_values[q];
            dum6 += fe_values.shape_value (i, q)*
                    nn.norm()*
                    fe_values.JxW(q);
          }
        }

        div_vel_norm                 [local_dofs_indices[i]] = dum0;
        dist_grad_pre_diff_norm      [local_dofs_indices[i]] = dum1;
        dist_grad_vel_diff_norm      [local_dofs_indices[i]] = dum2;
        dist_grad_concentr_flow_norm [local_dofs_indices[i]] = dum3;
        dist_grad_concentr_all_norm  [local_dofs_indices[i]] = dum4;
        dist_grad_vel_curr_norm      [local_dofs_indices[i]] = dum5;
        dist_grad_pre_curr_norm      [local_dofs_indices[i]] = dum6;

      }
    }

    out_errors << dist_vel_diff_norm.l2_norm ()          << " "
               << dist_grad_vel_diff_norm.l2_norm ()     << " "
               << dist_pre_diff_norm.l2_norm ()          << " "
               << dist_grad_pre_diff_norm.l2_norm ()     << " "
               << div_vel_norm.l2_norm ()                << " "
               << dist_vel.l2_norm()                     << " "
               << dist_grad_vel_curr_norm.l2_norm()      << " "
               << dist_pre_n_plus_1.l2_norm()            << " "
               << dist_grad_pre_curr_norm.l2_norm ()     << " "
               << dist_grad_concentr_flow_norm.l2_norm() << " "
               << dist_grad_concentr_all_norm.l2_norm()  << " "
               << entropy_viscosity_for_ns.l2_norm ()    << " "
               << energy_norm_for_ns.l2_norm ()          << " "
               << mixing_min_max.first                   << " "
               << mixing_min_max.second                  << std::endl;
  }

// Explicit instantiations
// template class UBC_mis_mixing<1>;
template class UBC_mis_mixing<2>;
template class UBC_mis_mixing<3>;
