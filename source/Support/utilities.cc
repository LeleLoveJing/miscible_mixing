#include <mismix/include.h>
#include <mismix/equation_data.h>
#include <mismix/assembly_copydata.h>
#include <mismix/class.h>
#include <mismix/parameter.h>

  //----------------------------------------------------------------------


    template <int dim>
    void UBC_mis_mixing<dim>::print_input_parameters ()
    {
      double inertial_vel_scale = std::sqrt(std::abs(parameters.Atwood_number)*
                      EquationData::gravitiy_accelation*EquationData::pipe_diameter);
      double viscous_vel_scale = parameters.Atwood_number*EquationData::gravitiy_accelation*
                    EquationData::pipe_diameter*EquationData::pipe_diameter
                    /EquationData::kinematic_viscosity;
                    
      pcout << "################# Simulation Start #################" << std::endl;
      pcout << "- Reynolds Number = " << parameters.Reynolds_number << ", " << parameters.reference_velocity <<  std::endl;
      pcout << "- Froude Number = " << parameters.Froude_number << std::endl;
      pcout << "- Atwood Number = " << parameters.Atwood_number << std::endl;
      pcout << "- Shear Thinning Fluid K  = " << parameters.ratio_pow_law << std::endl;
      pcout << "- Shear Thinning Fluid N  = " << parameters.n_pow_law << std::endl;
      pcout << "- CFL Number = " << parameters.CFL_number << std::endl;
      pcout << "- Time Interval = " << computed_time_step << std::endl;
      if (parameters.is_density_stable_flow)
        pcout << "- Density Stable Flow.." << std::endl;
      if (parameters.is_density_stable_flow == false)
        pcout << "- Density Un-Stable Flow.." << std::endl;
      pcout << "- Inclined Angle Vector = " << parameters.inclined_angle_vector << std::endl;
      pcout << "- Optimization Method = " << parameters.ist_optimization_method << std::endl;
      pcout << "- Vv_Viscous Velocity Scale = " << viscous_vel_scale << std::endl;
      pcout << "- Vt_Inertial Velocity Scale = " << inertial_vel_scale << std::endl;
      pcout << "- Re_t : Char. Reynolds_number = " << viscous_vel_scale/inertial_vel_scale << std::endl;
      pcout << "- Cos (Beta) = " << std::cos(parameters.inclined_angle*numbers::PI/180) << std::endl;
      pcout << "- Vv X cosB = " << viscous_vel_scale*std::cos(parameters.inclined_angle*numbers::PI/180.0) << std::endl;
      pcout << "- Re_t X cosB = " << (viscous_vel_scale/inertial_vel_scale)*std::cos(parameters.inclined_angle*numbers::PI/180) << std::endl;
      pcout << "- Kay (cotB/theta = 2VvcosB/V0) = " << 2*viscous_vel_scale*std::cos(parameters.inclined_angle*numbers::PI/180)
                            /parameters.reference_velocity << std::endl; 
      pcout << "- PolyNormial Order = "
        << parameters.degree_of_velocity << " | "
        << parameters.degree_of_pressure << " | "
        << parameters.degree_of_concentr << std::endl;
      pcout << "- Init. Gate Valve = " << parameters.init_sep_x << ", " 
            << parameters.init_sep_x*parameters.reference_length
            << std::endl;
                
      pcout << "- Ref. Time = " << parameters.reference_time << std::endl;
      pcout << "- Ref. Length = " << parameters.reference_length << std::endl;
      pcout << "- Ref. Velocity = " << parameters.reference_velocity << std::endl;
      
      std::ostringstream filename_get_global_values;
      filename_get_global_values << "output/data/" + Utilities::int_to_string(parameters.data_id, 2) +  "_system.dat";
      std::ofstream out_get_global_values (filename_get_global_values.str().c_str());

      out_get_global_values << "################# Simulation Start #################" << std::endl;
      out_get_global_values << "- Reynolds Number = " << parameters.Reynolds_number << ", " << parameters.reference_velocity <<  std::endl;
      out_get_global_values << "- Froude Number = " << parameters.Froude_number << std::endl;
      out_get_global_values << "- Atwood Number = " << parameters.Atwood_number << std::endl;
      out_get_global_values << "- Shear Thinning Fluid K  = " << parameters.ratio_pow_law << std::endl;
      out_get_global_values << "- Shear Thinning Fluid N  = " << parameters.n_pow_law << std::endl;
      out_get_global_values << "- Time Interval = " << parameters.computed_time_step << std::endl;
      if (parameters.is_density_stable_flow)
        out_get_global_values << "- Density Stable Flow.." << std::endl;
      if (parameters.is_density_stable_flow == false)
        out_get_global_values << "- Density Un-Stable Flow.." << std::endl;
      out_get_global_values << "- Optimization Method = " << parameters.ist_optimization_method << std::endl;
      out_get_global_values << "- Vv_Viscous Velocity Scale = " << viscous_vel_scale << std::endl;
      out_get_global_values << "- Vt_Inertial Velocity Scale = " << inertial_vel_scale << std::endl;
      out_get_global_values << "- Re_t : Char. Reynolds_number = " << viscous_vel_scale/inertial_vel_scale << std::endl;
      out_get_global_values << "- Cos (Beta) = " << std::cos(parameters.inclined_angle*numbers::PI/180) << std::endl;
      out_get_global_values << "- Vv X cosB = " << viscous_vel_scale*std::cos(parameters.inclined_angle*numbers::PI/180.0) << std::endl;
      out_get_global_values << "- Re_t X cosB = " << (viscous_vel_scale/inertial_vel_scale)*std::cos(parameters.inclined_angle*numbers::PI/180) << std::endl;
      out_get_global_values << "- Kay (cotB/theta = 2VvcosB/V0) = " << 2*viscous_vel_scale*std::cos(parameters.inclined_angle*numbers::PI/180)
                                      /parameters.reference_velocity << std::endl; 
      out_get_global_values << "- PolyNormial Order = "
                            << parameters.degree_of_velocity << " | "
                            << parameters.degree_of_pressure << " | "
                            << parameters.degree_of_concentr << std::endl;
      out_get_global_values << "- Init. Gate Valve = " << parameters.init_sep_x << ", " 
                            << parameters.init_sep_x*parameters.reference_length
                            << std::endl;
    }

  template <int dim>
  double 
  UBC_mis_mixing<dim>::get_cfl_number () const
  {
    const QIterated<dim> quadrature_formula (QTrapez<1>(),
                                             parameters.degree_of_velocity);
    const unsigned int n_q_points = quadrature_formula.size();

    MappingQ<dim> mapping(parameters.degree_of_velocity);
    FEValues<dim> fe_values (mapping, fe_velocity, quadrature_formula, update_values);
    std::vector<Tensor<1,dim> > velocity_values(n_q_points);

    const FEValuesExtractors::Vector velocities (0);

    double max_local_cfl = 0;

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_velocity.begin_active(),
    endc = dof_handler_velocity.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit (cell);
          fe_values[velocities].get_function_values (vel_star,
                                                     velocity_values);

          double max_local_velocity = 1e-10;
          for (unsigned int q=0; q<n_q_points; ++q)
            max_local_velocity = std::max (max_local_velocity,
                                           velocity_values[q].norm());
          max_local_cfl = std::max(max_local_cfl,
                                   max_local_velocity / cell->diameter());
        }

    return Utilities::MPI::max (max_local_cfl, MPI_COMM_WORLD);
  }
  
  template <int dim>
  double 
  UBC_mis_mixing<dim>::get_maximal_velocity () const
  {
    const QIterated<dim> quadrature_formula (QTrapez<1>(),
                                             parameters.degree_of_velocity);
    const unsigned int n_q_points = quadrature_formula.size();

    MappingQ<dim> mapping(parameters.degree_of_velocity);
    FEValues<dim> fe_values (mapping, fe_velocity, quadrature_formula, update_values);
    FEValues<dim> fe_concentr_values (mapping, concentr_fe, quadrature_formula, update_values);
    std::vector<Tensor<1,dim> > velocity_values(n_q_points);
    std::vector<double > concentr_values(n_q_points);

    const FEValuesExtractors::Vector velocities (0);

    double max_local_velocity = 0;

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_velocity.begin_active(),
    endc = dof_handler_velocity.end(),
    con_cell = concentr_dof_handler.begin_active ();
   
    for (; cell!=endc; ++cell, ++con_cell)
      if (cell->is_locally_owned())
        {
          if (cell->center()[parameters.flow_direction]> 1.0)
          {
            fe_values.reinit (cell);
            fe_concentr_values.reinit (con_cell);
           
            fe_values[velocities].get_function_values (vel_star,
                                                       velocity_values);
           
            fe_concentr_values.get_function_values (concentr_solution,
                                                    concentr_values);
           
            for (unsigned int q=0; q<n_q_points; ++q)
              max_local_velocity = std::max (max_local_velocity,
                                             velocity_values[q].norm()*(1.0-concentr_values[q]));
          }
        }

    return Utilities::MPI::max (max_local_velocity, MPI_COMM_WORLD);
  }
  
  template <int dim>
  double
  UBC_mis_mixing<dim>::get_entropy_variation (const double average_concentr) const
  {
    if (parameters.stabilization_alpha != 2)
      return 1.;

    const QGauss<dim> quadrature_formula (parameters.degree_of_concentr+1);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values (concentr_fe, quadrature_formula,
                             update_values | update_JxW_values);
    std::vector<double> old_concentr_values(n_q_points);
    std::vector<double> old_old_concentr_values(n_q_points);

    double min_entropy = std::numeric_limits<double>::max(),
           max_entropy = -std::numeric_limits<double>::max(),
           area = 0,
           entropy_integrated = 0;

    typename DoFHandler<dim>::active_cell_iterator
    cell = concentr_dof_handler.begin_active(),
    endc = concentr_dof_handler.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit (cell);
          fe_values.get_function_values (old_concentr_solution,
                                         old_concentr_values);
          fe_values.get_function_values (old_old_concentr_solution,
                                         old_old_concentr_values);
          for (unsigned int q=0; q<n_q_points; ++q)
            {
              const double T = (old_concentr_values[q] +
                                old_old_concentr_values[q]) / 2;
              const double entropy = ((T-average_concentr) *
                                      (T-average_concentr));

              min_entropy = std::min (min_entropy, entropy);
              max_entropy = std::max (max_entropy, entropy);
              area += fe_values.JxW(q);
              entropy_integrated += fe_values.JxW(q) * entropy;
            }
        }
    const double local_sums[2]   = { entropy_integrated, area },
                 local_maxima[2] = { -min_entropy, max_entropy };
    double global_sums[2], global_maxima[2];

    Utilities::MPI::sum (local_sums,   MPI_COMM_WORLD, global_sums);
    Utilities::MPI::max (local_maxima, MPI_COMM_WORLD, global_maxima);

    const double average_entropy = global_sums[0] / global_sums[1];
    const double entropy_diff = std::max(global_maxima[1] - average_entropy,
                                         average_entropy - (-global_maxima[0]));
    return entropy_diff;
  }
  
  template <int dim>
  std::pair<double,double>
  UBC_mis_mixing<dim>::get_extrapolated_concentr_range () const
  {
    const QIterated<dim> quadrature_formula (QTrapez<1>(),
                                             parameters.degree_of_concentr);
    const unsigned int n_q_points = quadrature_formula.size();

    MappingQ<dim> mapping(concentr_fe.get_degree());
    FEValues<dim> fe_values (mapping, concentr_fe, quadrature_formula,
                             update_values);
    std::vector<double> old_concentr_values(n_q_points);
    std::vector<double> old_old_concentr_values(n_q_points);

    double min_local_concentr = +std::numeric_limits<double>::max(),
           max_local_concentr = -std::numeric_limits<double>::max();

    if (timestep_number != 0)
    {
      typename DoFHandler<dim>::active_cell_iterator
      cell = concentr_dof_handler.begin_active(),
      endc = concentr_dof_handler.end();
      for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
      {
        fe_values.reinit (cell);
        fe_values.get_function_values (old_concentr_solution,
                                       old_concentr_values);
        fe_values.get_function_values (old_old_concentr_solution,
                                       old_old_concentr_values);

        for (unsigned int q=0; q<n_q_points; ++q)
        {
          const double concentr =
            (1. + time_step/old_time_step) * old_concentr_values[q]-
            time_step/old_time_step * old_old_concentr_values[q];

          min_local_concentr = std::min (min_local_concentr, concentr);
          max_local_concentr = std::max (max_local_concentr, concentr);
        }
      }
    }
    else
    {
      typename DoFHandler<dim>::active_cell_iterator
      cell = concentr_dof_handler.begin_active(),
      endc = concentr_dof_handler.end();
      for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
      {
        fe_values.reinit (cell);
        fe_values.get_function_values (old_concentr_solution,
                                       old_concentr_values);

        for (unsigned int q=0; q<n_q_points; ++q)
        {
          const double concentr = old_concentr_values[q];

          min_local_concentr = std::min (min_local_concentr, concentr);
          max_local_concentr = std::max (max_local_concentr, concentr);
        }
      }
    }

    double local_extrema[2] = { -min_local_concentr,
                                max_local_concentr
                              };
    double global_extrema[2];
    Utilities::MPI::max (local_extrema, MPI_COMM_WORLD, global_extrema);

    return std::make_pair(-global_extrema[0], global_extrema[1]);
  }

  template <int dim>
  double
  UBC_mis_mixing<dim>::
  compute_entropy_viscosity_for_hyperbolic (const std::vector<double>          &old_concentr,
                                            const std::vector<double>          &old_old_concentr,
                                            const std::vector<Tensor<1,dim> >  &old_concentr_grads,
                                            const std::vector<Tensor<1,dim> >  &old_old_concentr_grads,
                                            const std::vector<double>          &old_concentr_laplacians,
                                            const std::vector<double>          &old_old_concentr_laplacians,
                                            const std::vector<Tensor<1,dim> >  &old_velocity_values,
                                            const std::vector<Tensor<1,dim> >  &old_old_velocity_values,
                                            const std::vector<SymmetricTensor<2,dim> >  &old_strain_rates,
                                            const std::vector<SymmetricTensor<2,dim> >  &old_old_strain_rates,
                                            const double                        global_u_infty,
                                            const double                        global_T_variation,
                                            const double                        average_concentr,
                                            const double                        global_entropy_variation,
                                            const double                        cell_diameter) const
  {
    if (global_u_infty == 0)
      return 5e-3 * cell_diameter;
    
    const unsigned int n_q_points = old_concentr.size();

    double max_residual = 0;
    double max_velocity = 0;

    for (unsigned int q=0; q < n_q_points; ++q)
      {
        const Tensor<1,dim> u = (old_velocity_values[q] +
                                 old_old_velocity_values[q]) / 2;

        const SymmetricTensor<2,dim> strain_rate = (old_strain_rates[q] +
                                                    old_old_strain_rates[q]) / 2;

        const double T = (old_concentr[q] + old_old_concentr[q]) / 2;
        const double dT_dt = (old_concentr[q] - old_old_concentr[q])
                             / old_time_step;
        const double u_grad_T = u * (old_concentr_grads[q] +
                                     old_old_concentr_grads[q]) / 2;

//         const double kappa_Delta_T = EquationData::kappa
//                                      * (old_concentr_laplacians[q] +
//                                         old_old_concentr_laplacians[q]) / 2;
//         const double gamma
//           = ((EquationData::radiogenic_heating * EquationData::density(T)
//               +
//               2 * EquationData::eta * strain_rate * strain_rate) /
//              (EquationData::density(T) * EquationData::specific_heat));

        const double kappa_Delta_T = 0.0;
        const double gamma = 0.0;
 
        double residual
          = std::abs(dT_dt + u_grad_T - kappa_Delta_T - gamma);
        if (parameters.stabilization_alpha == 2)
          residual *= std::abs(T - average_concentr);

        max_residual = std::max (residual,        max_residual);
        max_velocity = std::max (std::sqrt (u*u), max_velocity);
      }

    const double max_viscosity = (parameters.stabilization_beta *
                                  max_velocity * cell_diameter);
    if (timestep_number == 0 && parameters.is_restart == false)
      return max_viscosity;
    else
      {
        Assert (old_time_step > 0, ExcInternalError());

        double entropy_viscosity;
        if (parameters.stabilization_alpha == 2)
          entropy_viscosity = (parameters.stabilization_c_R *
                               cell_diameter * cell_diameter *
                               max_residual /
                               global_entropy_variation);
        else
          entropy_viscosity = (parameters.stabilization_c_R *
                               cell_diameter * global_Omega_diameter *
                               max_velocity * max_residual /
                               (global_u_infty * global_T_variation));

        return std::min (max_viscosity, entropy_viscosity);
      }
  }
  
  template <int dim>
  std::pair<double, double>
  UBC_mis_mixing<dim>::
  compute_entropy_viscosity_for_navier_stokes(
                              const std::vector<Tensor<1,dim> >     &old_velocity,
                              const std::vector<Tensor<1,dim> >     &old_old_velocity,
                              const std::vector<Tensor<2,dim> >     &old_velocity_star_grads,
                              const std::vector<Tensor<1,dim> >     &old_velocity_laplacians,
                              const std::vector<Tensor<1,dim> >     &old_pressure_grads,
                              const Tensor<1,dim>                    &source_vector,
                              const double                           coeff1_for_adv,
                              const double                           coeff2_for_visco,
                              const double                           coeff3_for_source,
                              const double                           cell_diameter) const
  {

    if (timestep_number < 2) return std::make_pair (0.0, 0.0);

    const unsigned int n_q_points = old_velocity.size();

    double avr_min_local_viscosity = 0.0;
    double avr_min_local_residual = 0.0;

    for (unsigned int q=0; q<n_q_points; ++q)
    {
      const Tensor<1,dim> u = old_velocity[q];

      const Tensor<1,dim> du_dt = coeff1_for_adv*
                                  (old_velocity[q]
                                   - old_old_velocity[q])
                                   /old_time_step;

      const Tensor<1,dim> u_grad_u = coeff1_for_adv*
                                     u*old_velocity_star_grads[q];

      const Tensor<1,dim> u_viscous = coeff2_for_visco*
                                      old_velocity_laplacians[q];

      const Tensor<1,dim> p_grad = old_pressure_grads[q];

      double residual = du_dt*u + u_grad_u*u
                        + p_grad*u - u_viscous*u
                        - coeff3_for_source*source_vector*u;

      double numer_viscosity = parameters.coeff_arti_viscosity*
                               cell_diameter*cell_diameter*
                               (std::abs(residual)/(u*u));

      double max_bound_viscosity = parameters.maximum_coeff_arti_viscosity*u.norm()*cell_diameter;

      double min_local_viscosity = std::min (max_bound_viscosity, numer_viscosity);

      avr_min_local_viscosity += min_local_viscosity;

      avr_min_local_residual  += residual;
    }

    return std::make_pair (avr_min_local_residual/double(n_q_points),
                           avr_min_local_viscosity/double(n_q_points));
  }

  template <int dim>
  void UBC_mis_mixing<dim>::project_concentr_field ()
  {
    pcout << "* Projection concentr Field.. ";
    
    QGauss<dim> quadrature(parameters.degree_of_concentr+2);
    UpdateFlags update_flags = UpdateFlags(update_values   |
                                           update_quadrature_points |
                                           update_JxW_values);
    MappingQ<dim> mapping(concentr_fe.get_degree());
    FEValues<dim> fe_values (mapping, concentr_fe, quadrature, update_flags);

    const unsigned int dofs_per_cell = fe_values.dofs_per_cell,
                       n_q_points    = fe_values.n_quadrature_points;

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    Vector<double> cell_vector (dofs_per_cell);
    FullMatrix<double> matrix_for_bc (dofs_per_cell, dofs_per_cell);

    std::vector<double> rhs_values(n_q_points);

    IndexSet row_concentr_matrix_partitioning(concentr_mass_matrix.n());
    row_concentr_matrix_partitioning.add_range(concentr_mass_matrix.local_range().first,
                                           concentr_mass_matrix.local_range().second);
    TrilinosWrappers::MPI::Vector rhs (row_concentr_matrix_partitioning),
                                  solution (row_concentr_matrix_partitioning);
   
    concentr_mass_matrix = 0;
     
    double dum = parameters.init_sep_x * 
                 parameters.length_of_domain [parameters.flow_direction];
 
    EquationData::concentrInitialValues<dim> initial_concentr (dum);

    typename DoFHandler<dim>::active_cell_iterator
    cell = concentr_dof_handler.begin_active(),
    endc = concentr_dof_handler.end();

    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices (local_dof_indices);
          fe_values.reinit (cell);

          initial_concentr.value_list (fe_values.get_quadrature_points(),rhs_values);

          cell_vector = 0;
          matrix_for_bc = 0;
          for (unsigned int point=0; point<n_q_points; ++point)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
           cell_vector(i) += rhs_values[point] *
                                fe_values.shape_value(i,point) *
                                fe_values.JxW(point);

//           if (concentr_constraints.is_inhomogeneously_constrained(local_dof_indices[i]))
           {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
             matrix_for_bc(i, i) += fe_values.shape_value(i,point) *
                                             fe_values.shape_value(j,point) *
                                             fe_values.JxW(point);
           }
          }

          concentr_constraints.distribute_local_to_global (cell_vector,
                                                           local_dof_indices,
                                                           rhs);
   
          concentr_constraints.distribute_local_to_global (matrix_for_bc,
                                                           local_dof_indices,
                                                           concentr_mass_matrix);
   
        }

    rhs.compress (VectorOperation::add);
    concentr_mass_matrix.compress (VectorOperation::add);

    SolverControl solver_control(5*rhs.size(), 1e-12*rhs.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector> cg(solver_control);

    TrilinosWrappers::PreconditionJacobi preconditioner_mass;
    preconditioner_mass.initialize(concentr_mass_matrix, 1.3);

    cg.solve (concentr_mass_matrix, solution, rhs, preconditioner_mass);

    concentr_constraints.distribute (solution);

    for (unsigned int i=0; i<solution.local_size(); ++i) 
    {
      double temp_value = solution.trilinos_vector()[0][i];
      if (temp_value > 1.0) solution.trilinos_vector()[0][i] = 1.0;
      if (temp_value < 0.0) solution.trilinos_vector()[0][i] = 0.0;
    }

    concentr_solution = solution;
    old_concentr_solution = solution;
    old_old_concentr_solution = solution;
    
    pcout << solver_control.last_step() << std::endl;
  }

// Explicit instantiations
// template class UBC_mis_mixing<1>;
template class UBC_mis_mixing<2>;
template class UBC_mis_mixing<3>;
