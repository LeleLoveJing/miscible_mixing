#include <mismix/include.h>
#include <mismix/equation_data.h>
#include <mismix/assembly_copydata.h>
#include <mismix/class.h>
#include <mismix/parameter.h>
#include <deal.II/base/types.h>

  template <int dim>
  void 
  UBC_mis_mixing<dim>::
  setup_matrix_velocity (const IndexSet &velocity_partitioning)
  {
    matrix_velocity.clear ();
    TrilinosWrappers::SparsityPattern sp (velocity_partitioning,
                                          MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern ( dof_handler_velocity,
                                      sp,
                                      constraints_velocity, 
                                      false,
                                      Utilities::MPI::
                                      this_mpi_process(MPI_COMM_WORLD));
    sp.compress();
    matrix_velocity.reinit (sp);
  }
  
  template <int dim>
  void 
  UBC_mis_mixing<dim>::
  setup_matrix_pressure (const IndexSet &pressure_partitioning)
  {
    matrix_pressure.clear ();
    TrilinosWrappers::SparsityPattern sp (pressure_partitioning,
                                          MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern ( dof_handler_pressure,
                                      sp,
                                      constraints_pressure, 
                                      false,
                                      Utilities::MPI::
                                      this_mpi_process(MPI_COMM_WORLD));
    sp.compress();
    matrix_pressure.reinit (sp);
  }
  
  template <int dim>
  void 
  UBC_mis_mixing<dim>::
  setup_matrix_auxilary (const IndexSet &auxilary_partitioning)
  {
    matrix_auxilary.clear ();
    TrilinosWrappers::SparsityPattern sp (auxilary_partitioning,
                                          MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern ( dof_handler_auxilary,
                                      sp,
                                      constraints_auxilary, 
                                      false,
                                      Utilities::MPI::
                                      this_mpi_process(MPI_COMM_WORLD));
    sp.compress();
    matrix_auxilary.reinit (sp);
  }
  
  template <int dim>
  void UBC_mis_mixing<dim>::
  setup_concentr_matrices (const IndexSet &concentr_partitioner)
  {
    T_preconditioner.reset ();
    concentr_mass_matrix.clear ();
    concentr_stiffness_matrix.clear ();
    concentr_matrix.clear ();

    TrilinosWrappers::SparsityPattern sp (concentr_partitioner,
                                          MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern (concentr_dof_handler, sp,
                                     concentr_constraints, false,
                                     Utilities::MPI::
                                     this_mpi_process(MPI_COMM_WORLD));
    sp.compress();

    concentr_matrix.reinit (sp);
    concentr_mass_matrix.reinit (sp);
    concentr_stiffness_matrix.reinit (sp);
  }

  
  template <int dim>
  void 
  UBC_mis_mixing<dim>::setup_dofs_velocity ()
  {
    pcout << "* Setup dofs for velocity..." << std::endl;
    
      
    dof_handler_velocity.distribute_dofs (fe_velocity);
    
    unsigned int n_u = dof_handler_velocity.n_dofs();
   
    std::locale s = pcout.get_stream().getloc();
    pcout.get_stream().imbue(std::locale(""));
    pcout << "* Elements = "
          << triangulation.n_global_active_cells()
          << std::endl
          << std::endl;
    pcout.get_stream().imbue(s);   
    
    IndexSet velocity_partitioning (n_u), velocity_relevant_partitioning (n_u);
    velocity_partitioning = dof_handler_velocity.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs (dof_handler_velocity,
                              velocity_relevant_partitioning);
    {
      constraints_velocity.clear ();
      constraints_velocity.reinit (velocity_relevant_partitioning);

      DoFTools::make_hanging_node_constraints (dof_handler_velocity,
                                               constraints_velocity);

      FEValuesExtractors::Vector velocity_components(0);

      VectorTools::interpolate_boundary_values (dof_handler_velocity,
                                                0,
																																																ZeroFunction<dim>(dim),
																																																constraints_velocity);

      DoFTools::make_periodicity_constraints (dof_handler_velocity,
                                              8, 9,
                                              parameters.flow_direction,
                                              constraints_velocity);

//      VectorTools::interpolate_boundary_values (dof_handler_velocity,
//                                                3,
//                                                EquationData::
//                                                Inflow_Velocity<dim> (1, 1),
//                                                constraints_velocity);

      if (dim == 3 && parameters.is_symmetry_boundary == true)
      {
      	 ComponentMask symmetric_bnd_velocity (dim, false);
        symmetric_bnd_velocity.set (parameters.depth_direction, true);

        VectorTools::interpolate_boundary_values (dof_handler_velocity,
                                                  6,
																																																  ZeroFunction<dim>(dim),
																																																  constraints_velocity,
																																																  symmetric_bnd_velocity);
      }
      constraints_velocity.close ();
    }
    
    setup_matrix_velocity   (velocity_partitioning);
    
    vel_star.reinit       (velocity_relevant_partitioning, MPI_COMM_WORLD);
    vel_star_old.reinit      (velocity_relevant_partitioning, MPI_COMM_WORLD);
    vel_n_plus_1.reinit      (velocity_relevant_partitioning, MPI_COMM_WORLD);
    vel_n.reinit           (velocity_relevant_partitioning, MPI_COMM_WORLD);
    vel_n_minus_1.reinit      (velocity_relevant_partitioning, MPI_COMM_WORLD);
    vel_n_minus_minus_1.reinit  (velocity_relevant_partitioning, MPI_COMM_WORLD);

//    vel_energy.reinit  (velocity_relevant_partitioning, MPI_COMM_WORLD);
//    old_vel_energy.reinit  (velocity_relevant_partitioning, MPI_COMM_WORLD);

    rhs_velocity.reinit      (velocity_partitioning, MPI_COMM_WORLD);

  }

  template <int dim>
  void 
  UBC_mis_mixing<dim>::setup_dofs_pressure ()
  {
    pcout << "* Setup dofs for pressure..." << std::endl;
    
    dof_handler_pressure.distribute_dofs (fe_pressure);
    
    unsigned int n_p = dof_handler_pressure.n_dofs();
    
    IndexSet pressure_partitioning (n_p), pressure_relevant_partitioning (n_p);
    pressure_partitioning = dof_handler_pressure.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs (dof_handler_pressure,
                                             pressure_relevant_partitioning);

    {
      constraints_pressure.clear ();
      constraints_pressure.reinit (pressure_relevant_partitioning);

      DoFTools::make_hanging_node_constraints (dof_handler_pressure,
                                               constraints_pressure);

      if (parameters.ist_pressure_boundary == 0)
      {
        VectorTools::interpolate_boundary_values (dof_handler_pressure,
                                                  4,
                                                  EquationData::
                                                  Outflow_Pressure<dim>(parameters.inclined_angle_vector[1],
                                                                        parameters.Froude_number),
                                                  constraints_pressure);
        
        VectorTools::interpolate_boundary_values (dof_handler_pressure,
                                                  5,
                                                  EquationData::
                                                  Outflow_Pressure<dim>(parameters.inclined_angle_vector[1],
                                                                        parameters.Froude_number),
                                                  constraints_pressure);
      }
     
      constraints_pressure.close ();
    }

    setup_matrix_pressure   (pressure_partitioning);
      
    pre_star.reinit   (pressure_relevant_partitioning, MPI_COMM_WORLD);
    pre_n_plus_1.reinit  (pressure_relevant_partitioning, MPI_COMM_WORLD);
    pre_n.reinit       (pressure_relevant_partitioning, MPI_COMM_WORLD);
    pre_n_minus_1.reinit  (pressure_relevant_partitioning, MPI_COMM_WORLD);
    rhs_pressure.reinit  (pressure_partitioning, MPI_COMM_WORLD);
 
  }

  template <int dim>
  void 
  UBC_mis_mixing<dim>::setup_dofs_auxilary ()
  {
    pcout << "* Setup dofs for auxilary..." << std::endl;
      
    dof_handler_auxilary.distribute_dofs (fe_auxilary);
    
    unsigned int n_a = dof_handler_auxilary.n_dofs();
    
    IndexSet auxilary_partitioning (n_a), auxilary_relevant_partitioning (n_a);
    auxilary_partitioning = dof_handler_auxilary.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs (dof_handler_auxilary,
                                             auxilary_relevant_partitioning);
    constraints_auxilary.clear ();
    constraints_auxilary.reinit (auxilary_relevant_partitioning);

    DoFTools::make_hanging_node_constraints (dof_handler_auxilary,
                                             constraints_auxilary);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==	0
								&&
							 parameters.domain_boundary[1] == 5
								&&
							 parameters.ist_pressure_boundary == 1)
    {
						std::set<types::boundary_id> boundary_set;
						boundary_set.insert(parameters.domain_boundary[0]);
						IndexSet selected_dofs (dof_handler_auxilary.n_dofs());
						ComponentMask selected_component (1, true);
						DoFTools::extract_boundary_dofs (dof_handler_auxilary,
																																							selected_component,
																																							selected_dofs,
																																							boundary_set);

						pcout << "selected_dofs.n_elements()  = " << selected_dofs.n_elements() << std::endl;

						Assert(selected_dofs.n_elements() > 0,
													ExcMessage ("No extract boundary dofs.."));

						std::vector<bool> boundary_dofs (dof_handler_auxilary.n_dofs(), true);

						types::global_dof_index first_boundary_dof = numbers::invalid_unsigned_int;

						for (unsigned int i=0; i<dof_handler_auxilary.n_dofs(); ++i)
						if (selected_dofs.is_element(i) == true)
						{
								first_boundary_dof = i;
								break;
						}

						pcout << "first_boundary_dof = " << first_boundary_dof << std::endl;

						constraints_auxilary.add_line (first_boundary_dof);
						for (unsigned int i=0; i<dof_handler_auxilary.n_dofs(); ++i)
							if (selected_dofs.is_element(i) == true && i != first_boundary_dof)
						{
								constraints_auxilary.add_entry (first_boundary_dof, i, -1);
						}
    }

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
								Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) - 1
								&&
							 parameters.domain_boundary[1] == 4
								&&
							 parameters.ist_pressure_boundary == 1)
    {
						std::set<types::boundary_id> boundary_set;
						boundary_set.insert(parameters.domain_boundary[1]);
						IndexSet selected_dofs (dof_handler_auxilary.n_dofs());
						ComponentMask selected_component (1, true);
						DoFTools::extract_boundary_dofs (dof_handler_auxilary,
																																							selected_component,
																																							selected_dofs,
																																							boundary_set);

						pcout << "selected_dofs.n_elements()  = " << selected_dofs.n_elements() << std::endl;

						Assert(selected_dofs.n_elements() > 0,
													ExcMessage ("No extract boundary dofs.."));

						std::vector<bool> boundary_dofs (dof_handler_auxilary.n_dofs(), true);

						types::global_dof_index first_boundary_dof = numbers::invalid_unsigned_int;

						for (unsigned int i=0; i<dof_handler_auxilary.n_dofs(); ++i)
						if (selected_dofs.is_element(i) == true)
						{
								first_boundary_dof = i;
								break;
						}

						pcout << "first_boundary_dof = " << first_boundary_dof << std::endl;

						constraints_auxilary.add_line (first_boundary_dof);
						for (unsigned int i=0; i<dof_handler_auxilary.n_dofs(); ++i)
							if (selected_dofs.is_element(i) == true && i != first_boundary_dof)
						{
								constraints_auxilary.add_entry (first_boundary_dof, i, -1);
						}
    }

				if (parameters.ist_pressure_boundary == 0)
				{
						VectorTools::interpolate_boundary_values (dof_handler_auxilary,
																																																4,
																																																ZeroFunction<dim>(1),
																																																constraints_auxilary);

						VectorTools::interpolate_boundary_values (dof_handler_auxilary,
																																																5,
																																																ZeroFunction<dim>(1),
																																																constraints_auxilary);
				}
     
    constraints_auxilary.close ();

    setup_matrix_auxilary   (auxilary_partitioning);
     
    aux_n_plus_1.reinit     (auxilary_relevant_partitioning, MPI_COMM_WORLD);
    aux_n.reinit            (auxilary_relevant_partitioning, MPI_COMM_WORLD);
    aux_n_minus_1.reinit    (auxilary_relevant_partitioning, MPI_COMM_WORLD);
    rhs_auxilary.reinit     (auxilary_partitioning, MPI_COMM_WORLD);
    
  }

  template <int dim>
  void 
  UBC_mis_mixing<dim>::setup_dofs_concentr ()
  {
    pcout << "* Setup dofs for concentr..." << std::endl;
      
    concentr_dof_handler.distribute_dofs (concentr_fe);
    
    unsigned int n_c = concentr_dof_handler.n_dofs();
     
    IndexSet concentr_partitioning (n_c), concentr_relevant_partitioning (n_c);
    concentr_partitioning = concentr_dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs (concentr_dof_handler,
                                             concentr_relevant_partitioning);
      
    {
      concentr_constraints.clear ();
      concentr_constraints.reinit (concentr_relevant_partitioning);
      
      DoFTools::make_hanging_node_constraints (concentr_dof_handler,
                                               concentr_constraints);

      VectorTools::interpolate_boundary_values (concentr_dof_handler,
                                                3,
                                                ZeroFunction<dim> (1),
                                                concentr_constraints);

      DoFTools::make_periodicity_constraints (concentr_dof_handler,
                                              8, 9,
                                              parameters.flow_direction,
                                              concentr_constraints);
      concentr_constraints.close ();
    }
    
    setup_concentr_matrices (concentr_partitioning);
      
    concentr_solution.reinit         (concentr_relevant_partitioning, MPI_COMM_WORLD);
    old_concentr_solution.reinit     (concentr_relevant_partitioning, MPI_COMM_WORLD);
    old_old_concentr_solution.reinit (concentr_relevant_partitioning, MPI_COMM_WORLD);
    post_error_crit1.reinit          (concentr_relevant_partitioning, MPI_COMM_WORLD);
    post_error_crit2.reinit          (concentr_relevant_partitioning, MPI_COMM_WORLD);
    concentr_rhs.reinit              (concentr_partitioning, MPI_COMM_WORLD);
    
    rebuild_concentr_matrices       = true;
    rebuild_concentr_preconditioner = true;

  }

  template <int dim>
  void
  UBC_mis_mixing<dim>::setup_dofs_error ()
  {
    pcout << "* Setup dofs for error..." << std::endl;

    dof_handler_error.distribute_dofs (fe_error);
    unsigned int n_e = dof_handler_error.n_dofs();
    IndexSet partitioning (n_e), relevant_partitioning (n_e);
    partitioning = dof_handler_error.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs (dof_handler_error,
                                             relevant_partitioning);

    entropy_viscosity_for_ns.reinit (partitioning, MPI_COMM_WORLD);
    energy_norm_for_ns.reinit       (partitioning, MPI_COMM_WORLD);

  }
// Explicit instantiations
// template class UBC_mis_mixing<1>;
template class UBC_mis_mixing<2>;
template class UBC_mis_mixing<3>;
