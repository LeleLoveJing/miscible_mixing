#include <mismix/include.h>
#include <mismix/equation_data.h>
#include <mismix/assembly_copydata.h>
#include <mismix/class.h>
#include <mismix/parameter.h>

  template <int dim>
  class UBC_mis_mixing<dim>::Postprocessor : public DataPostprocessor<dim>
  {
   public:
   Postprocessor (const unsigned int partition);

   virtual
   void
   compute_derived_quantities_vector (const std::vector<Vector<double> >               &uh,
                                       const std::vector<std::vector<Tensor<1,dim> > > &duh,
     const std::vector<std::vector<Tensor<2,dim> > > &dduh,
     const std::vector<Point<dim> >                  &normals,
     const std::vector<Point<dim> >                  &evaluation_points,
     std::vector<Vector<double> >                    &computed_quantities) const;
           
   virtual std::vector<std::string> get_names () const;

   virtual
   std::vector<DataComponentInterpretation::DataComponentInterpretation>
   get_data_component_interpretation () const;

   virtual UpdateFlags get_needed_update_flags () const;

   private:
   const unsigned int partition;
  };
  
  template <int dim>
  UBC_mis_mixing<dim>::Postprocessor::
  Postprocessor (const unsigned int partition)
    :
    partition (partition)
  {}
  
  template <int dim>
  std::vector<std::string>
  UBC_mis_mixing<dim>::Postprocessor::get_names() const
  {
    std::vector<std::string> solution_names (dim, "V");
    solution_names.push_back ("P");
    solution_names.push_back ("C");
    solution_names.push_back ("ENT_NS");
    solution_names.push_back ("REU_NS");
    solution_names.push_back ("Partition");

    return solution_names;
  }
  
  template <int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  UBC_mis_mixing<dim>::Postprocessor::
  get_data_component_interpretation () const
  {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation (dim,
                    DataComponentInterpretation::component_is_part_of_vector);

    interpretation.push_back (DataComponentInterpretation::component_is_scalar);
    interpretation.push_back (DataComponentInterpretation::component_is_scalar);
    interpretation.push_back (DataComponentInterpretation::component_is_scalar);

    interpretation.push_back (DataComponentInterpretation::component_is_scalar);
    interpretation.push_back (DataComponentInterpretation::component_is_scalar);
    
    return interpretation;
  }
  
  template <int dim>
  UpdateFlags
  UBC_mis_mixing<dim>::Postprocessor::get_needed_update_flags() const
  {
    return update_values | update_gradients | update_q_points;
  }
  
  template <int dim>
  void
  UBC_mis_mixing<dim>::Postprocessor::
  compute_derived_quantities_vector (const std::vector<Vector<double> >              &uh,
                                     const std::vector<std::vector<Tensor<1,dim> > > &duh,
                                     const std::vector<std::vector<Tensor<2,dim> > > &/*dduh*/,
                                     const std::vector<Point<dim> >                  &/*normals*/,
                                     const std::vector<Point<dim> >                  &/*evaluation_points*/,
                                     std::vector<Vector<double> >                    &computed_quantities) const
  {
    const unsigned int n_quadrature_points = uh.size();
    for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
       for (unsigned int d=0; d<dim; ++d)
         computed_quantities[q](d) = (uh[q](d));

       computed_quantities[q](dim) = uh[q](dim);

       computed_quantities[q](dim+1) = uh[q](dim+1);

       computed_quantities[q](dim+2) = uh[q](dim+2);

       computed_quantities[q](dim+3) = uh[q](dim+3);

       computed_quantities[q](dim+4) = partition;
    }
  }
  
  template <int dim>
  void UBC_mis_mixing<dim>::output_results (unsigned int out_index)
  {
    pcout << std::endl;
    pcout << "#######################  * Output Results... " << out_index << std::endl;
    pcout << std::endl;
    
    const FESystem<dim> joint_fe (fe_velocity, 1,
                                  fe_pressure, 1,
                                  concentr_fe, 1,
                                  fe_error, 1,
                                  fe_error, 1);

    DoFHandler<dim> joint_dof_handler (triangulation);
    joint_dof_handler.distribute_dofs (joint_fe);
    Assert (joint_dof_handler.n_dofs() ==
      dof_handler_velocity.n_dofs() +
      dof_handler_pressure.n_dofs() +
      concentr_dof_handler.n_dofs() +
      dof_handler_error.n_dofs () +
      dof_handler_error.n_dofs (),
      ExcInternalError());

    TrilinosWrappers::MPI::Vector joint_solution;
    joint_solution.reinit (joint_dof_handler.locally_owned_dofs(), MPI_COMM_WORLD);

    {
      std::vector<types::global_dof_index> local_joint_dof_indices (joint_fe.dofs_per_cell);
      std::vector<types::global_dof_index> local_velocity_dof_indices (fe_velocity.dofs_per_cell);
      std::vector<types::global_dof_index> local_pressure_dof_indices (fe_pressure.dofs_per_cell);
      std::vector<types::global_dof_index> local_concentr_dof_indices (concentr_fe.dofs_per_cell);
      std::vector<types::global_dof_index> local_error_dof_indices (fe_error.dofs_per_cell);
      
      typename DoFHandler<dim>::active_cell_iterator
        joint_cell  = joint_dof_handler.begin_active(),
        joint_endc  = joint_dof_handler.end(),
        velocity_cell  = dof_handler_velocity.begin_active(),
        pressure_cell  = dof_handler_pressure.begin_active(),
        concentr_cell  = concentr_dof_handler.begin_active(),
        error_cell     = dof_handler_error.begin_active();
      
      for (; joint_cell!=joint_endc;
           ++joint_cell, ++velocity_cell, ++pressure_cell, ++concentr_cell, ++error_cell)
        if (joint_cell->is_locally_owned())
          {
            joint_cell->get_dof_indices (local_joint_dof_indices);
            velocity_cell->get_dof_indices (local_velocity_dof_indices);
            pressure_cell->get_dof_indices (local_pressure_dof_indices);
            concentr_cell->get_dof_indices (local_concentr_dof_indices);
            error_cell->get_dof_indices (local_error_dof_indices);
     
            for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
              if (joint_fe.system_to_base_index(i).first.first == 0)
                {
                  joint_solution(local_joint_dof_indices[i])
                    = vel_star (local_velocity_dof_indices
                               [joint_fe.system_to_base_index(i).second]);
                }
              else if (joint_fe.system_to_base_index(i).first.first == 1)
                {
                  joint_solution(local_joint_dof_indices[i])
                    = pre_n_plus_1 (local_pressure_dof_indices
                                   [joint_fe.system_to_base_index(i).second]);
                }
              else if (joint_fe.system_to_base_index(i).first.first == 2)
                {
                  joint_solution(local_joint_dof_indices[i])
                    = concentr_solution (local_concentr_dof_indices
                         [joint_fe.system_to_base_index(i).second]);
                }
              else if (joint_fe.system_to_base_index(i).first.first == 3)
                {
                  joint_solution(local_joint_dof_indices[i])
                    = entropy_viscosity_for_ns (local_error_dof_indices
                         [joint_fe.system_to_base_index(i).second]);
                }
              else if (joint_fe.system_to_base_index(i).first.first == 4)
                {
                  joint_solution(local_joint_dof_indices[i])
                    = energy_norm_for_ns (local_error_dof_indices
                         [joint_fe.system_to_base_index(i).second]);
                }
          }
    }

    joint_solution.compress(VectorOperation::insert);

    IndexSet locally_relevant_joint_dofs(joint_dof_handler.n_dofs());
    DoFTools::extract_locally_relevant_dofs (joint_dof_handler, locally_relevant_joint_dofs);
    TrilinosWrappers::MPI::Vector locally_relevant_joint_solution;
    locally_relevant_joint_solution.reinit (locally_relevant_joint_dofs, MPI_COMM_WORLD);
    locally_relevant_joint_solution = joint_solution;

    Postprocessor postprocessor (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));

    DataOut<dim> data_out;
    data_out.attach_dof_handler (joint_dof_handler);
    data_out.add_data_vector (locally_relevant_joint_solution, postprocessor);

    data_out.build_patches (2);

    const std::string filename = ("output/vtu/s-" +
                                  Utilities::int_to_string (out_index, 4) +
                                  "." +
                                  Utilities::int_to_string
                                  (triangulation.locally_owned_subdomain(), 4) +
                                  ".vtu");
    std::ofstream output (filename.c_str());
    data_out.write_vtu (output);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
          filenames.push_back (std::string("s-") +
                               Utilities::int_to_string (out_index, 4) +
                               "." +
                               Utilities::int_to_string(i, 4) +
                               ".vtu");
        const std::string
        pvtu_master_filename = ("output/vtu/s-" +
                                Utilities::int_to_string (out_index, 4) +
                                ".pvtu");
        std::ofstream pvtu_master (pvtu_master_filename.c_str());
        data_out.write_pvtu_record (pvtu_master, filenames);
      }
  }  

// Explicit instantiations
// template class UBC_mis_mixing<1>;
template class UBC_mis_mixing<2>;
template class UBC_mis_mixing<3>;
