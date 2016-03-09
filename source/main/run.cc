#include <mismix/include.h>
#include <mismix/equation_data.h>
#include <mismix/assembly_copydata.h>
#include <mismix/class.h>
#include <mismix/parameter.h>

  template <int dim>
  void UBC_mis_mixing<dim>::run ()
  {
      pcout << std::endl;
      
      unsigned int index_plotting = 0;
      timestep_number = 0;
      
      std::ostringstream filename_avr_c;
      filename_avr_c << "output/data/"+ Utilities::int_to_string(parameters.data_id, 2) + "_avr_c.dat";
      std::ofstream out_avr_c (filename_avr_c.str().c_str());

      std::ostringstream filename_avr_vFlow_c;
      filename_avr_vFlow_c << "output/data/"+ Utilities::int_to_string(parameters.data_id, 2) + "_avr_vFlow_c.dat";
      std::ofstream out_avr_vFlow_c (filename_avr_vFlow_c.str().c_str());

      std::ostringstream filename_avr_vvFlow_c;
      filename_avr_vvFlow_c << "output/data/"+ Utilities::int_to_string(parameters.data_id, 2) + "_avr_vvFlow_c.dat";
      std::ofstream out_avr_vvFlow_c (filename_avr_vvFlow_c.str().c_str());

      std::ostringstream filename_avr_vvLati_c;
      filename_avr_vvLati_c << "output/data/"+ Utilities::int_to_string(parameters.data_id, 2) + "_avr_vvLati_c.dat";
      std::ofstream out_avr_vvLati_c (filename_avr_vvLati_c.str().c_str());

      std::ostringstream filename_avr_vvDept_c;
      filename_avr_vvDept_c << "output/data/"+ Utilities::int_to_string(parameters.data_id, 2) + "_avr_vvDept_c.dat";
      std::ofstream out_avr_vvDept_c (filename_avr_vvDept_c.str().c_str());

      std::ostringstream filename_avr_output;
      filename_avr_output << "output/data/"+ Utilities::int_to_string(parameters.data_id, 2) + "_output.dat";
      std::ofstream out_avr_output (filename_avr_output.str().c_str());

      std::ostringstream filename_avr_error;
      filename_avr_error << "output/data/"+ Utilities::int_to_string(parameters.data_id, 2) + "_error.dat";
      std::ofstream out_avr_error (filename_avr_error.str().c_str());


      if (parameters.is_restart)
      {
         pcout << "* --> Re-Start from Checkpoint..." << std::endl;
         load_data_and_mesh ();
         
         index_plotting  = parameters.index_for_restart + 1;
         timestep_number = parameters.restart_no_timestep + 1;

      }
      else
      {
         pcout << "* --> New Start..." << std::endl;
         
         create_triangulation ();
         assgined_boundary_indicator ();
      
         if (parameters.max_grid_level > 0)
           initial_refine_mesh ();
         
         setup_dofs_velocity ();
         setup_dofs_pressure ();
         setup_dofs_auxilary ();
         setup_dofs_concentr ();
         setup_dofs_error ();

         project_concentr_field ();
         output_results (index_plotting);

         compute_for_avrage_quantities_surfIntgl (out_avr_c,
                                                  out_avr_vFlow_c,
                                                  out_avr_vvFlow_c,
                                                  out_avr_vvLati_c,
                                                  out_avr_vvDept_c);
         compute_global_error_norm (out_avr_error);

         timestep_number = 1;
         index_plotting = 1;

      }
      
      print_input_parameters ();
      double given_time_step = 0.0;
      unsigned int delay_timestep_number = 0;
      unsigned int variable_refine_period = parameters.no_refine_period;
      double increase_dimless_vel = 0.1;
      
      pcout << "* Art Viscosity = " << parameters.coeff_arti_viscosity << std::endl;
      pcout << "* Max Art Viscosity = " << parameters.maximum_coeff_arti_viscosity << std::endl;
      pcout << "* Is Exclude Depth Convection = " << parameters.exclude_depth_direction << std::endl;
      do
      {
         pcout << "******************** NO. TIME STEP = " << timestep_number << std::endl;
         pcout << "* Time Interval = " << time_step << std::endl;
         given_time_step = time_step = old_time_step = computed_time_step;

         if (timestep_number == 1)
           variable_refine_period = parameters.intial_ratio_refinement*parameters.no_refine_period;
         else if (timestep_number > parameters.intial_ratio_refinement*parameters.no_refine_period)
           variable_refine_period = parameters.no_refine_period;

//===================================================================================/

         {
           unsigned int size_increse_dimless_vel = static_cast<unsigned int>(parameters.Reynolds_number/10);
           increase_dimless_vel = timestep_number*(1./size_increse_dimless_vel);
           if (increase_dimless_vel > 1.0) increase_dimless_vel = 1.0;

           solve_fluid_equation (increase_dimless_vel);
         
           double maximal_velocity = get_maximal_velocity();
           double mm = 1.0*(maximal_velocity);
           unsigned int rrr = static_cast<unsigned int> (mm) + 1;
           time_step = (1./static_cast<double>(rrr))*computed_time_step;
           old_time_step = time_step;

           pcout << "* "
                  << "Max. V = "
                  << maximal_velocity << ", "
                  << old_time_step << " (O) | "
                  << time_step << " (C)"
                  << std::endl;

           pcout <<"* Solve CD Eq... " << std::endl;

           assemble_concentr_matrix ();
           for (unsigned int i=0; i<rrr; ++i)
           {
             pcout << "*** BDF2... " << i << std::endl;
             solve_concentr_equation (maximal_velocity);
           }
           old_time_step = time_step = computed_time_step;

           solution_update ();
         }

         {
           compute_for_avrage_quantities_surfIntgl (out_avr_c,
                                                    out_avr_vFlow_c,
                                                    out_avr_vvFlow_c,
                                                    out_avr_vvLati_c,
                                                    out_avr_vvDept_c);

           compute_global_error_norm (out_avr_error);

           out_avr_output << timestep_number << " " << given_time_step << " "
                          << timestep_number*given_time_step << std::endl;

           if (timestep_number%parameters.output_fac_vtu == 0)
           {
             output_results (index_plotting);
             save_data_and_mesh ();
             out_avr_output << timestep_number << " "
                            << index_plotting  << "SAVE DATA AND MESH DONE"
                            << std::endl;
             ++index_plotting;
           }
         }

         {
           if (timestep_number%variable_refine_period == 0
               &&
               parameters.max_grid_level > 0)
             refine_mesh (parameters.max_grid_level);
         }

//===================================================================================//

         pcout << std::endl;
         ++timestep_number;
      } while (timestep_number < 1000000 + 1);
  }

// Explicit instantiations
// template class UBC_mis_mixing<1>;
template class UBC_mis_mixing<2>;
template class UBC_mis_mixing<3>;
