#ifndef __parameter_h__
#define __parameter_h__ 

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <limits>
#include <locale> 

#include "include.h"
#include "equation_data.h"
#include "class.h"

 template <int dim>
 UBC_mis_mixing<dim>::Parameters::Parameters (std::string &parameter_filename)
 :
 domain_size (6),
 domain_boundary (2)
 {
   ParameterHandler prm;
   UBC_mis_mixing<dim>::Parameters::declare_parameters (prm);
   std::ifstream parameter_file (parameter_filename.c_str());
   if (!parameter_file)
   {
     parameter_file.close ();
     std::ostringstream message;
     message << "Input parameter file <"
      << parameter_filename << "> not found. Creating a"
      << std::endl
      << "template file of the same name."
      << std::endl;
     std::ofstream parameter_out (parameter_filename.c_str());
     prm.print_parameters (parameter_out,
      ParameterHandler::Text);
     AssertThrow (false, ExcMessage (message.str().c_str()));
   }

   bool success = prm.read_input (parameter_file);
   AssertThrow (success, ExcMessage ("Invalid input parameter file."));
   parse_parameters (prm);
 }
 
 template <int dim>
 void 
 UBC_mis_mixing<dim>::Parameters::
 declare_parameters (ParameterHandler &prm)
 {
   prm.enter_subsection ("Mesh Information");
     prm.declare_entry ("Input File Name",   "",Patterns::Anything());
     prm.declare_entry ("Inlet Boundary",    "3" , Patterns::Integer(0, 100));
     prm.declare_entry ("Outlet Boundary",   "4" , Patterns::Integer(0, 100));
     prm.declare_entry ("Symmetry Boundary", "false" , Patterns::Bool());
     prm.declare_entry ("Number of Slices",    "1" , Patterns::Integer(1, 100000));
   prm.leave_subsection ();
 
   prm.enter_subsection ("Adaptive Mesh Refinement");
     prm.declare_entry ("Maximum Refinement Level" , "0" , Patterns::Integer(0,10000));
     prm.declare_entry ("Type Adaptivity Rule" , "0" , Patterns::Integer(0,10000));
     prm.declare_entry ("Error Threshold" , "0.01" , Patterns::Double(0, 1));
     prm.declare_entry ("Refinement Threshold" , "0.9" , Patterns::Double(0, 1));
     prm.declare_entry ("Coarsening Threshold" , "0.01" , Patterns::Double(0, 1));
     prm.declare_entry ("Initial Ratio" , "1" , Patterns::Integer(0,10000));
     prm.declare_entry ("Refinement Period" , "0" , Patterns::Integer(0,10000));
   prm.leave_subsection ();

   prm.enter_subsection ("Entropy Viscosity");
     prm.declare_entry ("Stablization Alpha", "1", Patterns::Double(0.0,100.));
     prm.declare_entry ("Stablization Beta", "1", Patterns::Double(0.0,100.));
     prm.declare_entry ("c_R Factor", "1", Patterns::Double(0.0,100.));
   prm.leave_subsection ();

   prm.enter_subsection ("Solve Algorithm");
     prm.declare_entry ("Optimization Method" , "0" , Patterns::Integer(0,10));
     prm.declare_entry ("Projection Method" , "0" , Patterns::Integer(0,10));
     prm.declare_entry ("Pressure Boundary" , "0" , Patterns::Integer(0,10));
     prm.declare_entry ("Relaxation for Div-Vel" , "0.0" , Patterns::Double(-1000000000,1000000000));
     prm.declare_entry ("Gamma for Grad-Div" , "0.0" , Patterns::Double(-1000000000,1000000000));
     prm.declare_entry ("Artificial Viscosity" , "0.5" , Patterns::Double(0,10));
     prm.declare_entry ("Max Artificial Viscosity" , "0.5" , Patterns::Double(0,10));
     prm.declare_entry ("Exclude Depth Convection" , "false" , Patterns::Bool());
   prm.leave_subsection ();
    
   prm.enter_subsection ("Parameters");
     prm.declare_entry ("Dimension" , "2" , Patterns::Integer(2,3));
     prm.declare_entry ("Verbal Output" , "false" , Patterns::Bool());
     prm.declare_entry ("CFL Number" , "0.28" , Patterns::Double(0,1));
     prm.declare_entry ("Initial Separation" , "0.0" , Patterns::Double(0.0, 1.0));
     prm.declare_entry ("Inclined Angle" , "0.0" , Patterns::Double(0, 100000));
     prm.declare_entry ("Atwood Number" , "1.0" , Patterns::Double(-100000, 100000));
     prm.declare_entry ("Mean Flow Velocity" , "1.0" , Patterns::Double(0, 100000));
     prm.declare_entry ("Degree of Velocity", "1", Patterns::Integer (0, 10));
     prm.declare_entry ("Degree of Pressure", "1", Patterns::Integer (0, 10));
     prm.declare_entry ("Degree of Concentration", "2", Patterns::Integer (0, 10));
   prm.leave_subsection ();

   prm.enter_subsection ("Constitutive Model");
     prm.declare_entry ("Mean Viscosity" , "1.0" , Patterns::Double(0.0, 100000));
     prm.declare_entry ("Viscosity Ratio" , "1.0" , Patterns::Double(0.0, 100000));
     prm.declare_entry ("Coefficient for Pow-Law Fluid", "1.0", Patterns::Double(0.0, 100000));
     prm.declare_entry ("Multiplier for Pow-Law Fluid", "0.0", Patterns::Double(-100000, 100000));
   prm.leave_subsection ();

   prm.enter_subsection ("Output");
     prm.declare_entry ("Data Id", "0",Patterns::Integer(0,10000));
     prm.declare_entry ("Output Period for VTU" , "0" , Patterns::Integer(0,10000));
     prm.declare_entry ("Output Period for Data" , "0" , Patterns::Integer(0,10000));
   prm.leave_subsection ();
    
   prm.enter_subsection ("Restart");
     prm.declare_entry ("Restart" , "false" , Patterns::Bool());
     prm.declare_entry ("Last Index for VTU"    , "0" , Patterns::Integer(0,10000));
     prm.declare_entry ("Last Time Step Number" , "0" , Patterns::Integer(0,10000));
   prm.leave_subsection ();
    
   prm.enter_subsection ("Solver");
     prm.declare_entry ("Epsilon for NS" , "0.0" , Patterns::Double(0, 100000));
     prm.declare_entry ("Epsilon for C" , "0.0" , Patterns::Double(0, 100000));
     prm.declare_entry ("Krylov Size", "1", Patterns::Integer(0,1000000));
   prm.leave_subsection ();   
 }
 
 template <int dim>
 void 
 UBC_mis_mixing<dim>::Parameters::
 parse_parameters (ParameterHandler &prm)
 {
   prm.enter_subsection ("Mesh Information");
     input_mesh_file = prm.get ("Input File Name");
     domain_boundary [0] = prm.get_double ("Inlet Boundary");
     domain_boundary [1] = prm.get_double ("Outlet Boundary");
     is_symmetry_boundary = prm.get_bool ("Symmetry Boundary");
     number_slices_coarse_mesh = prm.get_integer ("Number of Slices");
   prm.leave_subsection ();
   
   prm.enter_subsection ("Adaptive Mesh Refinement");
     max_grid_level = prm.get_integer ("Maximum Refinement Level");
     type_adaptivity_rule = prm.get_integer ("Type Adaptivity Rule");
     error_threshold = prm.get_double ("Error Threshold");
     ref_crit = prm.get_double ("Refinement Threshold");
     coar_crit = prm.get_double ("Coarsening Threshold");
     intial_ratio_refinement = prm.get_integer ("Initial Ratio");
     no_refine_period = prm.get_integer ("Refinement Period");
   prm.leave_subsection ();
   
   prm.enter_subsection ("Entropy Viscosity");
     stabilization_alpha = prm.get_double ("Stablization Alpha");
     stabilization_beta = prm.get_double ("Stablization Beta");
     stabilization_c_R = prm.get_double ("c_R Factor");
   prm.leave_subsection ();

   prm.enter_subsection ("Output");
     data_id = prm.get_integer("Data Id");
     output_fac_vtu = prm.get_integer ("Output Period for VTU");
     output_fac_data = prm.get_integer ("Output Period for Data");
   prm.leave_subsection ();
   
   prm.enter_subsection ("Restart");
     is_restart = prm.get_bool ("Restart");
     index_for_restart = prm.get_integer ("Last Index for VTU");
     restart_no_timestep = prm.get_integer ("Last Time Step Number");
   prm.leave_subsection ();
   
   prm.enter_subsection ("Solver");
     eps_ns = prm.get_double ("Epsilon for NS");
     eps_c = prm.get_double ("Epsilon for C");
     kry_size = prm.get_integer ("Krylov Size");
   prm.leave_subsection ();
   
   prm.enter_subsection ("Solve Algorithm");
     ist_optimization_method = prm.get_integer ("Optimization Method");
     ist_projection_method=prm.get_integer ("Projection Method");
     ist_pressure_boundary=prm.get_integer ("Pressure Boundary");
     coeff_relax_div_velocity = prm.get_double ("Relaxation for Div-Vel");
     coeff_gamma_grad_div = prm.get_double ("Gamma for Grad-Div");
     coeff_arti_viscosity = prm.get_double ("Artificial Viscosity");
     maximum_coeff_arti_viscosity = prm.get_double ("Max Artificial Viscosity");
     exclude_depth_direction = prm.get_bool ("Exclude Depth Convection");
   prm.leave_subsection ();
   
   prm.enter_subsection ("Parameters");
     is_verbal_output = prm.get_bool ("Verbal Output");
     CFL_number = prm.get_double ("CFL Number");
     init_sep_x = prm.get_double ("Initial Separation");
     inclined_angle = prm.get_double ("Inclined Angle");
     Atwood_number = prm.get_double ("Atwood Number");
     mean_velocity_inlet = prm.get_double ("Mean Flow Velocity");
     degree_of_velocity = prm.get_integer ("Degree of Velocity");
     degree_of_pressure = prm.get_integer ("Degree of Pressure");
     degree_of_concentr = prm.get_integer ("Degree of Concentration");
   prm.leave_subsection ();
   
   prm.enter_subsection ("Constitutive Model");
     mean_viscosity  = prm.get_double ("Mean Viscosity");
     ratio_pow_law   = prm.get_double ("Coefficient for Pow-Law Fluid");
     n_pow_law       = prm.get_double ("Multiplier for Pow-Law Fluid");
   prm.leave_subsection ();

   if (dim == 2) {flow_direction  = 0; latitude_direction = 1;}
   if (dim == 3) {depth_direction = 0; latitude_direction = 1; flow_direction = 2;}
  
   Reynolds_number = mean_velocity_inlet *
                     (EquationData::pipe_diameter/
                      EquationData::kinematic_viscosity);
   Froude_number   = mean_velocity_inlet/std::sqrt(std::abs(Atwood_number)*
                     EquationData::gravitiy_accelation*EquationData::pipe_diameter);

   is_density_stable_flow = false;
   if (Atwood_number < 0.0) 
   {
     is_density_stable_flow = true; 
     Atwood_number = std::abs(Atwood_number);
   }

   double incline_value = numbers::PI*(inclined_angle/180.00000);
   double incl_vec_axial = +std::cos(incline_value);
   double incl_vec_tranv = -std::sin(incline_value);
    
   if (dim == 2)
   {
     inclined_angle_vector[0] = incl_vec_axial; 
     inclined_angle_vector[1] = incl_vec_tranv; 
   } else if (dim == 3)
   {
     inclined_angle_vector[0] = 0.0; 
     inclined_angle_vector[1] = incl_vec_tranv;  
     inclined_angle_vector[2] = incl_vec_axial;
   }
    
   output_fac_vtu = output_fac_vtu*std::pow(2.0, static_cast<double>(max_grid_level));
   output_fac_data = output_fac_data*std::pow(2.0, static_cast<double>(max_grid_level));
    
   reference_length = EquationData::pipe_diameter;
   reference_velocity = mean_velocity_inlet;
   double inertial_vel_scale = std::sqrt(std::abs(Atwood_number)*
                               EquationData::gravitiy_accelation*EquationData::pipe_diameter);
   double viscous_vel_scale = Atwood_number*EquationData::gravitiy_accelation*
                              EquationData::pipe_diameter*EquationData::pipe_diameter
                              /EquationData::kinematic_viscosity;

   if (std::abs(mean_velocity_inlet)<1e-4)
     reference_velocity = inertial_vel_scale;
    
   reference_time = reference_length/reference_velocity;
   
 }
#endif
