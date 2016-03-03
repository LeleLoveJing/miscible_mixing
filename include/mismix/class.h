#ifndef __class_h__
#define __class_h__ 

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <limits>
#include <locale> 

#include "include.h"
#include "equation_data.h"

  template <int dim>
  class UBC_mis_mixing
  {
  public :
    struct Parameters;
    UBC_mis_mixing (Parameters &parameters);
    void run ();
    
  private :
    void print_input_parameters (); 
    void create_triangulation ();
    void assgined_boundary_indicator ();
    
    void setup_dofs_velocity ();
    void setup_dofs_pressure ();
    void setup_dofs_auxilary ();
    void setup_dofs_concentr ();
    void setup_dofs_error    ();
    
    void solve_fluid_equation (double increase_dimless_vel);
    void extrapolation_step ();
    void diffusion_step (double increase_dimless_vel);
    void relaxation_div_velocity_step ();
    void projection_step ();
    void pressure_rot_step ();
    void solution_update ();
    
    void solve_concentr_equation (const double maximal_velocity);
    void assemble_concentr_matrix ();
    void assemble_concentr_system (const double maximal_velocity);
    void project_concentr_field ();
    
    double get_maximal_velocity () const;
    double get_cfl_number () const;
    double get_entropy_variation (const double average_concentr) const;
    std::pair<double,double> get_extrapolated_concentr_range () const;
    
    void output_results (unsigned int out_index);
    
    void initial_refine_mesh ();
    void loop_over_cell_error_indicator ();
    void loop_over_cell_error_indicator2 ();
    void compute_for_SymmTensorFlux ();
   
    void refine_mesh (const unsigned int max_grid_level);

    void    save_data_and_mesh ();
    void    load_data_and_mesh ();
    void    move_file (const std::string &old_name,
                       const std::string &new_name);
    void    save_snapshot_template (std::vector<const TrilinosWrappers::MPI::Vector *> &system,
                                    std::stringstream                                  &file_stream,
                                    std::stringstream                                  &file_zlib_stream,
                                    DoFHandler<dim>                                    &dof_handler_this);
    
    void    load_snapshot_template (unsigned int                                       which_variable,
                                    std::stringstream                                  &file_stream,
                                    std::stringstream                                  &file_zlib_stream);
    
    void    compute_for_avrage_quantities_surfIntgl (std::ofstream &out_avr_c,
                                                     std::ofstream &out_avr_vFlow_c,
                                                     std::ofstream &out_avr_vvFlow_c,
                                                     std::ofstream &out_avr_vvLati_c,
                                                     std::ofstream &out_avr_vvDept_c);
    
    void    compute_for_post_error (unsigned int no_type, 
                                    Vector<float> &estimated_error_per_cell);
    
    double
    compute_entropy_viscosity_for_hyperbolic(
                              const std::vector<double>             &old_concentr,
                              const std::vector<double>             &old_old_concentr,
                              const std::vector<Tensor<1,dim> >     &old_concentr_grads,
                              const std::vector<Tensor<1,dim> >     &old_old_concentr_grads,
                              const std::vector<double>             &old_concentr_laplacians,
                              const std::vector<double>             &old_old_concentr_laplacians,
                              const std::vector<Tensor<1,dim> >     &old_velocity_values,
                              const std::vector<Tensor<1,dim> >     &old_old_velocity_values,
                              const std::vector<SymmetricTensor<2,dim> > &old_strain_rates,
                              const std::vector<SymmetricTensor<2,dim> > &old_old_strain_rates,
                              const double                           global_u_infty,
                              const double                           global_T_variation,
                              const double                           average_concentr,
                              const double                           global_entropy_variation,
                              const double                           cell_diameter) const;

    std::pair<double,double>
    compute_entropy_viscosity_for_navier_stokes(
                              const std::vector<Tensor<1,dim> >     &old_velocity,
                              const std::vector<Tensor<1,dim> >     &old_old_velocity,
                              const std::vector<Tensor<2,dim> >     &old_velocity_grads,
                              const std::vector<Tensor<1,dim> >     &old_velocity_laplacians,
                              const std::vector<Tensor<1,dim> >     &old_pressure_grads,
                              const Tensor<1,dim>                   &source_vector,
                              const double                           coeff1_for_adv,
                              const double                           coeff2_for_visco,
                              const double                           coeff3_for_source,
                              const double                           cell_diameter) const;

    void compute_global_error_norm (std::ofstream &);
    
  public :
    struct Parameters
    {
      Parameters (std::string &parameters_filename);
      
      static void declare_parameters (ParameterHandler &prm);
      void parse_parameters (ParameterHandler &prm);
      
      //GeometryInfo
      std::string         input_mesh_file;
      Point<dim>          length_of_domain;

      std::vector<double> domain_size;
      std::vector<double> domain_boundary;
      unsigned int        num_element_size;

      unsigned int       flow_direction;
      unsigned int       depth_direction;
      unsigned int       latitude_direction;
   
      unsigned int       num_slices_domain;
   
      bool               is_symmetry_boundary;
   
      // Adaptive Mesh Refinement
      unsigned int       max_grid_level;
      unsigned int       type_adaptivity_rule;
      double             error_threshold;
      double             ref_crit;
      double             coar_crit;
      unsigned int       no_refine_period;
      unsigned int       intial_ratio_refinement;
   
      // Entroypy Viscosity
      double        stabilization_alpha;
      double        stabilization_beta;
      double        stabilization_c_R;
   
      // Solve Algorithm
      unsigned int       ist_optimization_method;
      unsigned int       ist_projection_method;
      unsigned int       ist_pressure_boundary;
      unsigned int       ist_flow_source;
      bool               ist_uniform_flow;
      double             coeff_relax_div_velocity;
    
      unsigned int       no_steps_for_buffering;
      double             mesh_speed;
      unsigned int       dir_concentration;
      unsigned int       which_method_for_c;
      unsigned int       which_interpl_c;
      bool               ist_add_reinit;
      double             coeff_gamma_grad_div;
      double             coeff_arti_viscosity;
      double             maximum_coeff_arti_viscosity;
      bool               exclude_depth_direction;
        
      // Parameters
      bool         is_verbal_output;
      double       CFL_number;
      double       init_sep_x;
      double       inclined_angle;
      double       Atwood_number;
      double       mean_velocity_inlet;
      double       inlet_pressure;
      double       viscosity_ratio;
      double       computed_time_step;
        
      double       Reynolds_number;
      double       Froude_number;
      double       reference_length;
      double       reference_time;
      double       reference_velocity;
      bool         is_density_stable_flow;
        
      double       upstream_concentr;
      double       downstream_concentr;
        
      double       mean_viscosity;
      double       fluid1_viscosity;
      double       fluid2_viscosity;
        
      double       k_shear_thinn;
      double       n_shear_thinn;
      bool         is_shear_thinn;
        
      Point<dim>   inclined_angle_vector;
        
      double       tau_step;
      double       eps_v_concentr;
        
      // Polynomial Order
      unsigned int       degree_of_velocity;
      unsigned int       degree_of_pressure;
      unsigned int       degree_of_concentr;
        
      // Output
      unsigned int       data_id;
      unsigned int       output_fac_vtu;
      unsigned int       output_fac_data;
     
      //Data Extract
      unsigned int       number_slices_coarse_mesh;
        
      // Restart
      bool                  is_restart;
      unsigned int          save_fac_period;
      unsigned int          index_for_restart;
      unsigned int          restart_no_timestep;
      double                check_total_time;
      double                check_total_real_time;
      double                check_current_time_step;
      double                check_old_time_step;
        
      //Solver Parameter
      double                eps_ns;
      double                eps_c;
      unsigned int          kry_size;
      
      //Debug
      unsigned int          no_test_case;
      
    }; //struct Parameters
    
  private:
    Parameters                                 &parameters;
    ConditionalOStream                         pcout;
    parallel::distributed::Triangulation<dim>  triangulation;
    double                                     global_Omega_diameter;
    
    const FESystem<dim>             fe_velocity;
    FE_Q<dim>                       fe_pressure;
    FE_Q<dim>                       fe_auxilary;
    FE_Q<dim>                       concentr_fe;
    FE_DGQ<dim>                     fe_error;
    
    DoFHandler<dim>                 dof_handler_velocity;
    DoFHandler<dim>                 dof_handler_pressure;
    DoFHandler<dim>                 dof_handler_auxilary;
    DoFHandler<dim>                 concentr_dof_handler;
    DoFHandler<dim>                 dof_handler_error;
    
    ConstraintMatrix                constraints_velocity;
    ConstraintMatrix                constraints_pressure;
    ConstraintMatrix                constraints_auxilary;
    ConstraintMatrix                concentr_constraints;
    
    TrilinosWrappers::SparseMatrix           matrix_velocity;
    TrilinosWrappers::SparseMatrix           matrix_pressure;
    TrilinosWrappers::SparseMatrix           matrix_auxilary;
    TrilinosWrappers::SparseMatrix           concentr_mass_matrix;
    TrilinosWrappers::SparseMatrix           concentr_stiffness_matrix;
    TrilinosWrappers::SparseMatrix           concentr_matrix;
    
    TrilinosWrappers::MPI::Vector            vel_star, vel_star_old, vel_n_plus_1, vel_n;
    TrilinosWrappers::MPI::Vector            vel_n_minus_1, vel_n_minus_minus_1;
    TrilinosWrappers::MPI::Vector            rhs_velocity;
     
    TrilinosWrappers::MPI::Vector            pre_star, pre_n_plus_1, pre_n, pre_n_minus_1;
    TrilinosWrappers::MPI::Vector            rhs_pressure;
    
    TrilinosWrappers::MPI::Vector            aux_n_plus_1, aux_n, aux_n_minus_1;
    TrilinosWrappers::MPI::Vector            rhs_auxilary;
     
    TrilinosWrappers::MPI::Vector            concentr_solution;
    TrilinosWrappers::MPI::Vector            old_concentr_solution;
    TrilinosWrappers::MPI::Vector            old_old_concentr_solution;
    TrilinosWrappers::MPI::Vector            concentr_rhs;
    TrilinosWrappers::MPI::Vector            post_error_crit1, post_error_crit2;
    
    TrilinosWrappers::MPI::Vector            entropy_viscosity_for_ns;
    TrilinosWrappers::MPI::Vector            energy_norm_for_ns;

    double                         time_step;
    double                         old_time_step;
    double                         computed_time_step;
    unsigned int                   timestep_number;
    double                         dimless_vel_inlet;
    double                         min_h_size;

    double                         max_entropy_viscosity;
    std::pair<double, double>      mixing_min_max;
    std_cxx1x::shared_ptr<TrilinosWrappers::PreconditionJacobi>
                                   T_preconditioner;

    bool                           rebuild_concentr_matrices;
    bool                           rebuild_concentr_preconditioner;

    TimerOutput                    computing_timer;

    void setup_matrix_velocity (const IndexSet &velocity_partitioning);
    void setup_matrix_pressure (const IndexSet &pressure_partitioning);
    void setup_matrix_auxilary (const IndexSet &auxilary_partitioning);
    void setup_concentr_matrices (const IndexSet &concentr_partitioning);
    
    //------------ Diffusion ---------------------------
    void
      local_assemble_diffusion_step 
      (const typename DoFHandler<dim>::active_cell_iterator &cell,
       Assembly::Scratch::diffusion_step <dim> &scratch,
       Assembly::CopyData::diffusion_step <dim> &data);
      
    void
      copy_local_to_global_diffusion_step (const Assembly::CopyData::diffusion_step<dim> &data);
   
    //------------ Relaxation Div-Vel ---------------------------
    void
      local_assemble_relaxation_div_velocity_step
      (const typename DoFHandler<dim>::active_cell_iterator &cell,
       Assembly::Scratch::relaxation_div_velocity_step <dim> &scratch,
       Assembly::CopyData::relaxation_div_velocity_step <dim> &data);
      
    void
      copy_local_to_global_relaxation_div_velocity_step
      (const Assembly::CopyData::relaxation_div_velocity_step<dim> &data);
   
    //------------ Projection ---------------------------
    void
      local_assemble_projection_step 
      (const typename DoFHandler<dim>::active_cell_iterator &cell,
       Assembly::Scratch::projection_step <dim> &scratch,
       Assembly::CopyData::projection_step <dim> &data);
      
    void
      copy_local_to_global_projection_step 
      (const Assembly::CopyData::projection_step<dim> &data);
      
    //------------ Pressure Correction With Rotation --------------------------- 
    void
      local_assemble_pressure_rot_step
      (const typename DoFHandler<dim>::active_cell_iterator &cell,
       Assembly::Scratch::pressure_rot_step <dim> &scratch,
       Assembly::CopyData::pressure_rot_step <dim> &data);
      
    void
      copy_local_to_global_pressure_rot_step 
      (const Assembly::CopyData::pressure_rot_step<dim> &data);
    
  //------------ Solve concentr ---------------------------
   
    void
    local_assemble_concentr_matrix (const typename DoFHandler<dim>::active_cell_iterator &cell,
                                    Assembly::Scratch::concentrMatrix<dim>  &scratch,
                                    Assembly::CopyData::concentrMatrix<dim> &data);

    void
    copy_local_to_global_concentr_matrix (const Assembly::CopyData::concentrMatrix<dim> &data);

    void
    local_assemble_concentr_rhs (const std::pair<double,double> global_T_range,
                                 const double                   global_max_velocity,
                                 const double                   global_entropy_variation,
                                 const typename DoFHandler<dim>::active_cell_iterator &cell,
                                 Assembly::Scratch::concentrRHS<dim> &scratch,
                                 Assembly::CopyData::concentrRHS<dim> &data);

    void
    copy_local_to_global_concentr_rhs (const Assembly::CopyData::concentrRHS<dim> &data);
    
    class Postprocessor;
    
  }; //End_Class_Declare Part
#endif
