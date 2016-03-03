#ifndef __assembly_copydata_h__
#define __assembly_copydata_h__

#include "include.h"

  namespace Assembly
  {
    namespace Scratch
    {
      //------------ Scratch::Diffusion ---------------------------
      
      template <int dim>
      struct diffusion_step
      {
        diffusion_step (const FiniteElement<dim>  &fe_velocity,
                        const Mapping<dim>        &velocity_mapping,
                        const Quadrature<dim>     &quadrature,
                        const UpdateFlags         velocity_update_flags,
                        const FiniteElement<dim>  &fe_pressure,
                        const Mapping<dim>        &pressure_mapping,
                        const UpdateFlags         pressure_update_flags,
                        const FiniteElement<dim>  &concentr_fe,
                        const Mapping<dim>        &concentr_mapping,
                        const UpdateFlags         concentr_update_flags);

        diffusion_step (const diffusion_step &data);

        FEValues<dim>    fe_velocity_values;
        FEValues<dim>    fe_pressure_values;
        FEValues<dim>    concentr_fe_values;

        std::vector<Tensor<2,dim> >           grads_phi_u;
        std::vector<SymmetricTensor<2,dim> >  symm_grads_phi_u;
        std::vector<Tensor<1,dim> >           phi_u;
        std::vector<Tensor<1,dim> >           divergence_phi_u;
 
        std::vector<Tensor<1,dim> >   vel_star_values;
        std::vector<Tensor<1,dim> >   vel_n_values;
        std::vector<Tensor<1,dim> >   vel_n_minus_1_values;

        std::vector<Tensor<2,dim> >   grad_vel_star_values;
        std::vector<Tensor<1,dim> >   laplacian_vel_star_values;

        std::vector<Tensor<1,dim> >   grad_aux_n_values;
        std::vector<Tensor<1,dim> >   grad_aux_n_minus_1_values;
        std::vector<Tensor<1,dim> >   grad_pre_n_values;

        std::vector<Tensor<2,dim> >   grad_grad_aux_n_values;
        std::vector<Tensor<2,dim> >   grad_grad_aux_n_minus_1_values;
        std::vector<Tensor<2,dim> >   grad_grad_pre_n_values;
       
        std::vector<double>           aux_n_values;
        std::vector<double>           aux_n_minus_1_values;
        std::vector<double>           pre_n_values;
       
        std::vector<double>           concentr_values;
    
      };  
      
      template <int dim>
      diffusion_step<dim>::
        diffusion_step (const FiniteElement<dim>  &fe_velocity,
                        const Mapping<dim>        &velocity_mapping,
                        const Quadrature<dim>     &quadrature,
                        const UpdateFlags         velocity_update_flags,
                        const FiniteElement<dim>  &fe_pressure,
                        const Mapping<dim>        &pressure_mapping,
                        const UpdateFlags         pressure_update_flags,
                        const FiniteElement<dim>  &concentr_fe,
                        const Mapping<dim>        &concentr_mapping,
                        const UpdateFlags         concentr_update_flags)
        :
        fe_velocity_values (velocity_mapping,
                            fe_velocity,
                            quadrature,
                            velocity_update_flags),

        fe_pressure_values (pressure_mapping,
                            fe_pressure,
                            quadrature,
                            pressure_update_flags),

        concentr_fe_values (concentr_mapping,
                            concentr_fe,
                            quadrature,
                            concentr_update_flags),

        grads_phi_u          (fe_velocity.dofs_per_cell),
        symm_grads_phi_u     (fe_velocity.dofs_per_cell),
        phi_u                (fe_velocity.dofs_per_cell),
        divergence_phi_u     (fe_velocity.dofs_per_cell),

        vel_star_values      (quadrature.size()),
        vel_n_values         (quadrature.size()),
        vel_n_minus_1_values (quadrature.size()),

        grad_vel_star_values (quadrature.size()),
        laplacian_vel_star_values (quadrature.size()),
     
        grad_aux_n_values         (quadrature.size()),
        grad_aux_n_minus_1_values (quadrature.size()),
        grad_pre_n_values         (quadrature.size()),
 
        grad_grad_aux_n_values         (quadrature.size()),
        grad_grad_aux_n_minus_1_values (quadrature.size()),
        grad_grad_pre_n_values         (quadrature.size()),
     
        aux_n_values              (quadrature.size()),
        aux_n_minus_1_values      (quadrature.size()),
        pre_n_values              (quadrature.size()),
     
        concentr_values           (quadrature.size())
      {}
      
      template <int dim>
      diffusion_step<dim>::
      diffusion_step (const diffusion_step &scratch)
        :
        fe_velocity_values (scratch.fe_velocity_values.get_mapping(),
                            scratch.fe_velocity_values.get_fe(),
                            scratch.fe_velocity_values.get_quadrature(),
                            scratch.fe_velocity_values.get_update_flags()),

        fe_pressure_values (scratch.fe_pressure_values.get_mapping(),
                            scratch.fe_pressure_values.get_fe(),
                            scratch.fe_pressure_values.get_quadrature(),
                            scratch.fe_pressure_values.get_update_flags()),

        concentr_fe_values (scratch.concentr_fe_values.get_mapping(),
                            scratch.concentr_fe_values.get_fe(),
                            scratch.concentr_fe_values.get_quadrature(),
                            scratch.concentr_fe_values.get_update_flags()),

        grads_phi_u               (scratch.grads_phi_u),
        symm_grads_phi_u          (scratch.symm_grads_phi_u),
        phi_u                     (scratch.phi_u),
        divergence_phi_u          (scratch.divergence_phi_u),

        vel_star_values           (scratch.vel_star_values),
        vel_n_values              (scratch.vel_n_values),
        vel_n_minus_1_values      (scratch.vel_n_minus_1_values),

        grad_vel_star_values      (scratch.grad_vel_star_values),
        laplacian_vel_star_values (scratch.laplacian_vel_star_values),
     
        grad_aux_n_values         (scratch.grad_aux_n_values),
        grad_aux_n_minus_1_values (scratch.grad_aux_n_minus_1_values),
        grad_pre_n_values         (scratch.grad_pre_n_values),

        grad_grad_aux_n_values         (scratch.grad_grad_aux_n_values),
        grad_grad_aux_n_minus_1_values (scratch.grad_grad_aux_n_minus_1_values),
        grad_grad_pre_n_values         (scratch.grad_grad_pre_n_values),
     
        aux_n_values              (scratch.aux_n_values),
        aux_n_minus_1_values      (scratch.aux_n_minus_1_values),
        pre_n_values              (scratch.pre_n_values),
     
        concentr_values           (scratch.concentr_values)
      {}

      //------------ Scratch::relaxation_div_velocity_step ---------------------------

      template <int dim>
      struct relaxation_div_velocity_step
      {
        relaxation_div_velocity_step (
                 const FiniteElement<dim>  &fe_auxilary,
                 const Mapping<dim>        &auxilary_mapping,
                 const Quadrature<dim>     &quadrature,
                 const UpdateFlags         auxilary_update_flags,
                 const FiniteElement<dim>  &fe_velocity,
                 const Mapping<dim>        &velocity_mapping,
                 const UpdateFlags         velocity_update_flags,
                 const FiniteElement<dim>  &concentr_fe,
                 const Mapping<dim>        &concentr_mapping,
                 const UpdateFlags         concentr_update_flags);

        relaxation_div_velocity_step (const relaxation_div_velocity_step &data);

        FEValues<dim>   fe_auxilary_values;
        FEValues<dim>   fe_velocity_values;
        FEValues<dim>   concentr_fe_values;

        std::vector<Tensor<1,dim> >  grads_phi_p;
        std::vector<double>          phi_p;
        std::vector<Tensor<2,dim> >  grad_vel_n_plus_1_values;

        std::vector<double>          concentr_values;
      };

      template <int dim>
      relaxation_div_velocity_step<dim>::
        relaxation_div_velocity_step (
                 const FiniteElement<dim>  &fe_auxilary,
                 const Mapping<dim>        &auxilary_mapping,
                 const Quadrature<dim>     &quadrature,
                 const UpdateFlags         auxilary_update_flags,
                 const FiniteElement<dim>  &fe_velocity,
                 const Mapping<dim>        &velocity_mapping,
                 const UpdateFlags         velocity_update_flags,
                 const FiniteElement<dim>  &concentr_fe,
                 const Mapping<dim>        &concentr_mapping,
                 const UpdateFlags         concentr_update_flags)
        :
        fe_auxilary_values (auxilary_mapping,
                            fe_auxilary,
                            quadrature,
                            auxilary_update_flags),

        fe_velocity_values (velocity_mapping,
                            fe_velocity,
                            quadrature,
                            velocity_update_flags),

        concentr_fe_values (concentr_mapping,
                            concentr_fe,
                            quadrature,
                            concentr_update_flags),

        grads_phi_p              (fe_auxilary.dofs_per_cell),
        phi_p                    (fe_auxilary.dofs_per_cell),
        grad_vel_n_plus_1_values (quadrature.size()),
        concentr_values          (quadrature.size())
      {}

      template <int dim>
      relaxation_div_velocity_step<dim>::
      relaxation_div_velocity_step (const relaxation_div_velocity_step &scratch)
        :
        fe_auxilary_values (scratch.fe_auxilary_values.get_mapping(),
                            scratch.fe_auxilary_values.get_fe(),
                            scratch.fe_auxilary_values.get_quadrature(),
                            scratch.fe_auxilary_values.get_update_flags()),

        fe_velocity_values (scratch.fe_velocity_values.get_mapping(),
                            scratch.fe_velocity_values.get_fe(),
                            scratch.fe_velocity_values.get_quadrature(),
                            scratch.fe_velocity_values.get_update_flags()),

        concentr_fe_values (scratch.concentr_fe_values.get_mapping(),
                            scratch.concentr_fe_values.get_fe(),
                            scratch.concentr_fe_values.get_quadrature(),
                            scratch.concentr_fe_values.get_update_flags()),

        grads_phi_p               (scratch.grads_phi_p),
        phi_p                     (scratch.phi_p),
        grad_vel_n_plus_1_values  (scratch.grad_vel_n_plus_1_values),
        concentr_values           (scratch.concentr_values)
      {}

      //------------ Scratch::Projection ---------------------------
      
      template <int dim>
      struct projection_step
      {
        projection_step (
                 const FiniteElement<dim>  &fe_auxilary,
                 const Mapping<dim>        &auxilary_mapping,
                 const Quadrature<dim>     &quadrature,
                 const UpdateFlags         auxilary_update_flags,
                 const FiniteElement<dim>  &fe_velocity,
                 const Mapping<dim>        &velocity_mapping,
                 const UpdateFlags         velocity_update_flags,
                 const FiniteElement<dim>  &concentr_fe,
                 const Mapping<dim>        &concentr_mapping,
                 const UpdateFlags         concentr_update_flags);

        projection_step (const projection_step &data);

        FEValues<dim>   fe_auxilary_values;
        FEValues<dim>   fe_velocity_values;
        FEValues<dim>   concentr_fe_values;

        std::vector<Tensor<1,dim> >  grads_phi_p;
        std::vector<double>          phi_p;
        std::vector<Tensor<2,dim> >  grad_vel_n_plus_1_values;

        std::vector<double>          concentr_values;
        std::vector<double>          div_vel_values;
      };  
      
      template <int dim>
      projection_step<dim>::
        projection_step (
                 const FiniteElement<dim>  &fe_auxilary,
                 const Mapping<dim>        &auxilary_mapping,
                 const Quadrature<dim>     &quadrature,
                 const UpdateFlags         auxilary_update_flags,
                 const FiniteElement<dim>  &fe_velocity,
                 const Mapping<dim>        &velocity_mapping,
                 const UpdateFlags         velocity_update_flags,
                 const FiniteElement<dim>  &concentr_fe,
                 const Mapping<dim>        &concentr_mapping,
                 const UpdateFlags         concentr_update_flags)
        :
        fe_auxilary_values (auxilary_mapping,
                            fe_auxilary,
                            quadrature,
                            auxilary_update_flags),

        fe_velocity_values (velocity_mapping,
                            fe_velocity,
                            quadrature,
                            velocity_update_flags),

        concentr_fe_values (concentr_mapping,
                            concentr_fe,
                            quadrature,
                            concentr_update_flags),

        grads_phi_p              (fe_auxilary.dofs_per_cell),
        phi_p                    (fe_auxilary.dofs_per_cell),
        grad_vel_n_plus_1_values (quadrature.size()),
        concentr_values          (quadrature.size()),
        div_vel_values           (quadrature.size())
      {}
      
      template <int dim>
      projection_step<dim>::
      projection_step (const projection_step &scratch)
        :
        fe_auxilary_values (scratch.fe_auxilary_values.get_mapping(),
                            scratch.fe_auxilary_values.get_fe(),
                            scratch.fe_auxilary_values.get_quadrature(),
                            scratch.fe_auxilary_values.get_update_flags()),

        fe_velocity_values (scratch.fe_velocity_values.get_mapping(),
                            scratch.fe_velocity_values.get_fe(),
                            scratch.fe_velocity_values.get_quadrature(),
                            scratch.fe_velocity_values.get_update_flags()),

        concentr_fe_values (scratch.concentr_fe_values.get_mapping(),
                            scratch.concentr_fe_values.get_fe(),
                            scratch.concentr_fe_values.get_quadrature(),
                            scratch.concentr_fe_values.get_update_flags()),

        grads_phi_p               (scratch.grads_phi_p),
        phi_p                     (scratch.phi_p),
        grad_vel_n_plus_1_values  (scratch.grad_vel_n_plus_1_values),
        concentr_values           (scratch.concentr_values),
        div_vel_values            (scratch.div_vel_values)
      {}
      
      //------------ Scratch::Pressure Correction With Rotation ---------------------------
      
      template <int dim>
      struct pressure_rot_step
      {
        pressure_rot_step (const FiniteElement<dim>  &fe_pressure,
                           const Mapping<dim>        &pressure_mapping,
                           const Quadrature<dim>     &quadrature,
                           const UpdateFlags         pressure_update_flags,
                           const FiniteElement<dim>  &fe_velocity,
                           const Mapping<dim>        &velocity_mapping,
                           const UpdateFlags         velocity_update_flags,
                           const FiniteElement<dim>  &concentr_fe,
                           const Mapping<dim>        &concentr_mapping,
                           const UpdateFlags         concentr_update_flags);

        pressure_rot_step (const pressure_rot_step &data);

        FEValues<dim>    fe_pressure_values;
        FEValues<dim>    fe_velocity_values;
        FEValues<dim>    concentr_fe_values;

        std::vector<double>         phi_p;
        std::vector<double>         aux_sol_values;
        std::vector<double>         pre_sol_values;
        std::vector<Tensor<2,dim> > grad_vel_sol_values;
        std::vector<double>         concentr_values;
      };  
      
      template <int dim>
      pressure_rot_step<dim>::
        pressure_rot_step (const FiniteElement<dim>  &fe_pressure,
                           const Mapping<dim>        &pressure_mapping,
                           const Quadrature<dim>     &quadrature,
                           const UpdateFlags         pressure_update_flags,
                           const FiniteElement<dim>  &fe_velocity,
                           const Mapping<dim>        &velocity_mapping,
                           const UpdateFlags         velocity_update_flags,
                           const FiniteElement<dim>  &concentr_fe,
                           const Mapping<dim>        &concentr_mapping,
                           const UpdateFlags         concentr_update_flags)
        :
        fe_pressure_values (pressure_mapping, 
                            fe_pressure,
                            quadrature,
                            pressure_update_flags),

        fe_velocity_values (velocity_mapping, 
                            fe_velocity,
                            quadrature,
                            velocity_update_flags),

        concentr_fe_values (concentr_mapping,
                            concentr_fe,
                            quadrature,
                            concentr_update_flags),

        phi_p               (fe_pressure.dofs_per_cell),
        aux_sol_values      (quadrature.size()),
        pre_sol_values      (quadrature.size()),
        grad_vel_sol_values (quadrature.size()),
        concentr_values     (quadrature.size())
      {}
      
      template <int dim>
      pressure_rot_step<dim>::
      pressure_rot_step (const pressure_rot_step &scratch)
        :
        fe_pressure_values (scratch.fe_pressure_values.get_mapping(),
                            scratch.fe_pressure_values.get_fe(),
                            scratch.fe_pressure_values.get_quadrature(),
                            scratch.fe_pressure_values.get_update_flags()),

        fe_velocity_values (scratch.fe_velocity_values.get_mapping(),
                            scratch.fe_velocity_values.get_fe(),
                            scratch.fe_velocity_values.get_quadrature(),
                            scratch.fe_velocity_values.get_update_flags()),

        concentr_fe_values (scratch.concentr_fe_values.get_mapping(),
                            scratch.concentr_fe_values.get_fe(),
                            scratch.concentr_fe_values.get_quadrature(),
                            scratch.concentr_fe_values.get_update_flags()),

        phi_p               (scratch.phi_p),
        aux_sol_values      (scratch.aux_sol_values),
        pre_sol_values      (scratch.pre_sol_values),
        grad_vel_sol_values (scratch.grad_vel_sol_values),
        concentr_values     (scratch.concentr_values)
      {}
      
      //------------Scratch::Concentration Matrix ---------------------------
      
      template <int dim>
      struct concentrMatrix
      {
        concentrMatrix (const FiniteElement<dim>  &concentr_fe,
                        const Mapping<dim>        &mapping,
                        const Quadrature<dim>     &concentr_quadrature);

        concentrMatrix (const concentrMatrix &data);


        FEValues<dim>               concentr_fe_values;

        std::vector<double>         phi_T;
        std::vector<Tensor<1,dim> > grad_phi_T;
      };


      template <int dim>
      concentrMatrix<dim>::
      concentrMatrix (const FiniteElement<dim> &concentr_fe,
                      const Mapping<dim>       &mapping,
        const Quadrature<dim>    &concentr_quadrature)
        :
        concentr_fe_values (mapping,
                            concentr_fe, concentr_quadrature,
                            update_values    |
                            update_gradients |
                            update_JxW_values),
        phi_T       (concentr_fe.dofs_per_cell),
        grad_phi_T  (concentr_fe.dofs_per_cell)
      {}

      template <int dim>
      concentrMatrix<dim>::
      concentrMatrix (const concentrMatrix &scratch)
        :
        concentr_fe_values (scratch.concentr_fe_values.get_mapping(),
                            scratch.concentr_fe_values.get_fe(),
        scratch.concentr_fe_values.get_quadrature(),
        scratch.concentr_fe_values.get_update_flags()),
        phi_T      (scratch.phi_T),
        grad_phi_T (scratch.grad_phi_T)
      {}
      
      //------------Scratch::Concentration RHS ---------------------------
      
      template <int dim>
      struct concentrRHS
      {
        concentrRHS (const FiniteElement<dim> &concentr_fe,
                     const FiniteElement<dim> &fe_velocity,
                     const Mapping<dim>       &mapping,
                     const Quadrature<dim>    &quadrature);

        concentrRHS (const concentrRHS &data);


        FEValues<dim>                        concentr_fe_values;
        FEValues<dim>                        fe_velocity_values;

        std::vector<double>                  phi_T;
        std::vector<Tensor<1,dim> >          grad_phi_T;

        std::vector<Tensor<1,dim> >          old_velocity_values;
        std::vector<Tensor<1,dim> >          old_old_velocity_values;

        std::vector<SymmetricTensor<2,dim> > old_strain_rates;
        std::vector<SymmetricTensor<2,dim> > old_old_strain_rates;

        std::vector<double>                  old_concentr_values;
        std::vector<double>                  old_old_concentr_values;
        std::vector<Tensor<1,dim> >          old_concentr_grads;
        std::vector<Tensor<1,dim> >          old_old_concentr_grads;
        std::vector<double>                  old_concentr_laplacians;
        std::vector<double>                  old_old_concentr_laplacians;
      };


      template <int dim>
      concentrRHS<dim>::
      concentrRHS (const FiniteElement<dim> &concentr_fe,
                   const FiniteElement<dim> &fe_velocity,
                   const Mapping<dim>       &mapping,
                   const Quadrature<dim>    &quadrature)
        :
        concentr_fe_values (mapping,
                            concentr_fe, quadrature,
                            update_values    |
                            update_gradients |
                            update_hessians  |
                            update_quadrature_points |
                            update_JxW_values),

        fe_velocity_values (mapping,
                            fe_velocity, quadrature,
                            update_values |
                            update_gradients),

        phi_T                      (concentr_fe.dofs_per_cell),
        grad_phi_T                 (concentr_fe.dofs_per_cell),

        old_velocity_values        (quadrature.size()),
        old_old_velocity_values    (quadrature.size()),
        old_strain_rates           (quadrature.size()),
        old_old_strain_rates       (quadrature.size()),

        old_concentr_values        (quadrature.size()),
        old_old_concentr_values    (quadrature.size()),
        old_concentr_grads         (quadrature.size()),
        old_old_concentr_grads     (quadrature.size()),
        old_concentr_laplacians    (quadrature.size()),
        old_old_concentr_laplacians(quadrature.size())
      {}


      template <int dim>
      concentrRHS<dim>::
      concentrRHS (const concentrRHS &scratch)
        :
        concentr_fe_values (scratch.concentr_fe_values.get_mapping(),
                            scratch.concentr_fe_values.get_fe(),
                            scratch.concentr_fe_values.get_quadrature(),
                            scratch.concentr_fe_values.get_update_flags()),
        fe_velocity_values (scratch.fe_velocity_values.get_mapping(),
                            scratch.fe_velocity_values.get_fe(),
                            scratch.fe_velocity_values.get_quadrature(),
                            scratch.fe_velocity_values.get_update_flags()),
        phi_T (scratch.phi_T),
        grad_phi_T (scratch.grad_phi_T),

        old_velocity_values         (scratch.old_velocity_values),
        old_old_velocity_values     (scratch.old_old_velocity_values),
        old_strain_rates            (scratch.old_strain_rates),
        old_old_strain_rates        (scratch.old_old_strain_rates),

        old_concentr_values         (scratch.old_concentr_values),
        old_old_concentr_values     (scratch.old_old_concentr_values),
        old_concentr_grads          (scratch.old_concentr_grads),
        old_old_concentr_grads      (scratch.old_old_concentr_grads),
        old_concentr_laplacians     (scratch.old_concentr_laplacians),
        old_old_concentr_laplacians (scratch.old_old_concentr_laplacians)
      {}
      
    } //Assembly-Scratch
    
    namespace CopyData
    {

      //------------ CopyData::Diffusion ---------------------------
      
      template <int dim>
      struct diffusion_step
      {
        diffusion_step (const FiniteElement<dim> &fe_velocity);
        diffusion_step (const diffusion_step &data);

        FullMatrix<double>                    local_matrix;
        Vector<double>                        local_rhs;
        std::vector<types::global_dof_index>  local_dof_indices;
      };

      template <int dim>
      diffusion_step<dim>::
      diffusion_step (const FiniteElement<dim> &fe_velocity)
        :
        local_matrix (fe_velocity.dofs_per_cell,
                             fe_velocity.dofs_per_cell),
        local_rhs (fe_velocity.dofs_per_cell),
        local_dof_indices (fe_velocity.dofs_per_cell)
      {}

      template <int dim>
      diffusion_step<dim>::
      diffusion_step (const diffusion_step &data)
        :
        local_matrix (data.local_matrix),
        local_rhs (data.local_rhs),
        local_dof_indices (data.local_dof_indices)
      {} 
  
      //------------ CopyData::relaxation_div_velocity_step ---------------------------
      
      template <int dim>
      struct relaxation_div_velocity_step
      {
        relaxation_div_velocity_step (const FiniteElement<dim> &fe_auxilary);
        relaxation_div_velocity_step (const relaxation_div_velocity_step &data);

        FullMatrix<double>                    local_matrix;
        Vector<double>                        local_rhs;
        std::vector<types::global_dof_index>  local_dof_indices;
      };

      template <int dim>
      relaxation_div_velocity_step<dim>::
      relaxation_div_velocity_step (const FiniteElement<dim> &fe_auxilary)
        :
        local_matrix (fe_auxilary.dofs_per_cell,
                          fe_auxilary.dofs_per_cell),
        local_rhs  (fe_auxilary.dofs_per_cell),
        local_dof_indices (fe_auxilary.dofs_per_cell)
      {}

      template <int dim>
      relaxation_div_velocity_step<dim>::
      relaxation_div_velocity_step (const relaxation_div_velocity_step &data)
        :
        local_matrix   (data.local_matrix),
        local_rhs   (data.local_rhs),
        local_dof_indices (data.local_dof_indices)
      {}
     
      //------------ CopyData::Projection ---------------------------
      
      template <int dim>
      struct projection_step
      {
        projection_step (const FiniteElement<dim> &fe_auxilary);
        projection_step (const projection_step &data);

        FullMatrix<double>                    local_matrix;
        Vector<double>                        local_rhs;
        std::vector<types::global_dof_index>  local_dof_indices;
      };

      template <int dim>
      projection_step<dim>::
      projection_step (const FiniteElement<dim> &fe_auxilary)
        :
       local_matrix       (fe_auxilary.dofs_per_cell,
                           fe_auxilary.dofs_per_cell),
       local_rhs          (fe_auxilary.dofs_per_cell),
       local_dof_indices  (fe_auxilary.dofs_per_cell)
      {}

      template <int dim>
      projection_step<dim>::
      projection_step (const projection_step &data)
        :
       local_matrix       (data.local_matrix),
       local_rhs          (data.local_rhs),
       local_dof_indices  (data.local_dof_indices)
      {}       
      
      //------------ CopyData::Pressure Correction With Rotation ---------------------------
      
      template <int dim>
      struct pressure_rot_step
      {
        pressure_rot_step (const FiniteElement<dim> &fe_pressure);
        pressure_rot_step (const pressure_rot_step &data);

        FullMatrix<double>                    local_matrix;
        Vector<double>                        local_rhs;
        std::vector<types::global_dof_index>  local_dof_indices;
      };

      template <int dim>
      pressure_rot_step<dim>::
      pressure_rot_step (const FiniteElement<dim> &fe_pressure)
        :
        local_matrix       (fe_pressure.dofs_per_cell,
                            fe_pressure.dofs_per_cell),
        local_rhs          (fe_pressure.dofs_per_cell),
        local_dof_indices  (fe_pressure.dofs_per_cell)
      {}

      template <int dim>
      pressure_rot_step<dim>::
      pressure_rot_step (const pressure_rot_step &data)
        :
        local_matrix       (data.local_matrix),
        local_rhs          (data.local_rhs),
        local_dof_indices  (data.local_dof_indices)
      {}   
      
      //------------ CopyData::concentrMatrix ---------------------------

      template <int dim>
      struct concentrMatrix
      {
        concentrMatrix (const FiniteElement<dim> &concentr_fe);
        concentrMatrix (const concentrMatrix &data);

        FullMatrix<double>                      local_mass_matrix;
        FullMatrix<double>                      local_stiffness_matrix;
        std::vector<types::global_dof_index>    local_dof_indices;
      };

      template <int dim>
      concentrMatrix<dim>::
      concentrMatrix (const FiniteElement<dim> &concentr_fe)
        :
        local_mass_matrix (concentr_fe.dofs_per_cell,
                           concentr_fe.dofs_per_cell),
        local_stiffness_matrix (concentr_fe.dofs_per_cell,
                                concentr_fe.dofs_per_cell),
        local_dof_indices (concentr_fe.dofs_per_cell)
      {}

      template <int dim>
      concentrMatrix<dim>::
      concentrMatrix (const concentrMatrix &data)
        :
        local_mass_matrix   (data.local_mass_matrix),
        local_stiffness_matrix  (data.local_stiffness_matrix),
        local_dof_indices   (data.local_dof_indices)
      {}

      //------------ CopyData::concentrRHS ---------------------------

      template <int dim>
      struct concentrRHS
      {
        concentrRHS (const FiniteElement<dim> &concentr_fe);
        concentrRHS (const concentrRHS &data);

        Vector<double>                    local_rhs;
        std::vector<types::global_dof_index>  local_dof_indices;
        FullMatrix<double>                matrix_for_bc;
      };

      template <int dim>
      concentrRHS<dim>::
      concentrRHS (const FiniteElement<dim> &concentr_fe)
        :
        local_rhs (concentr_fe.dofs_per_cell),
        local_dof_indices (concentr_fe.dofs_per_cell),
        matrix_for_bc (concentr_fe.dofs_per_cell,
                       concentr_fe.dofs_per_cell)
      {}

      template <int dim>
      concentrRHS<dim>::
      concentrRHS (const concentrRHS &data)
        :
        local_rhs (data.local_rhs),
        local_dof_indices (data.local_dof_indices),
        matrix_for_bc (data.matrix_for_bc)
      {}
      
    } //Assembly-CopyData
       
  }// End-Assembly
#endif
