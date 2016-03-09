#ifndef __ps_mmm__immicible_levelset_h
#define __ps_mmm__immicible_levelset_h

#include <mismix/include.h>

namespace ps_mmm
{
  namespace immicible
  {
    template <int dim>
    class Levelset
    {
      public:
        Levelset (ParameterHandler &);
        ~Levelset ();

      private:
        const Mapping<dim>                                *mapping;
        const parallel::distributed::Triangulation<dim>   *triangulation;
        const DoFHandler<dim>                             *dof_handler;
        const TrilinosWrappers::MPI::Vector               *solution;

      public:
        void initial_distribution ();
        void get_parameters ();
        void advection_equation ();
        void compute_normal_vector (unsigned int);
        void reinitialization_levelset ();
    };
  }
}

#endif
