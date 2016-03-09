#include <mismix/phase/levelset.h>

namespace ps_mmm
{
  namespace immicible
  {

    template <int dim>
    void Levelset<dim>::initial_distribution ()
    {}

    template <int dim>
    void Levelset<dim>::get_parameters ()
    {}

    template <int dim>
    void Levelset<dim>::advection_equation ()
    {}

    template <int dim>
    void Levelset<dim>::compute_normal_vector (unsigned int current_dimension)
    {}

    template <int dim>
    void Levelset<dim>::reinitialization_levelset ()
    {}

    // explicit instantiation
    template class Levelset<2>;
    template class Levelset<3>;

  }
}
