#ifndef __ps_mmm__particle_generator_h
#define __ps_mmm__particle_generator_h

#include <mismix/particle/particle.h>
#include <mismix/particle/world.h>

namespace ps_mmm
{
  namespace Particle
  {
    namespace Generator
    {
      /**
       * Abstract base class used for classes that generate particles
       */
      template <int dim, class T>
      class Interface
      {
        public:
          /**
           * Constructor.
           */
          Interface() {}

          /**
           * Destructor. Made virtual so that derived classes can be created
           * and destroyed through pointers to the base class.
           */
          virtual ~Interface () {}

          /**
           * Generate a specified number of particles in the specified world
           * using the type of generation function implemented by this
           * Generator.
           *
           * @param [in] world The particle world the particles will exist in
           * @param [in] total_num_particles Total number of particles to
           * generate. The actual number of generated particles may differ,
           * for example if the generator reads particles from a file this
           * parameter may be ignored.
           */
          virtual
          void
          generate_particles(Particle::World<dim, T> &world,
                             const double total_num_particles) = 0;
      };


      /**
       * Create a generator object.
       *
       * @param[in] generator_type Name of the type of generator to create
       * @return pointer to the generator. Caller needs to delete this
       * pointer.
       */
      template <int dim, class T>
      Interface<dim, T> *
      create_generator_object (const std::string &generator_type);


      /**
       * Return a list of names (separated by '|') of possible particle
       * generators.
       */
      std::string
      generator_object_names ();

    }
  }
}

#endif
