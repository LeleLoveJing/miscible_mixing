#ifndef __equation_data_h__
#define __equation_data_h__

#include "include.h"

namespace EquationData
  {
    const double pipe_diameter = 19.05; /* mm */
    const double gravitiy_accelation = 9800; /* mm/s^2 */
    const double upstream_concentr = 0.0;
    const double downstream_concentr = 1.0;
    const double kinematic_viscosity = 1.0; /* mm/s^2 */
      
    template <int dim>
    class Inflow_Velocity : public Function<dim>
    {

     public:

     Inflow_Velocity (double, unsigned int);
     virtual double value (const Point<dim>   &p,
                     const unsigned int  component = 0) const;

     virtual void vector_value (const Point<dim> &p,
                       Vector<double>   &value) const;

     virtual void vector_value_list (const std::vector<Point<dim> > &p,
                         std::vector<Vector<double> > &values) const;

     double init_mean_vel;
     unsigned int which_inflow_type;
    };

    template <int dim>
    Inflow_Velocity<dim>::Inflow_Velocity (double init_mean_vel,
                            unsigned int which_inflow_type) :
    Function<dim> (dim),
    init_mean_vel (init_mean_vel),
    which_inflow_type (which_inflow_type)
    {}

    template <int dim>
    double Inflow_Velocity<dim>::value (const Point<dim>  &p,
                         const unsigned int component) const
    {
      double zz = 0.0;
      double H = 1.0;
      
      if (dim == 2 && component == 0)
      {
     double pp = p(1) + 0.5;
     zz = 4.*1.5*init_mean_vel*pp*(H - pp)/(H*H);
      if (which_inflow_type == 0) zz = 1.0;
      } 
      else if (dim == 3 && component == 2)
      {
     double pp = p[0]*p[0] + p[1]*p[1]; pp = std::sqrt(pp);
     zz = -4.0*2.0*init_mean_vel*(pp - 0.5)*(pp + 0.5);
        if (which_inflow_type == 0) zz = 1.0;
      }
      
      return zz;
    }

    template <int dim>
    void Inflow_Velocity<dim>::vector_value (const Point<dim> &p,
                                             Vector<double>   &values) const
    {
     for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = Inflow_Velocity<dim>::value (p, c);
    }

    template <int dim>
    void Inflow_Velocity<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                                  std::vector<Vector<double> >   &value_list) const
    {
     for (unsigned int p=0; p<points.size(); ++p)
      Inflow_Velocity<dim>::vector_value (points[p], value_list[p]);
    }

    template <int dim>
    class Outflow_Pressure : public Function<dim>
    {

     public:

     Outflow_Pressure (double, double);
     virtual double value (const Point<dim>   &p,
         const unsigned int  component = 0) const;

     virtual void vector_value (const Point<dim> &p,
       Vector<double>   &value) const;

     virtual void vector_value_list (const std::vector<Point<dim> > &p,
           std::vector<Vector<double> > &values) const;

     double inclined_angle, Froude_number;
    
    };

    template <int dim>
    Outflow_Pressure<dim>::Outflow_Pressure (double inclined_angle,
          double Froude_number) :
    Function<dim> (1),
    inclined_angle (inclined_angle),
    Froude_number (Froude_number)
    {}

    template <int dim>
    double Outflow_Pressure<dim>::value (const Point<dim>  &p,
                                         const unsigned int component) const
    {
     return (1./(Froude_number*Froude_number))*inclined_angle*(1.0-p[1])  + 0.0;
    }

    template <int dim>
    void Outflow_Pressure<dim>::vector_value (const Point<dim> &p,
                                              Vector<double>   &values) const
    {
     for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = Outflow_Pressure<dim>::value (p, c);
    }

    template <int dim>
    void Outflow_Pressure<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                                   std::vector<Vector<double> >   &value_list) const
    {
     for (unsigned int p=0; p<points.size(); ++p)
      Outflow_Pressure<dim>::vector_value (points[p], value_list[p]);
    }

    template <int dim>
    class concentrInletValues : public Function<dim>
    {
     public:
     concentrInletValues () : Function<dim>(1) {}

     virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;

     virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &value) const;
    };



    template <int dim>
    double
    concentrInletValues<dim>::value (const Point<dim>  &p,
                                     const unsigned int) const
    {
     return 0;
    }


    template <int dim>
    void
    concentrInletValues<dim>::vector_value (const Point<dim> &p,
                                            Vector<double>   &values) const
    {
     for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = concentrInletValues<dim>::value (p, c);
    }
    
    template <int dim>
    class concentrInitialValues : public Function<dim>
    {

     public:

     concentrInitialValues (double x);

     virtual double value (const Point<dim>   &p,
                             const unsigned int  component = 0) const;

     virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &value) const;

     virtual void vector_value_list (const std::vector<Point<dim> > &p,
                                     std::vector<Vector<double> > &values) const;

     double x;

    };

    template <int dim>
    concentrInitialValues<dim>::concentrInitialValues (double x) :
    Function<dim>(1),
    x (x)
    {}

    template <int dim>
    double concentrInitialValues<dim>::value (const Point<dim>  &p,
           const unsigned int component) const
    {
     double zz = downstream_concentr;

     if (p[0] < x && dim == 2)  zz = upstream_concentr;
     if (p[dim-1] < x && dim == 3)  zz = upstream_concentr;

     return zz;
    }

    template <int dim>
    void concentrInitialValues<dim>::vector_value (const Point<dim> &p,
                                            Vector<double>   &values) const
    {
     for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = concentrInitialValues<dim>::value (p, c);
    }

    template <int dim>
    void concentrInitialValues<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                                 std::vector<Vector<double> >   &value_list) const
    {
     for (unsigned int p=0; p<points.size(); ++p)
       concentrInitialValues<dim>::vector_value (points[p], value_list[p]);
    }
    
  } //END-EquationData

#endif
