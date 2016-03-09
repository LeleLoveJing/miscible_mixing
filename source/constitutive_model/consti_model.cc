#include <mismix/include.h>
#include <mismix/equation_data.h>
#include <mismix/assembly_copydata.h>
#include <mismix/class.h>
#include <mismix/parameter.h>
  
  template <int dim>
  std::pair<double,double>
  UBC_mis_mixing<dim>::compute_discont_variable_on_cell (unsigned int                         number_of_data,
                                                         std::vector<double>                  &data,
                                                         std::vector<Tensor<2,dim> >          &tensor_data)
  {
    std::pair<double, double> local_value;

    /* Average for density over quadrature points */
    for (unsigned int qq=0; qq<number_of_data; ++qq)
    {
      double b = data[qq];
      if (b>1.0) data[qq] = 1.0;
      if (b<0.0) data[qq] = 0.0;
      local_value.first += b/double(number_of_data);
    }

    /* Average for viscosity with pow-law model over quadrature points */
    double fluid1_viscosity = parameters.mean_viscosity;
    double fluid2_viscosity = parameters.mean_viscosity;

    /* Set the viscosity of the fluid with pow-law model */
    std::list<double> list;

    for (unsigned int m=0; m<number_of_data; ++m)
    {
      double sum_shear_rate_at_data = 0.0;
      for (unsigned int i=0; i<dim; ++i)
      for (unsigned int j=0; j<dim; ++j)
      {
        double shear_rate_at_data = tensor_data[m][i][j] +tensor_data[m][j][i];

        sum_shear_rate_at_data += std::sqrt(0.5*shear_rate_at_data*shear_rate_at_data);
      }
      list.push_back (sum_shear_rate_at_data);
    }

    assert (!list.empty());
    assert (list.size() == number_of_data);

    double average_shear_rate = std::accumulate(list.begin(), list.end(), 0.0) / list.size();

    if (std::abs(average_shear_rate) > 1e-4)
    {
      fluid2_viscosity = parameters.ratio_pow_law *
                         std::pow(average_shear_rate, parameters.n_pow_law);
    }
    else if (std::abs(average_shear_rate) < 1e-4 || (parameters.n_pow_law - 1.0) < 1e-4)
    {
      if (parameters.ratio_pow_law > 1.0)
      { fluid2_viscosity = parameters.ratio_pow_law * parameters.mean_viscosity;}
      else if (parameters.ratio_pow_law <= 1.0)
      { fluid1_viscosity = parameters.ratio_pow_law * parameters.mean_viscosity;}
    }

    /* Determine the viscosity */
    local_value.second = (1.0-local_value.first)*fluid1_viscosity +
                         local_value.first*fluid2_viscosity;

    /* Return */
    return std::make_pair (local_value.first, local_value.second);
  }

// Explicit instantiations
// template class UBC_mis_mixing<1>;
template class UBC_mis_mixing<2>;
template class UBC_mis_mixing<3>;
