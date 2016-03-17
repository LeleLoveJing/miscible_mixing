#include <mismix/include.h>
#include <mismix/equation_data.h>
#include <mismix/assembly_copydata.h>
#include <mismix/class.h>
#include <mismix/parameter.h>

// extract the dimension in which to run ps_MMM from the
// the contents of the parameter file. this is something that
// we need to do before processing the parameter file since we
// need to know whether to use the dim=2 or dim=3 instantiation
// of the main classes
unsigned int
get_dimension(std::ifstream &input_string_dimension)
{
  std::string output_string;
  input_string_dimension >> output_string;

  return dealii::Utilities::string_to_int (output_string);
}

int main (int argc, char *argv[])
{
  using namespace dealii;



  try
  {
     deallog.depth_console (0);

     Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                         numbers::invalid_unsigned_int);

     std::string parameter_filename;
     if (argc>=2)
       parameter_filename = argv[1];
     else
     {
       std::cout << "ERROR : Please specify the input file name..." << std::endl;

       return 0;
     }

     std::ifstream input_string_dimension ("dimension.prm");
     const unsigned int dimn = get_dimension(input_string_dimension);
     std::cout << "#### " << dimn << " Dimensional ps-MMM Simulation......" << std::endl;

     switch (dimn)
     {
       case 2 :
       {
         UBC_mis_mixing<2>::Parameters parameters(parameter_filename);
         UBC_mis_mixing<2> flow_problem (parameters);
         flow_problem.run ();
         break;
       }
       case 3 :
       {
         UBC_mis_mixing<3>::Parameters parameters(parameter_filename);
         UBC_mis_mixing<3> flow_problem (parameters);
         flow_problem.run ();
         break;
       }
       default:
            AssertThrow((dimn >= 2) && (dimn <= 3),
                        ExcMessage ("ps_MMM can only be run in 2d and 3d but a "
                                    "different space dimension is given in the parameter file."));
     }
    }

    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Exception on processing: " << std::endl
        << exc.what() << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;

        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Unknown exception!" << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }

    return 0;
}

// Explicit instantiations
// template class UBC_mis_mixing<1>;
template class UBC_mis_mixing<2>;
template class UBC_mis_mixing<3>;
