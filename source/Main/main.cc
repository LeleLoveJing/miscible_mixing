#include <mismix/include.h>
#include <mismix/equation_data.h>
#include <mismix/assembly_copydata.h>
#include <mismix/class.h>
#include <mismix/parameter.h>

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

     int dimn;
     std::ifstream dim_xx ("dimension.prm");
     dim_xx >> dimn;
 
        if ( dimn == 2)
        {
          UBC_mis_mixing<2>::Parameters parameters(parameter_filename);
          UBC_mis_mixing<2> flow_problem (parameters);
          flow_problem.run ();
        }
        else if (dimn == 3)
        {
          UBC_mis_mixing<3>::Parameters parameters(parameter_filename);
          UBC_mis_mixing<3> flow_problem (parameters);
          flow_problem.run ();   
        }
        else 
        {
          std::cout << "ERROR : Please put the correct dimensions [2, 3]" << std::endl;
          
          return 0;   
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
