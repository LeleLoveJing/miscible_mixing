#include <mismix/include.h>
#include <mismix/equation_data.h>
#include <mismix/assembly_copydata.h>
#include <mismix/class.h>
#include <mismix/parameter.h>

// get the value of a particular parameter from the contents of the input
// file. return an empty string if not found
std::string
get_last_value_of_parameter(const std::string &parameters,
                            const std::string &parameter_name)
{
  std::string return_value;

  std::istringstream x_file(parameters);
  while (x_file)
    {
      // get one line and strip spaces at the front and back
      std::string line;
      std::getline(x_file, line);
      while ((line.size() > 0) && (line[0] == ' ' || line[0] == '\t'))
        line.erase(0, 1);
      while ((line.size() > 0)
             && (line[line.size() - 1] == ' ' || line[line.size() - 1] == '\t'))
        line.erase(line.size() - 1, std::string::npos);
      // now see whether the line starts with 'set' followed by multiple spaces
      // if not, try next line
      if (line.size() < 4)
        continue;

      if ((line[0] != 's') || (line[1] != 'e') || (line[2] != 't')
          || !(line[3] == ' ' || line[3] == '\t'))
        continue;

      // delete the "set " and then delete more spaces if present
      line.erase(0, 4);
      while ((line.size() > 0) && (line[0] == ' ' || line[0] == '\t'))
        line.erase(0, 1);
      // now see whether the next word is the word we look for
      if (line.find(parameter_name) != 0)
        continue;

      line.erase(0, parameter_name.size());
      while ((line.size() > 0) && (line[0] == ' ' || line[0] == '\t'))
        line.erase(0, 1);

      // we'd expect an equals size here
      if ((line.size() < 1) || (line[0] != '='))
        continue;

      // remove comment
      std::string::size_type pos = line.find('#');
      if (pos != std::string::npos)
        line.erase (pos);

      // trim the equals sign at the beginning and possibly following spaces
      // as well as spaces at the end
      line.erase(0, 1);
      while ((line.size() > 0) && (line[0] == ' ' || line[0] == '\t'))
        line.erase(0, 1);
      while ((line.size() > 0) && (line[line.size()-1] == ' ' || line[line.size()-1] == '\t'))
        line.erase(line.size()-1, std::string::npos);

      // the rest should now be what we were looking for
      return_value = line;
    }

  return return_value;
}

// extract the dimension in which to run ps_MMM from the
// the contents of the parameter file. this is something that
// we need to do before processing the parameter file since we
// need to know whether to use the dim=2 or dim=3 instantiation
// of the main classes
unsigned int
get_dimension(const std::string &parameters)
{
  const std::string dimension = get_last_value_of_parameter(parameters, "Dimension");
  if (dimension.size() > 0)
    return dealii::Utilities::string_to_int (dimension);
  else
    return 2;
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

//     int dimn;
//     std::ifstream dim_xx ("dimension.prm");
//     dim_xx >> dimn;

     const unsigned int dimn = get_dimension(parameter_filename);

     std::cout << "<-- " << dimn << " Dimension Input" << std::endl;

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
