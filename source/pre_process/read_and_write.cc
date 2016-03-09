#include <mismix/include.h>
#include <mismix/equation_data.h>
#include <mismix/assembly_copydata.h>
#include <mismix/class.h>
#include <mismix/parameter.h>

#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#ifdef DEAL_II_WITH_ZLIB
#  include <zlib.h>
#endif

  template <int dim>
  void 
  UBC_mis_mixing<dim>::move_file (const std::string &old_name,
                                  const std::string &new_name)
  {
    const int error = system (("mv " + old_name + " " + new_name).c_str());

    AssertThrow (error == 0, ExcMessage(std::string ("Can't move files: ")
                             +
                             old_name + " -> " + new_name));
  
  }

  template <int dim>
  void 
  UBC_mis_mixing<dim>::save_snapshot_template (std::vector<const TrilinosWrappers::MPI::Vector *> &system,
                                               std::stringstream                                  &file_stream,
                                               std::stringstream                                  &file_zlib_stream,
                                               DoFHandler<dim>                                    &dof_handler_this)
  {
    
    unsigned int my_id = Utilities::MPI::this_mpi_process (MPI_COMM_WORLD);
    
    parallel::distributed::SolutionTransfer<dim,
                                           TrilinosWrappers::MPI::Vector>
      trans (dof_handler_this);
    
    trans.prepare_serialization (system);
    
    triangulation.save (file_stream.str().c_str());
      
    std::ostringstream oss (file_stream.str().c_str());

    boost::archive::binary_oarchive oa (oss);
    
#ifdef DEAL_II_WITH_ZLIB
      if (my_id == 0)
        {
          uLongf compressed_data_length = compressBound (oss.str().length());
          std::vector<char *> compressed_data (compressed_data_length);
          int err = compress2 ((Bytef *) &compressed_data[0],
                               &compressed_data_length,
                               (const Bytef *) oss.str().data(),
                               oss.str().length(),
                               Z_BEST_COMPRESSION);
          (void)err;
          Assert (err == Z_OK, ExcInternalError());

          // build compression header
          const uint32_t compression_header[4]
            = { 1,                            /* number of blocks */
                (uint32_t)oss.str().length(), /* size of block */
                (uint32_t)oss.str().length(), /* size of last block */
                (uint32_t)compressed_data_length
              }; /* list of compressed sizes of blocks */

          std::ofstream f (file_zlib_stream.str().c_str());
          f.write((const char *)compression_header, 4 * sizeof(compression_header[0]));
          f.write((char *)&compressed_data[0], compressed_data_length);
        }
#else
      AssertThrow (false,
                   ExcMessage ("You need to have deal.II configured with the 'libz' "
                               "option to support checkpoint/restart, but deal.II "
                               "did not detect its presence when you called 'cmake'."));
#endif

  }

  template <int dim>
  void 
  UBC_mis_mixing<dim>::load_snapshot_template (unsigned int         which_variable,
                                               std::stringstream    &file_stream,
                                               std::stringstream    &file_zlib_stream)
  {
  
    try
      {
        if (which_variable == 0)
        {
          triangulation.clear ();
          create_triangulation ();
          assgined_boundary_indicator ();
        }
        triangulation.load (file_stream.str().c_str());

      }
    catch (...)
      {
        AssertThrow(false, ExcMessage("Cannot open snapshot mesh file or read the triangulation stored there."));
      }
    
    switch (which_variable)
    {
      case 0: {
                  setup_dofs_concentr ();
                  TrilinosWrappers::MPI::Vector d0_system (concentr_rhs),
                                                d1_system (concentr_rhs),
                                                d2_system (concentr_rhs);

                  std::vector<TrilinosWrappers::MPI::Vector *> temp_sol (3);
                    temp_sol[0] = & (d0_system);
                    temp_sol[1] = & (d1_system);
                    temp_sol[2] = & (d2_system);

                  parallel::distributed::SolutionTransfer<dim,
                                                         TrilinosWrappers::MPI::Vector>
                  trans (concentr_dof_handler);

                  trans.deserialize (temp_sol);

                  concentr_solution         = d0_system;
                  old_concentr_solution     = d1_system;
                  old_old_concentr_solution = d2_system;
      
                break;
              }
      case 1: {
                setup_dofs_velocity ();
                TrilinosWrappers::MPI::Vector d0_system (rhs_velocity),
                                              d1_system (rhs_velocity),
                                              d2_system (rhs_velocity);
      
                std::vector<TrilinosWrappers::MPI::Vector *> temp_sol (3);
                  temp_sol[0] = & (d0_system);
                  temp_sol[1] = & (d1_system);
                  temp_sol[2] = & (d2_system);
        
                parallel::distributed::SolutionTransfer<dim,
                                           TrilinosWrappers::MPI::Vector>
                  trans (dof_handler_velocity);
      
                trans.deserialize (temp_sol);
          
                vel_n_plus_1  = d0_system;
                vel_n         = d1_system;
                vel_n_minus_1 = d2_system;
          
                break;
              }
      case 2: {
                setup_dofs_pressure ();
                TrilinosWrappers::MPI::Vector d0_system (rhs_pressure),
                                              d1_system (rhs_pressure),
                                              d2_system (rhs_pressure),
                                              d3_system (rhs_pressure);
      
                std::vector<TrilinosWrappers::MPI::Vector *> temp_sol (4);
                  temp_sol[0] = & (d0_system);
                  temp_sol[1] = & (d1_system);
                  temp_sol[2] = & (d2_system);
                  temp_sol[3] = & (d3_system);
        
                parallel::distributed::SolutionTransfer<dim,
                                           TrilinosWrappers::MPI::Vector>
                  trans (dof_handler_pressure);
      
                trans.deserialize (temp_sol);
          
                pre_star       = d0_system;
                pre_n_plus_1   = d1_system;
                pre_n          = d2_system;
                pre_n_minus_1  = d3_system;
          
                break;
              }
      case 3: {
                setup_dofs_auxilary ();
                TrilinosWrappers::MPI::Vector d0_system (rhs_auxilary),
                                              d1_system (rhs_auxilary),
                                              d2_system (rhs_auxilary);
      
                std::vector<TrilinosWrappers::MPI::Vector *> temp_sol (3);
                  temp_sol[0] = & (d0_system);
                  temp_sol[1] = & (d1_system);
                  temp_sol[2] = & (d2_system);
        
                parallel::distributed::SolutionTransfer<dim,
                                           TrilinosWrappers::MPI::Vector>
                  trans (dof_handler_auxilary);
      
                trans.deserialize (temp_sol);
          
                aux_n_plus_1  = d0_system;
                aux_n         = d1_system;
                aux_n_minus_1 = d2_system;
          
                break;
              }

    }

    
 // read zlib compressed resume.z
    try
      {
#ifdef DEAL_II_WITH_ZLIB
        std::ifstream ifs (file_zlib_stream.str().c_str());
        AssertThrow(ifs.is_open(),
                    ExcMessage("Cannot open snapshot resume file."));

        uint32_t compression_header[4];
        ifs.read((char *)compression_header, 4 * sizeof(compression_header[0]));
        Assert(compression_header[0]==1, ExcInternalError());

        std::vector<char> compressed(compression_header[3]);
        std::vector<char> uncompressed(compression_header[1]);
        ifs.read(&compressed[0],compression_header[3]);
        uLongf uncompressed_size = compression_header[1];

        const int err = uncompress((Bytef *)&uncompressed[0], &uncompressed_size,
                                   (Bytef *)&compressed[0], compression_header[3]);
        AssertThrow (err == Z_OK,
                     ExcMessage (std::string("Uncompressing the data buffer resulted in an error with code <")
                                 +
                                 Utilities::int_to_string(err)));

//        {
//          std::istringstream ss (file_stream.str().c_str());
//          ss.str(std::string (&uncompressed[0], uncompressed_size));
//          boost::archive::binary_iarchive ia (ss);
////          ia >> (*this);
//
//        }
        
#else
        AssertThrow (false,
                     ExcMessage ("You need to have deal.II configured with the 'libz' "
                                 "option to support checkpoint/restart, but deal.II "
                                 "did not detect its presence when you called 'cmake'."));
#endif
      }
    catch (std::exception &e)
      {
        AssertThrow (false,
                     ExcMessage (std::string("Cannot seem to deserialize the data previously stored!\n")
                                 +
                                 "Some part of the machinery generated an exception that says <"
                                 +
                                 e.what()
                                 +
                                 ">"));
      }
      
      
  }

  template <int dim>
  void 
  UBC_mis_mixing<dim>::save_data_and_mesh ()
  {
    pcout << "* Save Data and Mesh... " << std::endl;
    
    unsigned int my_id = Utilities::MPI::this_mpi_process (MPI_COMM_WORLD);
     
      
    {
      std::vector<const TrilinosWrappers::MPI::Vector *> system (3);
        system[0] = &vel_n_plus_1;
        system[1] = &vel_n;
        system[2] = &vel_n_minus_1;

      std::stringstream file_stream, file_zlib_stream;
      file_stream      << "output/save/save_velocity.mesh";
      file_zlib_stream << "output/save/save_velocity_resume.z";
      save_snapshot_template (system,
                              file_stream,
                              file_zlib_stream,
                              dof_handler_velocity);
    }
 
    {
      std::vector<const TrilinosWrappers::MPI::Vector *> system (4);
        system[0] = &pre_star;
        system[1] = &pre_n_plus_1;
        system[2] = &pre_n;
        system[3] = &pre_n_minus_1;
        
      std::stringstream file_stream, file_zlib_stream;
      file_stream      << "output/save/save_pressure.mesh";
      file_zlib_stream << "output/save/save_pressure_resume.z";
      save_snapshot_template (system,
                              file_stream,
                              file_zlib_stream,
                              dof_handler_pressure);
    }
  
    {
      std::vector<const TrilinosWrappers::MPI::Vector *> system (3);
        system[0] = &aux_n_plus_1;
        system[1] = &aux_n;
        system[2] = &aux_n_minus_1;

      std::stringstream file_stream, file_zlib_stream;
      file_stream      << "output/save/save_auxilary.mesh";
      file_zlib_stream << "output/save/save_auxilary_resume.z";
      save_snapshot_template (system,
                              file_stream,
                              file_zlib_stream,
                              dof_handler_auxilary);
    }
  
    {
      std::vector<const TrilinosWrappers::MPI::Vector *> system (3);
        system[0] = &concentr_solution;
        system[1] = &old_concentr_solution;
        system[2] = &old_old_concentr_solution;

      std::stringstream file_stream, file_zlib_stream;
      file_stream      << "output/save/save_concentr.mesh";
      file_zlib_stream << "output/save/save_concentr_resume.z";
      save_snapshot_template (system,
                              file_stream,
                              file_zlib_stream,
                              concentr_dof_handler);
    }
    
    pcout << "* Snapshot Created!" << std::endl;
    
  }

  template <int dim>
  void 
  UBC_mis_mixing<dim>::load_data_and_mesh ()
  {
    pcout << "* Load Data and Mesh... " << std::endl;
    
    {
      std::stringstream file_stream, file_zlib_stream;
      file_stream      << "output/save/save_concentr.mesh";
      file_zlib_stream << "output/save/save_concentr_resume.z";
        
      load_snapshot_template (0,
                              file_stream,
                              file_zlib_stream);

    }
    
    {
      std::stringstream file_stream, file_zlib_stream;
      file_stream      << "output/save/save_velocity.mesh";
      file_zlib_stream << "output/save/save_velocity_resume.z";
      
      load_snapshot_template (1,
                              file_stream,
                              file_zlib_stream);
    }
    
    {
      std::stringstream file_stream, file_zlib_stream;
      file_stream      << "output/save/save_pressure.mesh";
      file_zlib_stream << "output/save/save_pressure_resume.z";
      
      load_snapshot_template (2,
                              file_stream,
                              file_zlib_stream);
    }
    
    {
      std::stringstream file_stream, file_zlib_stream;
      file_stream      << "output/save/save_auxilary.mesh";
      file_zlib_stream << "output/save/save_auxilary_resume.z";
      
      load_snapshot_template (3,
                              file_stream,
                              file_zlib_stream);
    }
      
    setup_dofs_error ();

  }

// Explicit instantiations
// template class UBC_mis_mixing<1>;
template class UBC_mis_mixing<2>;
template class UBC_mis_mixing<3>;
