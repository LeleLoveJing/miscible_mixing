#include <mismix/include.h>
#include <mismix/equation_data.h>
#include <mismix/assembly_copydata.h>
#include <mismix/class.h>
#include <mismix/parameter.h>

  template <int dim>
  void 
  UBC_mis_mixing<dim>::create_triangulation ()
  {
    if (parameters.is_verbal_output == true)
      pcout << "* Create Triangulation.." << std::endl;
          
    std::ostringstream input_xx;
    input_xx << "mesh/" << parameters.input_mesh_file;
    GridIn<dim> gmsh_input;
    std::ifstream in(input_xx.str().c_str());
    gmsh_input.attach_triangulation (triangulation);
    gmsh_input.read_msh (in);
    
    min_h_size = 10000000.0;
    Point<dim> domain_flow_dir;
    
    parameters.length_of_domain[parameters.flow_direction] = 0.0;
    parameters.length_of_domain[parameters.latitude_direction] = 0.0;
    if (dim == 3)parameters.length_of_domain[parameters.depth_direction] = 0.0;
    
    for (typename Triangulation<dim>::active_cell_iterator
         cell=triangulation.begin_active();
         cell!=triangulation.end(); ++cell)
    {
      
      min_h_size = std::min (cell->minimum_vertex_distance (), min_h_size);
      
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      {
        const Point<dim> face_center = cell->face(f)->center();
        
        if (cell->face(f)->at_boundary())
        {
          domain_flow_dir[parameters.flow_direction] = std::max (domain_flow_dir[parameters.flow_direction],
                                                       face_center[parameters.flow_direction]);
          domain_flow_dir[parameters.latitude_direction] = std::max (domain_flow_dir[parameters.latitude_direction],
                                                           face_center[parameters.latitude_direction]);
            
          if (dim == 3)
          domain_flow_dir[parameters.depth_direction] = std::max (domain_flow_dir[parameters.depth_direction],
                                                        face_center[parameters.depth_direction]);
        }
      }
    }
    
    parameters.length_of_domain[parameters.flow_direction] = domain_flow_dir[parameters.flow_direction];
    parameters.length_of_domain[parameters.latitude_direction] = domain_flow_dir[parameters.latitude_direction];
    if (dim == 3)
      parameters.length_of_domain[parameters.depth_direction] = domain_flow_dir[parameters.depth_direction];
    
    computed_time_step = min_h_size*
                         (parameters.CFL_number/
                         double(parameters.max_grid_level+1.0));
      
    pcout << "- Length of Domain = " << parameters.length_of_domain [parameters.flow_direction] << std::endl;
    pcout << "- Min. of h = " << min_h_size << std::endl;
    pcout << "- Computed Time Step = " << computed_time_step << std::endl;
    
    
    assgined_boundary_indicator ();
  }
  
  template <int dim>
  void 
  UBC_mis_mixing<dim>::assgined_boundary_indicator ()
  {   
    global_Omega_diameter = GridTools::diameter (triangulation);
    for (typename Triangulation<dim>::active_cell_iterator
         cell=triangulation.begin_active();
         cell!=triangulation.end(); ++cell)
    {
      const Point<dim> cell_center = cell->center();

      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      {
       const Point<dim> face_center = cell->face(f)->center();
     
       if (cell->face(f)->at_boundary())
       {
        cell->face(f)->set_boundary_id (0);
       
        if (dim == 2)
        {
         if (std::abs(face_center[parameters.flow_direction]-0) < 1e-6)
          cell->face(f)->set_boundary_id (parameters.domain_boundary[0]);

         if (std::abs(face_center[parameters.flow_direction]-parameters.length_of_domain[parameters.flow_direction]) < 1e-6)
          cell->face(f)->set_boundary_id (parameters.domain_boundary[1]);
        }
   
        if (dim == 3)
        {
        	if (parameters.is_symmetry_boundary == true &&
        			  std::abs(face_center[parameters.depth_direction]-0) < 1e-6)
         	cell->face(f)->set_boundary_id (6);

         if (std::abs(face_center[parameters.flow_direction]-0) < 1e-6)
          cell->face(f)->set_boundary_id (parameters.domain_boundary[0]);
  
         if (std::abs(face_center[parameters.flow_direction]-parameters.length_of_domain[parameters.flow_direction]) < 1e-6)
          cell->face(f)->set_boundary_id (parameters.domain_boundary[1]);
        }
       }
      }
    }
  }

// Explicit instantiations
// template class UBC_mis_mixing<1>;
template class UBC_mis_mixing<2>;
template class UBC_mis_mixing<3>;
