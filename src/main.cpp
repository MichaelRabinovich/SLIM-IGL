#include <iostream>

#include "SLIMData.h"
#include "Slim.h"
#include "geometric_utils.h"

#include "igl/components.h"
#include "igl/writeOBJ.h"
#include "igl/Timer.h"

#include "igl/boundary_loop.h"
#include "igl/map_vertices_to_circle.h"
#include "igl/harmonic.h"
#include <igl/serialize.h>
#include <igl/read_triangle_mesh.h>
#include <igl/viewer/Viewer.h>

#include <stdlib.h>

#include <string>
#include <vector>

using namespace std;

void check_mesh_for_issues(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::VectorXd& areas);
void param_2d_demo_iter(igl::viewer::Viewer& viewer);
void soft_const_demo_iter(igl::viewer::Viewer& viewer);

Eigen::MatrixXd V;
Eigen::MatrixXi F;
bool first_iter = true;
SLIMData* sData = NULL;
Slim* slim = NULL;

double uv_scale_param;

enum DEMO_TYPE {
  PARAM_2D,
  SOFT_CONST,
  DEFORM_3D
};
DEMO_TYPE demo_type;

bool key_down(igl::viewer::Viewer& viewer, unsigned char key, int modifier){
  /*
  if (key == '1')
    show_uv = false;
  else if (key == '2')
    show_uv = true;

  if (key == 'q')
    V_uv = initial_guess;

  if (show_uv)
  {
    viewer.data.set_mesh(V_uv,F);
    viewer.core.align_camera_center(V_uv,F);
  }
  else
  {
    viewer.data.set_mesh(V,F);
    viewer.core.align_camera_center(V,F);
  }
  */
  if (key == '6') {
    switch (demo_type) {
      case PARAM_2D: {
        param_2d_demo_iter(viewer);
        break;
      }
      case SOFT_CONST: {
        soft_const_demo_iter(viewer);
        break;
      }
      default:
        break;
    }
  }

  return false;
}

void param_2d_demo_iter(igl::viewer::Viewer& viewer) {
  if (first_iter) {
    //cout << "Reading mesh " << input_mesh << endl;
    igl::read_triangle_mesh("../camelhead.obj", V, F);

    sData = new SLIMData(V,F);
    //SLIMData sData(V,F);
    check_mesh_for_issues(sData->V,sData->F, sData->M);
    cout << "\tMesh is valid!" << endl;

    sData->slim_energy = SLIMData::SYMMETRIC_DIRICHLET;
    
    Eigen::VectorXi bnd; Eigen::MatrixXd bnd_uv;
    igl::boundary_loop(F,bnd);
    igl::map_vertices_to_circle(V,bnd,bnd_uv);

    igl::harmonic(V,F,bnd,bnd_uv,1,sData->V_o);
    if (count_flips(sData->V,sData->F,sData->V_o) > 0) {
      igl::harmonic(F,bnd,bnd_uv,1,sData->V_o); // use uniform laplacian
    }

    cout << "initialized parametrization" << endl;
    slim = new Slim(*sData);
    slim->precompute();

    uv_scale_param = 15 * (1./sqrt(sData->mesh_area));
    viewer.data.set_mesh(V, F);
    viewer.data.set_uv(sData->V_o*uv_scale_param);
    viewer.data.compute_normals();

    first_iter = false;
  } else {
    slim->solve(1); // 1 iter
    viewer.data.set_uv(sData->V_o*uv_scale_param);
  }
}

void soft_const_demo_iter(igl::viewer::Viewer& viewer) {

}

int main(int argc, char *argv[]) {

   if (argc < 2) {
      cerr << "Syntax: " << argv[0] << " demo_number(1 to 3)" << std::endl;
      return -1;
  }

  switch (std::atoi(argv[1])) {
    case 1: {
      demo_type = PARAM_2D;
      break;
    }
    default: {
      cerr << "Wrong demo number - Please choose one between 1-3" << std:: endl;
      exit(1);
    }
  }

  // Launch the viewer
  igl::viewer::Viewer viewer;
  //viewer.data.set_uv(V_uv);
  viewer.callback_key_down = &key_down;

  // Disable wireframe
  viewer.core.show_lines = false;

  // Draw checkerboard texture
  viewer.core.show_texture = true;
  viewer.launch();

  return 0;
}

void check_mesh_for_issues(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::VectorXd& areas) {

  Eigen::SparseMatrix<double> A;
  igl::adjacency_matrix(F,A);

  Eigen::MatrixXi C, Ci;
  igl::components(A, C, Ci);
  
  int connected_components = Ci.rows();
  if (connected_components!=1) {
    cout << "Error! Input has multiple connected components" << endl; exit(1);
  }
  int euler_char = get_euler_char(V, F);
  if (!euler_char) {
    cout << "Error! Input does not have a disk topology, it's euler char is " << euler_char << endl; exit(1);
  }
  bool is_edge_manifold = igl::is_edge_manifold(V, F);
  if (!is_edge_manifold) {
    cout << "Error! Input is not an edge manifold" << endl; exit(1);
  }
  const double eps = 1e-14;
  for (int i = 0; i < areas.rows(); i++) {
    if (areas(i) < eps) {
      cout << "Error! Input has zero area faces" << endl; exit(1);
    }
  }
}
