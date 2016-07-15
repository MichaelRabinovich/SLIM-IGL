#include <iostream>

#include "SLIMData.h"
#include "Slim.h"
#include "geometric_utils.h"

#include "igl/components.h"
#include "igl/readOBJ.h"
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
using namespace Eigen;

void check_mesh_for_issues(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::VectorXd& areas);
void param_2d_demo_iter(igl::viewer::Viewer& viewer);
void set_soft_constraint_for_circle();
void soft_const_demo_iter(igl::viewer::Viewer& viewer);
void deform_3d_demo_iter(igl::viewer::Viewer& viewer);
void display_3d_mesh(igl::viewer::Viewer& viewer);

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
      case DEFORM_3D: {
        deform_3d_demo_iter(viewer);
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
    viewer.core.show_texture = true;

    first_iter = false;
  } else {
    slim->solve(1); // 1 iter
    viewer.data.set_uv(sData->V_o*uv_scale_param);
  }
}

void soft_const_demo_iter(igl::viewer::Viewer& viewer) {
  if (first_iter) {

    igl::read_triangle_mesh("../circle.obj", V, F);

    sData = new SLIMData(V,F);
    check_mesh_for_issues(sData->V,sData->F, sData->M);
    cout << "\tMesh is valid!" << endl;
    sData->V_o = V.block(0,0,V.rows(),2);

    set_soft_constraint_for_circle();

    sData->slim_energy = SLIMData::SYMMETRIC_DIRICHLET;

    sData->soft_const_p = 1e5;
    slim = new Slim(*sData);
    slim->precompute();


    viewer.data.set_mesh(V, F);
    viewer.data.compute_normals();
    viewer.core.show_lines = true;

    first_iter = false;

  } else {
    slim->solve(1); // 1 iter
    viewer.data.set_mesh(sData->V_o, F);
  }
}

void deform_3d_demo_iter(igl::viewer::Viewer& viewer) {
  if (first_iter) {
    igl::readOBJ("../cube_4k.obj", V, F);

    sData = new SLIMData(V,F);
    sData->V_o = V;

    cout << "V.rows() = " << V.rows() << " F.rows() = " << F.rows() << endl;
    display_3d_mesh(viewer);
    first_iter = false;
  } else {

  }
}

void display_3d_mesh(igl::viewer::Viewer& viewer) {
  MatrixXd V_temp; MatrixXi F_temp;
  Eigen::MatrixXd Barycenters;

  igl::barycenter(sData->V,sData->F,Barycenters);
  //cout << "Barycenters.rows() = " << Barycenters.rows() << endl;
  //double t = double((key - '1')+1) / 9.0;
  double view_depth = 10.;
  double t = view_depth/9.;

  VectorXd v = Barycenters.col(2).array() - Barycenters.col(2).minCoeff();
  v /= v.col(0).maxCoeff();

  vector<int> s;

  for (unsigned i=0; i<v.size();++i)
    if (v(i) < t)
      s.push_back(i);

  V_temp.resize(s.size()*4,3);
  F_temp.resize(s.size()*4,3);

  for (unsigned i=0; i<s.size();++i){
    V_temp.row(i*4+0) = sData->V_o.row(sData->F(s[i],0));
    V_temp.row(i*4+1) = sData->V_o.row(sData->F(s[i],1));
    V_temp.row(i*4+2) = sData->V_o.row(sData->F(s[i],2));
    V_temp.row(i*4+3) = sData->V_o.row(sData->F(s[i],3));
    F_temp.row(i*4+0) << (i*4)+0, (i*4)+1, (i*4)+3;
    F_temp.row(i*4+1) << (i*4)+0, (i*4)+2, (i*4)+1;
    F_temp.row(i*4+2) << (i*4)+3, (i*4)+2, (i*4)+0;
    F_temp.row(i*4+3) << (i*4)+1, (i*4)+2, (i*4)+3;
  }
  cout << "V_temp.rows() = " << V_temp.rows() << endl;
  viewer.data.set_mesh(V_temp,F_temp);
  viewer.data.set_face_based(true);
  viewer.core.show_lines = true;
}

void get_mesh_for_igl_viewer(Eigen::MatrixXd& V_temp, Eigen::MatrixXi& F_temp) {
  using namespace Eigen;


}

int main(int argc, char *argv[]) {

   if (argc < 2) {
      cerr << "Syntax: " << argv[0] << " demo_number (1 to 3)" << std::endl;
      return -1;
  }

  switch (std::atoi(argv[1])) {
    case 1: {
      demo_type = PARAM_2D;
      break;
    } case 2: {
      demo_type = SOFT_CONST;
      break;
    } case 3: {
      demo_type = DEFORM_3D;
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
  viewer.core.show_texture = false;
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

void set_soft_constraint_for_circle() {

    Eigen::VectorXi bnd;
    igl::boundary_loop(sData->F,bnd);
    const int B_STEPS = 22;
    cout << "bnd.rows() = " << bnd.rows() << endl;
    
    sData->b.resize(bnd.rows()/B_STEPS - 1);
    sData->bc.resize(sData->b.rows(),2);

    int c_idx = 0;
    cout << "consts num = " << sData->b.rows() << endl;
    for (int i = B_STEPS; i < bnd.rows(); i+=B_STEPS) {
        cout << "i = " << i << " bnd(i) = " << bnd(i) << " uv = " << sData->V_o.row(bnd(i)) << endl;
        sData->b(c_idx) = bnd(i);
        //sData->bc.row(c_idx) << sData->V_o(bnd(i),0), 0.1*c_idx*(sData->V_o(bnd(i),1));
        c_idx++;
    }
    cout << "bc.rows() = " << sData->bc.rows() << endl; //exit(1);
    
    sData->bc.row(0) = sData->V_o.row(sData->b(0)); // keep it there for now
    sData->bc.row(1) = sData->V_o.row(sData->b(2));
    sData->bc.row(2) = sData->V_o.row(sData->b(3));
    sData->bc.row(3) = sData->V_o.row(sData->b(4));
    sData->bc.row(4) = sData->V_o.row(sData->b(5));


    sData->bc.row(0) << sData->V_o(sData->b(0),0), 0;
    sData->bc.row(4) << sData->V_o(sData->b(4),0), 0;
    sData->bc.row(2) << sData->V_o(sData->b(2),0), 0.1;
    sData->bc.row(3) << sData->V_o(sData->b(3),0), 0.05;
    sData->bc.row(1) << sData->V_o(sData->b(1),0), -0.15;
    sData->bc.row(5) << sData->V_o(sData->b(5),0), +0.15;
}