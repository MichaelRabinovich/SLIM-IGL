#include <iostream>

#include "SLIMData.h"
#include "Slim.h"
#include "eigen_stl_utils.h"
#include "geometric_utils.h"

#include "igl/components.h"
#include "igl/writeOBJ.h"
#include "igl/Timer.h"

#include <igl/serialize.h>
#include <igl/read_triangle_mesh.h>

#include <stdlib.h>

#include <string>
#include <vector>

using namespace std;

void check_mesh_for_issues(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::VectorXd& areas);

Eigen::MatrixXd V;
Eigen::MatrixXi F;

const int ITER_NUM = 7;

int main(int argc, char *argv[]) {
  if (argc < 3) {
      cerr << "Syntax: " << argv[0] << " <input mesh> <output mesh>" << std::endl;
      return -1;
  }
  const string input_mesh = argv[1]; 
  const string output_mesh = argv[2];

  cout << "Reading mesh " << input_mesh << endl;
  igl::read_triangle_mesh(input_mesh, V, F);

  SLIMData sData(V,F);
  check_mesh_for_issues(sData.V,sData.F, sData.M);
  cout << "\tMesh is valid!" << endl;

  sData.slim_energy = SLIMData::SYMMETRIC_DIRICHLET;
  
  dirichlet_on_circle(sData.V,sData.F,sData.V_o);
  if (count_flips(sData.V,sData.F,sData.V_o) > 0) {
      tutte_on_circle(sData.V,sData.F,sData.V_o);
  }
  
  cout << "initialized parametrization" << endl;
  Slim slim(sData);
  slim.precompute();
  slim.solve(ITER_NUM);

  cout << "Finished, saving results to " << output_mesh << endl;
  igl::writeOBJ(output_mesh, sData.V, sData.F, Eigen::MatrixXd(), Eigen::MatrixXi(), sData.V_o, sData.F);

  return 0;
}

void check_mesh_for_issues(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::VectorXd& areas) {

  Eigen::SparseMatrix<double> A;
  igl::adjacency_matrix(F,A);

  Eigen::MatrixXi C, Ci;
  igl::components(A, C, Ci);
  //cout << "#Connected_Components = " << Ci.rows() << endl;
  //cout << "is edge manifold = " << igl::is_edge_manifold(V,F) << endl;
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
