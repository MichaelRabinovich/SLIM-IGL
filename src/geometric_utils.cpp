#include "geometric_utils.h"

#include <igl/adjacency_matrix.h>
#include <igl/arap.h>
#include <igl/avg_edge_length.h>
#include <igl/boundary_loop.h>
#include <igl/colon.h>
#include <igl/harmonic.h>
#include <igl/edge_topology.h>
#include <igl/grad.h>
#include <igl/slice_into.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/lscm.h>
#include <igl/project_isometrically_to_plane.h>
#include <igl/doublearea.h>
#include <igl/volume.h>
#include <igl/per_face_normals.h>
#include <igl/writeOBJ.h>

#include <igl/project_isometrically_to_plane.h>
#include <igl/repdiag.h>
#include <igl/covariance_scatter_matrix.h>
#include <igl/edge_lengths.h>

#undef NDEBUG
#include <assert.h>
#define NDEBUG

void compute_surface_gradient_matrix(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                                     const Eigen::MatrixXd& F1, const Eigen::MatrixXd& F2,
                                     Eigen::SparseMatrix<double>& D1, Eigen::SparseMatrix<double>& D2) {
  using namespace Eigen;
  Eigen::SparseMatrix<double> G;
  //igl::grad(V,F,G);


  // Get grad
  const int fn = F.rows();  const int vn = V.rows();
  Eigen::MatrixXd grad3_3f(3, 3*fn);
  Eigen::MatrixXd fN; igl::per_face_normals(V,F,fN);
  Eigen::VectorXd Ar; igl::doublearea(V,F, Ar);
  for (int i = 0; i < fn; i++) {
     // renaming indices of vertices of triangles for convenience
    int i1 = F(i,0);
    int i2 = F(i,1);
    int i3 = F(i,2);

    // #F x 3 matrices of triangle edge vectors, named after opposite vertices
    Eigen::Matrix<double, 3,3> e;
    e.col(0) = V.row(i2) - V.row(i1);
    e.col(1) = V.row(i3) - V.row(i2);
    e.col(2) = V.row(i1) - V.row(i3);;
    
    Eigen::Matrix<double, 3,1> Fni = fN.row(i);
    double Ari = Ar(i);

    //grad3_3f(:,[3*i,3*i-2,3*i-1])=[0,-Fni(3), Fni(2);Fni(3),0,-Fni(1);-Fni(2),Fni(1),0]*e/(2*Ari);
    Eigen::Matrix<double, 3,3> n_M;
    n_M << 0,-Fni(2),Fni(1),Fni(2),0,-Fni(0),-Fni(1),Fni(0),0;
    Eigen::VectorXi R = igl::colon<int>(0,2);
    Eigen::VectorXi C(3); C  << 3*i+2,3*i,3*i+1;
    Eigen::MatrixXd res = (1./Ari)*(n_M*e);
    igl::slice_into(res,R,C,grad3_3f);
  }
  std::vector<Triplet<double> > Gx_trip,Gy_trip,Gz_trip;
  int val_idx = 0;
  for (int i = 0; i < fn; i++) {
    for (int j = 0; j < 3; j++) {
      Gx_trip.push_back(Triplet<double>(i, F(i,j), grad3_3f(0, val_idx)));
      Gy_trip.push_back(Triplet<double>(i, F(i,j), grad3_3f(1, val_idx)));
      Gz_trip.push_back(Triplet<double>(i, F(i,j), grad3_3f(2, val_idx)));
      val_idx++;
    }
  }
  SparseMatrix<double> Dx(fn,vn);  Dx.setFromTriplets(Gx_trip.begin(), Gx_trip.end());
  SparseMatrix<double> Dy(fn,vn);  Dy.setFromTriplets(Gy_trip.begin(), Gy_trip.end());
  SparseMatrix<double> Dz(fn,vn);  Dz.setFromTriplets(Gz_trip.begin(), Gz_trip.end());

  D1 = F1.col(0).asDiagonal()*Dx + F1.col(1).asDiagonal()*Dy + F1.col(2).asDiagonal()*Dz;
  D2 = F2.col(0).asDiagonal()*Dx + F2.col(1).asDiagonal()*Dy + F2.col(2).asDiagonal()*Dz;
}

void map_vertices_to_circle_area_normalized(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::VectorXi& bnd,
  Eigen::MatrixXd& UV) {
  
  Eigen::VectorXd dblArea_orig; // TODO: remove me later, waste of computations
  igl::doublearea(V,F, dblArea_orig);
  double area = dblArea_orig.sum()/2;
  double radius = sqrt(area / (M_PI));
  cout << "map_vertices_to_circle_area_normalized, area = " << area << " radius = " << radius << endl;

  // Get sorted list of boundary vertices
  std::vector<int> interior,map_ij;
  map_ij.resize(V.rows());
  interior.reserve(V.rows()-bnd.size());

  std::vector<bool> isOnBnd(V.rows(),false);
  for (int i = 0; i < bnd.size(); i++)
  {
    isOnBnd[bnd[i]] = true;
    map_ij[bnd[i]] = i;
  }

  for (int i = 0; i < (int)isOnBnd.size(); i++)
  {
    if (!isOnBnd[i])
    {
      map_ij[i] = interior.size();
      interior.push_back(i);
    }
  }

  // Map boundary to unit circle
  std::vector<double> len(bnd.size());
  len[0] = 0.;

  for (int i = 1; i < bnd.size(); i++)
  {
    len[i] = len[i-1] + (V.row(bnd[i-1]) - V.row(bnd[i])).norm();
  }
  double total_len = len[len.size()-1] + (V.row(bnd[0]) - V.row(bnd[bnd.size()-1])).norm();

  UV.resize(bnd.size(),2);
  for (int i = 0; i < bnd.size(); i++)
  {
    double frac = len[i] * (2. * M_PI) / total_len;
    UV.row(map_ij[bnd[i]]) << radius*cos(frac), radius*sin(frac);
  }

}

bool tutte_on_circle(const Eigen::MatrixXd& V,
              const Eigen::MatrixXi& F,
              Eigen::MatrixXd& uv) {
  using namespace Eigen;
  typedef Matrix<double,Dynamic,1> VectorXS;
// generate boundary conditions to a circle

  Eigen::SparseMatrix<double> A;
  igl::adjacency_matrix(F,A);

  Eigen::VectorXi b;
  igl::boundary_loop(F,b);
  Eigen::MatrixXd bc;
  map_vertices_to_circle_area_normalized(V,F,b,bc);

  
  // sum each row 
  Eigen::SparseVector<double> Asum;
  igl::sum(A,1,Asum);
  //Convert row sums into diagonal of sparse matrix
  Eigen::SparseMatrix<double> Adiag;
  igl::diag(Asum,Adiag);
  // Build uniform laplacian
  Eigen::SparseMatrix<double> Q;
  Q = Adiag - A;
  uv.resize(V.rows(),bc.cols());

  const Eigen::VectorXd B = Eigen::VectorXd::Zero(V.rows(),1);
  igl::min_quad_with_fixed_data<double> data;
  igl::min_quad_with_fixed_precompute(Q,b,Eigen::SparseMatrix<double>(),true,data);
  for(int w = 0;w<bc.cols();w++)
  {
    const Eigen::VectorXd bcw = bc.col(w);
    Eigen::VectorXd Ww;
    if(!igl::min_quad_with_fixed_solve(data,B,bcw,Eigen::VectorXd(),Ww))
    {
      return false;
    }
    uv.col(w) = Ww;
  }
  
  return true;
}

int get_euler_char(const Eigen::MatrixXd& V,
              const Eigen::MatrixXi& F) {

  int euler_v = V.rows();
  Eigen::MatrixXi EV, FE, EF;
  igl::edge_topology(V, F, EV, FE, EF);
  int euler_e = EV.rows();
  int euler_f = F.rows();
    
  int euler_char = euler_v - euler_e + euler_f;
  return euler_char;
}
