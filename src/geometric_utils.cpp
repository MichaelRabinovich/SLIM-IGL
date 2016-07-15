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

void get_flips(const Eigen::MatrixXd& V,
               const Eigen::MatrixXi& F,
               const Eigen::MatrixXd& uv,
               std::vector<int>& flip_idx) {
  flip_idx.resize(0);
  for (int i = 0; i < F.rows(); i++) {

    Eigen::Vector2d v1_n = uv.row(F(i,0)); Eigen::Vector2d v2_n = uv.row(F(i,1)); Eigen::Vector2d v3_n = uv.row(F(i,2));

    Eigen::MatrixXd T2_Homo(3,3);
    T2_Homo.col(0) << v1_n(0),v1_n(1),1;
    T2_Homo.col(1) << v2_n(0),v2_n(1),1;
    T2_Homo.col(2) << v3_n(0),v3_n(1),1;
    double det = T2_Homo.determinant();
    assert (det == det);
    if (det < 0) {
      flip_idx.push_back(i);
    }
  }
}
int count_flips(const Eigen::MatrixXd& V,
              const Eigen::MatrixXi& F,
              const Eigen::MatrixXd& uv) {

  std::vector<int> flip_idx;
  get_flips(V,F,uv,flip_idx);

  
  return flip_idx.size();
}

void compute_surface_gradient_matrix(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                                     const Eigen::MatrixXd& F1, const Eigen::MatrixXd& F2,
                                     Eigen::SparseMatrix<double>& D1, Eigen::SparseMatrix<double>& D2) {
  using namespace Eigen;
  Eigen::SparseMatrix<double> G;

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

void compute_tet_grad_matrix(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
                            Eigen::SparseMatrix<double>& Dx, Eigen::SparseMatrix<double>& Dy, Eigen::SparseMatrix<double>& Dz, bool uniform) {
  using namespace Eigen;
  cout << "compute_tet_grad_matrix" << endl;
  assert(T.cols() == 4);
  const int n = V.rows(); int m = T.rows();

  /*
      F = [ ...
      T(:,1) T(:,2) T(:,3); ...
      T(:,1) T(:,3) T(:,4); ...
      T(:,1) T(:,4) T(:,2); ...
      T(:,2) T(:,4) T(:,3)]; */
  MatrixXi F(4*m,3);
  for (int i = 0; i < m; i++) {
    F.row(0*m + i) << T(i,0), T(i,1), T(i,2);
    F.row(1*m + i) << T(i,0), T(i,2), T(i,3);
    F.row(2*m + i) << T(i,0), T(i,3), T(i,1);
    F.row(3*m + i) << T(i,1), T(i,3), T(i,2);
  }
  // compute volume of each tet
  VectorXd vol; igl::volume(V,T,vol);

  VectorXd A(F.rows());
  //N = normalizerow(normals(V,F)); switch normals and volumnes to switch gradients (keep volumes the same or switch to a constant volume)
  MatrixXd N(F.rows(),3);
  if (!uniform) {
    // compute tetrahedron face normals
    igl::per_face_normals(V,F,N); int norm_rows = N.rows();// Should we protect against degeneracy? 
    for (int i = 0; i < norm_rows; i++)
      N.row(i) /= N.row(i).norm();
    igl::doublearea(V,F,A); A/=2.;
  } else {
    // Use uniform tetrahedra:
    //      V = h*[0,0,0;1,0,0;0.5,sqrt(3)/2.,0;0.5,sqrt(3)/6.,sqrt(2)/sqrt(3)] (Base is same as for uniform 2d grad but with the appropriate 4th vertex)
    //      
    // With normals
    //         0         0    1.0000
    //         0.8165   -0.4714   -0.3333
    //         0          0.9428   -0.3333
    //         -0.8165   -0.4714   -0.3333
    for (int i = 0; i < m; i++) {
      N.row(0*m+i) << 0,0,1;
      double a = sqrt(2)*std::cbrt(3*vol(i));
      A(0*m+i) = (pow(a,2)*sqrt(3))/4.;
    }
    for (int i = 0; i < m; i++) {
      N.row(1*m+i) << 0.8165,-0.4714,-0.3333;
      double a = sqrt(2)*std::cbrt(3*vol(i));
      A(1*m+i) = (pow(a,2)*sqrt(3))/4.;
    }
    for (int i = 0; i < m; i++) {
      N.row(2*m+i) << 0,0.9428,-0.3333;
      double a = sqrt(2)*std::cbrt(3*vol(i));
      A(2*m+i) = (pow(a,2)*sqrt(3))/4.;
    }
    for (int i = 0; i < m; i++) {
      N.row(3*m+i) << -0.8165,-0.4714,-0.3333;
      double a = sqrt(2)*std::cbrt(3*vol(i));
      A(3*m+i) = (pow(a,2)*sqrt(3))/4.;
    }
    
  }

  /*  G = sparse( ...
      [0*m + repmat(1:m,1,4) ...
       1*m + repmat(1:m,1,4) ...
       2*m + repmat(1:m,1,4)], ...
      repmat([T(:,4);T(:,2);T(:,3);T(:,1)],3,1), ...
      repmat(A./(3*repmat(vol,4,1)),3,1).*N(:), ...
      3*m,n);*/
  std::vector<Triplet<double> > Dx_t,Dy_t,Dz_t;
  for (int i = 0; i < 4*m; i++) {
    int T_j; // j indexes : repmat([T(:,4);T(:,2);T(:,3);T(:,1)],3,1)
    switch (i/m) {
      case 0:
        T_j = 3;
        break;
      case 1:
        T_j = 1;
        break;
      case 2:
        T_j = 2;
        break;
      case 3:
        T_j = 0;
        break;
      default:
        assert(1<0); // should not get here
    }
    int i_idx = i%m;
    int j_idx = T(i_idx,T_j);

    double val_before_n = A(i)/(3*vol(i_idx));
    Dx_t.push_back(Triplet<double>(i_idx, j_idx, val_before_n * N(i,0)));
    Dy_t.push_back(Triplet<double>(i_idx, j_idx, val_before_n * N(i,1)));
    Dz_t.push_back(Triplet<double>(i_idx, j_idx, val_before_n * N(i,2)));
  }
  Dx.resize(m,n); Dy.resize(m,n); Dz.resize(m,n);
  Dx.setFromTriplets(Dx_t.begin(), Dx_t.end());
  Dy.setFromTriplets(Dy_t.begin(), Dy_t.end());
  Dz.setFromTriplets(Dz_t.begin(), Dz_t.end());

}