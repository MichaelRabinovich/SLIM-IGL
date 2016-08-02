#include "Slim.h"

#include <igl/boundary_loop.h>
#include <igl/cotmatrix.h>
#include <igl/edge_lengths.h>
#include <igl/local_basis.h>
#include <igl/readOBJ.h>
#include <igl/repdiag.h>
#include <igl/vector_area_matrix.h>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>
#include <set>
#include <vector>

#include "igl/arap.h"
#include "igl/cat.h"
#include "igl/doublearea.h"
#include "igl/grad.h"
#include "igl/local_basis.h"
#include "igl/per_face_normals.h"
#include "igl/slice_into.h"

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>

using namespace std;
using namespace Eigen;

///////// Helper functions to compute gradient matrices

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




// Computes the weights and solve the linear system for the quadratic proxy specified in the paper
// The output of this is used to generate a search direction that will be fed to the Linesearch class
class WeightedGlobalLocal {

public:
  WeightedGlobalLocal(SLIMData& state);

  // Compute necessary information before solving the proxy quadratic
  void pre_calc();

  // Solve the weighted proxy global step
  // Output:
  //    V_new #V by dim list of mesh positions (will be fed to a linesearch algorithm)
  void solve_weighted_proxy(Eigen::MatrixXd& V_new);

  // Compute the energy specified in the SLIMData structure + the soft constraint energy (in case there are soft constraints)
  // Input:
  //    V_new #V by dim list of mesh positions
  virtual double compute_energy(Eigen::MatrixXd& V_new);

private:

  void compute_jacobians(const Eigen::MatrixXd& V_o);
  double compute_energy_with_jacobians(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& Ji, Eigen::MatrixXd& V_o, Eigen::VectorXd& areas);
  double compute_soft_const_energy(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                                             Eigen::MatrixXd& V_o);

  void update_weights_and_closest_rotations(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& uv);
  void solve_weighted_arap(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& uv, Eigen::VectorXi& b,
      Eigen::MatrixXd& bc);

  void build_linear_system(Eigen::SparseMatrix<double> &L);
  void buildA(Eigen::SparseMatrix<double>& A);
  void buildRhs(const Eigen::SparseMatrix<double>& At);

  void add_soft_constraints(Eigen::SparseMatrix<double> &L);
  void add_proximal_penalty();

  SLIMData& m_state;
  Eigen::VectorXd M;
  Eigen::VectorXd rhs;
  Eigen::MatrixXd Ri,Ji;
  Eigen::VectorXd W_11; Eigen::VectorXd W_12; Eigen::VectorXd W_13;
  Eigen::VectorXd W_21; Eigen::VectorXd W_22; Eigen::VectorXd W_23;
  Eigen::VectorXd W_31; Eigen::VectorXd W_32; Eigen::VectorXd W_33;
  Eigen::SparseMatrix<double> Dx,Dy,Dz;

  int f_n,v_n;

  bool first_solve;
  bool has_pre_calc = false;

  int dim;
};


class Linesearch {

public:
  Linesearch(SLIMData& param_state);

  // A simple backtracking linesearch
  // Input:
  //    V #V by 3 list of mesh positions (original mesh)
  //    F #F by simplex-size list of triangle|tet indices into V
  //    cur_v #V by dim list of the current mesh positions
  //    dst_v #V by dim list the destination mesh positions (d = cur_v - dst_v)
  //    energy A class used to evaluate the current objective value
  //    cur_energy (OPTIONAL) The current objective funcational at cur_v
  //
  // Output:
  //    cur_v A new set of vertices such that energy(cur_v) < previous energy
  //    Returns the new objective value
  double compute( const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    Eigen::MatrixXd& cur_v,
    Eigen::MatrixXd& dst_v,
    WeightedGlobalLocal* energy,
    double cur_energy = -1);

private:

  double compute_max_step_from_singularities(const Eigen::MatrixXd& uv,
                                            const Eigen::MatrixXi& F,
                                            Eigen::MatrixXd& d);

  double line_search(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                              Eigen::MatrixXd& uv, const Eigen::MatrixXd& d,
                              double step_size, WeightedGlobalLocal* energy, double cur_energy);

  double get_min_pos_root_2D(const Eigen::MatrixXd& uv,const Eigen::MatrixXi& F,
            Eigen::MatrixXd& direc, int f);

  double get_min_pos_root_3D(const Eigen::MatrixXd& uv,const Eigen::MatrixXi& F,
            Eigen::MatrixXd& direc, int f);

  double get_smallest_pos_quad_zero(double a,double b, double c);
  int SolveP3(std::vector<double>& x,double a,double b,double c);

  SLIMData& m_state;
};




//// Implementation

WeightedGlobalLocal::WeightedGlobalLocal(SLIMData& state) :
                                  m_state(state) {
}

void WeightedGlobalLocal::solve_weighted_proxy(Eigen::MatrixXd& V_new) {

  update_weights_and_closest_rotations(m_state.V,m_state.F,V_new);
  solve_weighted_arap(m_state.V,m_state.F,V_new,m_state.b,m_state.bc);
}

void WeightedGlobalLocal::compute_jacobians(const Eigen::MatrixXd& uv) {
  if (m_state.F.cols() == 3){
    // Ji=[D1*u,D2*u,D1*v,D2*v];
    Ji.col(0) = Dx*uv.col(0); Ji.col(1) = Dy*uv.col(0);
    Ji.col(2) = Dx*uv.col(1); Ji.col(3) = Dy*uv.col(1);
  } else /*tet mesh*/{
    // Ji=[D1*u,D2*u,D3*u, D1*v,D2*v, D3*v, D1*w,D2*w,D3*w];
    Ji.col(0) = Dx*uv.col(0); Ji.col(1) = Dy*uv.col(0); Ji.col(2) = Dz*uv.col(0);
    Ji.col(3) = Dx*uv.col(1); Ji.col(4) = Dy*uv.col(1); Ji.col(5) = Dz*uv.col(1);
    Ji.col(6) = Dx*uv.col(2); Ji.col(7) = Dy*uv.col(2); Ji.col(8) = Dz*uv.col(2);
  }
}

void WeightedGlobalLocal::update_weights_and_closest_rotations(const Eigen::MatrixXd& V,
       const Eigen::MatrixXi& F, Eigen::MatrixXd& uv) {
  compute_jacobians(uv);

  const double eps = 1e-8;
  double exp_f = m_state.exp_factor;

  if (dim==2) {
    for(int i=0; i <Ji.rows(); ++i ) {
    typedef Eigen::Matrix<double,2,2> Mat2;
    typedef Eigen::Matrix<double,2,1> Vec2;
    Mat2 ji,ri,ti,ui,vi; Vec2 sing; Vec2 closest_sing_vec;Mat2 mat_W;
    Vec2 m_sing_new;
    double s1,s2;

    ji(0,0) = Ji(i,0); ji(0,1) = Ji(i,1);
    ji(1,0) = Ji(i,2); ji(1,1) = Ji(i,3);

    igl::polar_svd(ji,ri,ti,ui,sing,vi);

    s1 = sing(0); s2 = sing(1);

    // Update Weights according to energy
    switch(m_state.slim_energy) {
    case SLIMData::ARAP: {
      m_sing_new << 1,1;
      break;
    } case SLIMData::SYMMETRIC_DIRICHLET: {
        double s1_g = 2* (s1-pow(s1,-3));
        double s2_g = 2 * (s2-pow(s2,-3));
        m_sing_new << sqrt(s1_g/(2*(s1-1))), sqrt(s2_g/(2*(s2-1)));
        break;
    } case SLIMData::LOG_ARAP: {
        double s1_g = 2 * (log(s1)/s1);
        double s2_g = 2 * (log(s2)/s2);
        m_sing_new << sqrt(s1_g/(2*(s1-1))), sqrt(s2_g/(2*(s2-1)));
        break;
    } case SLIMData::CONFORMAL: {
        double s1_g = 1/(2*s2) - s2/(2*pow(s1,2));
        double s2_g = 1/(2*s1) - s1/(2*pow(s2,2));

        double geo_avg = sqrt(s1*s2);
        double s1_min = geo_avg; double s2_min = geo_avg;

        m_sing_new << sqrt(s1_g/(2*(s1-s1_min))), sqrt(s2_g/(2*(s2-s2_min)));

        // change local step
        closest_sing_vec << s1_min,s2_min;
        ri = ui*closest_sing_vec.asDiagonal()*vi.transpose();
        break;
    } case SLIMData::EXP_CONFORMAL: {
        double s1_g = 2* (s1-pow(s1,-3));
        double s2_g = 2 * (s2-pow(s2,-3));

        double geo_avg = sqrt(s1*s2);
        double s1_min = geo_avg; double s2_min = geo_avg;

        double in_exp = exp_f*((pow(s1,2)+pow(s2,2))/(2*s1*s2));
        double exp_thing = exp(in_exp);

        s1_g *= exp_thing*exp_f;
        s2_g *= exp_thing*exp_f;

        m_sing_new << sqrt(s1_g/(2*(s1-1))), sqrt(s2_g/(2*(s2-1)));
        break;
    } case SLIMData::EXP_SYMMETRIC_DIRICHLET: {
        double s1_g = 2* (s1-pow(s1,-3));
        double s2_g = 2 * (s2-pow(s2,-3));

        double in_exp = exp_f*(pow(s1,2)+pow(s1,-2)+pow(s2,2)+pow(s2,-2));
        double exp_thing = exp(in_exp);

        s1_g *= exp_thing*exp_f;
        s2_g *= exp_thing*exp_f;

        m_sing_new << sqrt(s1_g/(2*(s1-1))), sqrt(s2_g/(2*(s2-1)));
        break;
      }
    }

    if (abs(s1-1) < eps) m_sing_new(0) = 1; if (abs(s2-1) < eps) m_sing_new(1) = 1;
    mat_W = ui*m_sing_new.asDiagonal()*ui.transpose();

    W_11(i) = mat_W(0,0); W_12(i) = mat_W(0,1); W_21(i) = mat_W(1,0); W_22(i) = mat_W(1,1);

    // 2) Update local step (doesn't have to be a rotation, for instance in case of conformal energy)
    Ri(i,0) = ri(0,0); Ri(i,1) = ri(1,0); Ri(i,2) = ri(0,1); Ri(i,3) = ri(1,1);
   }
  } else {
    typedef Eigen::Matrix<double,3,1> Vec3; typedef Eigen::Matrix<double,3,3> Mat3;
    Mat3 ji; Vec3 m_sing_new; Vec3 closest_sing_vec;
    const double sqrt_2 = sqrt(2);
    for(int i=0; i <Ji.rows(); ++i ) {
      ji(0,0) = Ji(i,0); ji(0,1) = Ji(i,1); ji(0,2) = Ji(i,2);
      ji(1,0) = Ji(i,3); ji(1,1) = Ji(i,4); ji(1,2) = Ji(i,5);
      ji(2,0) = Ji(i,6); ji(2,1) = Ji(i,7); ji(2,2) = Ji(i,8);

      Mat3 ri,ti,ui,vi;
      Vec3 sing;
      igl::polar_svd(ji,ri,ti,ui,sing,vi);

      double s1 = sing(0); double s2 = sing(1); double s3 = sing(2);

      // 1) Update Weights
      switch(m_state.slim_energy) {
        case SLIMData::ARAP: {
          m_sing_new << 1,1,1;
          break;
        } case SLIMData::LOG_ARAP: {
            double s1_g = 2 * (log(s1)/s1);
            double s2_g = 2 * (log(s2)/s2);
            double s3_g = 2 * (log(s3)/s3);
            m_sing_new << sqrt(s1_g/(2*(s1-1))), sqrt(s2_g/(2*(s2-1))), sqrt(s3_g/(2*(s3-1)));
            break;
          } case SLIMData::SYMMETRIC_DIRICHLET: {
            double s1_g = 2* (s1-pow(s1,-3));
            double s2_g = 2 * (s2-pow(s2,-3));
            double s3_g = 2 * (s3-pow(s3,-3));
            m_sing_new << sqrt(s1_g/(2*(s1-1))), sqrt(s2_g/(2*(s2-1))), sqrt(s3_g/(2*(s3-1)));
            break;
          } case SLIMData::EXP_SYMMETRIC_DIRICHLET: {
           double s1_g = 2* (s1-pow(s1,-3));
          double s2_g = 2 * (s2-pow(s2,-3));
          double s3_g = 2 * (s3-pow(s3,-3));
          m_sing_new << sqrt(s1_g/(2*(s1-1))), sqrt(s2_g/(2*(s2-1))), sqrt(s3_g/(2*(s3-1)));

          double in_exp = exp_f*(pow(s1,2)+pow(s1,-2)+pow(s2,2)+pow(s2,-2)+pow(s3,2)+pow(s3,-2));
          double exp_thing = exp(in_exp);

          s1_g *= exp_thing*exp_f;
          s2_g *= exp_thing*exp_f;
          s3_g *= exp_thing*exp_f;

          m_sing_new << sqrt(s1_g/(2*(s1-1))), sqrt(s2_g/(2*(s2-1))), sqrt(s3_g/(2*(s3-1)));

          break;
        }
        case SLIMData::CONFORMAL: {
          double common_div = 9*(pow(s1*s2*s3,5./3.));

          double s1_g = (-2*s2*s3*(pow(s2,2)+pow(s3,2)-2*pow(s1,2)) ) / common_div;
          double s2_g = (-2*s1*s3*(pow(s1,2)+pow(s3,2)-2*pow(s2,2)) ) / common_div;
          double s3_g = (-2*s1*s2*(pow(s1,2)+pow(s2,2)-2*pow(s3,2)) ) / common_div;

          double closest_s = sqrt(pow(s1,2)+pow(s3,2)) / sqrt_2;
          double s1_min = closest_s; double s2_min = closest_s; double s3_min = closest_s;

          m_sing_new << sqrt(s1_g/(2*(s1-s1_min))), sqrt(s2_g/(2*(s2-s2_min))), sqrt(s3_g/(2*(s3-s3_min)));

          // change local step
          closest_sing_vec << s1_min,s2_min,s3_min;
          ri = ui*closest_sing_vec.asDiagonal()*vi.transpose();
          break;
        }
        case SLIMData::EXP_CONFORMAL: {
          // E_conf = (s1^2 + s2^2 + s3^2)/(3*(s1*s2*s3)^(2/3) )
          // dE_conf/ds1 = (-2*(s2*s3)*(s2^2+s3^2 -2*s1^2) ) / (9*(s1*s2*s3)^(5/3))
          // Argmin E_conf(s1): s1 = sqrt(s1^2+s2^2)/sqrt(2)
          double common_div = 9*(pow(s1*s2*s3,5./3.));

          double s1_g = (-2*s2*s3*(pow(s2,2)+pow(s3,2)-2*pow(s1,2)) ) / common_div;
          double s2_g = (-2*s1*s3*(pow(s1,2)+pow(s3,2)-2*pow(s2,2)) ) / common_div;
          double s3_g = (-2*s1*s2*(pow(s1,2)+pow(s2,2)-2*pow(s3,2)) ) / common_div;

          double in_exp = exp_f*( (pow(s1,2)+pow(s2,2)+pow(s3,2))/ (3*pow((s1*s2*s3),2./3)) ); ;
          double exp_thing = exp(in_exp);

          double closest_s = sqrt(pow(s1,2)+pow(s3,2)) / sqrt_2;
          double s1_min = closest_s; double s2_min = closest_s; double s3_min = closest_s;

          s1_g *= exp_thing*exp_f;
          s2_g *= exp_thing*exp_f;
          s3_g *= exp_thing*exp_f;

          m_sing_new << sqrt(s1_g/(2*(s1-s1_min))), sqrt(s2_g/(2*(s2-s2_min))), sqrt(s3_g/(2*(s3-s3_min)));

          // change local step
          closest_sing_vec << s1_min,s2_min,s3_min;
          ri = ui*closest_sing_vec.asDiagonal()*vi.transpose();
        }
      }
      if (abs(s1-1) < eps) m_sing_new(0) = 1; if (abs(s2-1) < eps) m_sing_new(1) = 1; if (abs(s3-1) < eps) m_sing_new(2) = 1;
      Mat3 mat_W;
      mat_W = ui*m_sing_new.asDiagonal()*ui.transpose();

      W_11(i) = mat_W(0,0);
      W_12(i) = mat_W(0,1);
      W_13(i) = mat_W(0,2);
      W_21(i) = mat_W(1,0);
      W_22(i) = mat_W(1,1);
      W_23(i) = mat_W(1,2);
      W_31(i) = mat_W(2,0);
      W_32(i) = mat_W(2,1);
      W_33(i) = mat_W(2,2);

      // 2) Update closest rotations (not rotations in case of conformal energy)
      Ri(i,0) = ri(0,0); Ri(i,1) = ri(1,0); Ri(i,2) = ri(2,0);
      Ri(i,3) = ri(0,1); Ri(i,4) = ri(1,1); Ri(i,5) = ri(2,1);
      Ri(i,6) = ri(0,2); Ri(i,7) = ri(1,2); Ri(i,8) = ri(2,2);
    } // for loop end

  } // if dim end

}

void WeightedGlobalLocal::solve_weighted_arap(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
        Eigen::MatrixXd& uv, Eigen::VectorXi& soft_b_p, Eigen::MatrixXd& soft_bc_p) {
  using namespace Eigen;

  Eigen::SparseMatrix<double> L;
  build_linear_system(L);

  // solve
  Eigen::VectorXd Uc;
  if (dim == 2) {
    SimplicialLDLT<SparseMatrix<double> > solver;
    Uc = solver.compute(L).solve(rhs);
  } else { // seems like CG performs much worse for 2D and way better for 3D
    Eigen::VectorXd guess(uv.rows()*dim);
    for (int i = 0; i < dim; i++) for (int j = 0; j < dim; j++) guess(uv.rows()*i + j) = uv(i,j); // flatten vector
    ConjugateGradient<SparseMatrix<double>, Eigen::Upper> solver;
    solver.setTolerance(1e-8);
    Uc = solver.compute(L).solveWithGuess(rhs,guess);
  }

  for (int i = 0; i < dim; i++)
    uv.col(i) = Uc.block(i*v_n,0,v_n,1);
}

void WeightedGlobalLocal::pre_calc() {
  if (!has_pre_calc) {
    v_n = m_state.v_num; f_n = m_state.f_num;

    if (m_state.F.cols() == 3) {
      dim = 2;
      Eigen::MatrixXd F1,F2,F3;
      igl::local_basis(m_state.V,m_state.F,F1,F2,F3);
      compute_surface_gradient_matrix(m_state.V,m_state.F,F1,F2,Dx,Dy);

      W_11.resize(f_n); W_12.resize(f_n); W_21.resize(f_n); W_22.resize(f_n);
    } else {
      dim = 3;
      compute_tet_grad_matrix(m_state.V,m_state.F,Dx,Dy,Dz,
        m_state.mesh_improvement_3d /*use normal gradient, or one from a "regular" tet*/);

      W_11.resize(f_n);W_12.resize(f_n);W_13.resize(f_n);
      W_21.resize(f_n);W_22.resize(f_n);W_23.resize(f_n);
      W_31.resize(f_n);W_32.resize(f_n);W_33.resize(f_n);
    }

    Dx.makeCompressed();Dy.makeCompressed(); Dz.makeCompressed();
    Ri.resize(f_n, dim*dim); Ji.resize(f_n, dim*dim);
    rhs.resize(dim*m_state.v_num);

    // flattened weight matrix
    M.resize(dim*dim*f_n);
    for (int i = 0; i < dim*dim; i++)
      for (int j = 0; j < f_n; j++)
        M(i*f_n + j) = m_state.M(j);

    first_solve = true;
    has_pre_calc = true;
  }
}

void WeightedGlobalLocal::build_linear_system(Eigen::SparseMatrix<double> &L) {
  // formula (35) in paper
  Eigen::SparseMatrix<double> A(dim*dim*f_n, dim*v_n);
  buildA(A);

  Eigen::SparseMatrix<double> At = A.transpose();
  At.makeCompressed();

  Eigen::SparseMatrix<double> id_m(At.rows(),At.rows()); id_m.setIdentity();

  // add proximal penalty
  L = At*M.asDiagonal()*A + m_state.proximal_p * id_m; //add also a proximal term
  L.makeCompressed();

  buildRhs(At);
  Eigen::SparseMatrix<double> OldL = L;
  add_soft_constraints(L);
  L.makeCompressed();
}

void WeightedGlobalLocal::add_soft_constraints(Eigen::SparseMatrix<double> &L) {
  int v_n = m_state.v_num;
  for (int d = 0; d < dim; d++) {
    for (int i = 0; i < m_state.b.rows(); i++) {
      int v_idx = m_state.b(i);
      rhs(d*v_n + v_idx) += m_state.soft_const_p * m_state.bc(i,d); // rhs
      L.coeffRef(d*v_n + v_idx, d*v_n + v_idx) += m_state.soft_const_p; // diagonal of matrix
    }
  }
}

double WeightedGlobalLocal::compute_energy(Eigen::MatrixXd& V_new) {
  compute_jacobians(V_new);
  return compute_energy_with_jacobians(m_state.V,m_state.F, Ji, V_new,m_state.M) + compute_soft_const_energy(m_state.V,m_state.F,V_new);
}

double WeightedGlobalLocal::compute_soft_const_energy(const Eigen::MatrixXd& V,
                                                       const Eigen::MatrixXi& F,
                                                       Eigen::MatrixXd& V_o) {
  double e = 0;
  for (int i = 0; i < m_state.b.rows(); i++) {
    e += m_state.soft_const_p*(m_state.bc.row(i)-V_o.row(m_state.b(i))).squaredNorm();
  }
  return e;
}

double WeightedGlobalLocal::compute_energy_with_jacobians(const Eigen::MatrixXd& V,
       const Eigen::MatrixXi& F, const Eigen::MatrixXd& Ji, Eigen::MatrixXd& uv, Eigen::VectorXd& areas) {

  double energy = 0;
  if (dim == 2) {
    Eigen::Matrix<double,2,2> ji;
    for (int i = 0; i < f_n; i++) {
      ji(0,0) = Ji(i,0); ji(0,1) = Ji(i,1);
      ji(1,0) = Ji(i,2); ji(1,1) = Ji(i,3);

      typedef Eigen::Matrix<double,2,2> Mat2;
      typedef Eigen::Matrix<double,2,1> Vec2;
      Mat2 ri,ti,ui,vi; Vec2 sing;
      igl::polar_svd(ji,ri,ti,ui,sing,vi);
      double s1 = sing(0); double s2 = sing(1);

      switch(m_state.slim_energy) {
        case SLIMData::ARAP: {
          energy+= areas(i) * (pow(s1-1,2) + pow(s2-1,2));
          break;
        }
        case SLIMData::SYMMETRIC_DIRICHLET: {
          energy += areas(i) * (pow(s1,2) +pow(s1,-2) + pow(s2,2) + pow(s2,-2));
          break;
        }
        case SLIMData::EXP_SYMMETRIC_DIRICHLET: {
          energy += areas(i) * exp(m_state.exp_factor*(pow(s1,2) +pow(s1,-2) + pow(s2,2) + pow(s2,-2)));
          break;
        }
        case SLIMData::LOG_ARAP: {
          energy += areas(i) * (pow(log(s1),2) + pow(log(s2),2));
          break;
        }
        case SLIMData::CONFORMAL: {
          energy += areas(i) * ( (pow(s1,2)+pow(s2,2))/(2*s1*s2) );
          break;
        }
        case SLIMData::EXP_CONFORMAL: {
          energy += areas(i) * exp(m_state.exp_factor*((pow(s1,2)+pow(s2,2))/(2*s1*s2)));
          break;
        }

      }

    }
  } else {
    Eigen::Matrix<double,3,3> ji;
    for (int i = 0; i < f_n; i++) {
      ji(0,0) = Ji(i,0); ji(0,1) = Ji(i,1); ji(0,2) = Ji(i,2);
      ji(1,0) = Ji(i,3); ji(1,1) = Ji(i,4); ji(1,2) = Ji(i,5);
      ji(2,0) = Ji(i,6); ji(2,1) = Ji(i,7); ji(2,2) = Ji(i,8);

      typedef Eigen::Matrix<double,3,3> Mat3;
      typedef Eigen::Matrix<double,3,1> Vec3;
      Mat3 ri,ti,ui,vi; Vec3 sing;
      igl::polar_svd(ji,ri,ti,ui,sing,vi);
      double s1 = sing(0); double s2 = sing(1); double s3 = sing(2);

      switch(m_state.slim_energy) {
        case SLIMData::ARAP: {
          energy+= areas(i) * (pow(s1-1,2) + pow(s2-1,2) + pow(s3-1,2));
          break;
        }
        case SLIMData::SYMMETRIC_DIRICHLET: {
          energy += areas(i) * (pow(s1,2) +pow(s1,-2) + pow(s2,2) + pow(s2,-2) + pow(s3,2) + pow(s3,-2));
          break;
        }
        case SLIMData::EXP_SYMMETRIC_DIRICHLET: {
          energy += areas(i) * exp(m_state.exp_factor*(pow(s1,2) +pow(s1,-2) + pow(s2,2) + pow(s2,-2) + pow(s3,2) + pow(s3,-2)));
          break;
        }
        case SLIMData::LOG_ARAP: {
          energy += areas(i) * (pow(log(s1),2) + pow(log(abs(s2)),2) + pow(log(abs(s3)),2));
          break;
        }
        case SLIMData::CONFORMAL: {
          energy += areas(i) * ( ( pow(s1,2)+pow(s2,2)+pow(s3,2) ) /(3*pow(s1*s2*s3,2./3.)) );
          break;
        }
        case SLIMData::EXP_CONFORMAL: {
          energy += areas(i) * exp( ( pow(s1,2)+pow(s2,2)+pow(s3,2) ) /(3*pow(s1*s2*s3,2./3.)) );
          break;
        }
      }
    }
  }

  return energy;
}

void WeightedGlobalLocal::buildA(Eigen::SparseMatrix<double>& A) {
  // formula (35) in paper
  std::vector<Triplet<double> > IJV;
  if (dim == 2) {
    IJV.reserve(4*(Dx.outerSize()+ Dy.outerSize()));

    /*A = [W11*Dx, W12*Dx;
         W11*Dy, W12*Dy;
         W21*Dx, W22*Dx;
         W21*Dy, W22*Dy];*/
    for (int k=0; k<Dx.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(Dx,k); it; ++it) {
          int dx_r = it.row();
          int dx_c = it.col();
          double val = it.value();

          IJV.push_back(Triplet<double>(dx_r,dx_c, val*W_11(dx_r)));
          IJV.push_back(Triplet<double>(dx_r,v_n + dx_c, val*W_12(dx_r)));

          IJV.push_back(Triplet<double>(2*f_n+dx_r,dx_c, val*W_21(dx_r)));
          IJV.push_back(Triplet<double>(2*f_n+dx_r,v_n + dx_c, val*W_22(dx_r)));
      }
    }

    for (int k=0; k<Dy.outerSize(); ++k) {
      for (SparseMatrix<double>::InnerIterator it(Dy,k); it; ++it) {
        int dy_r = it.row();
        int dy_c = it.col();
        double val = it.value();

        IJV.push_back(Triplet<double>(f_n+dy_r,dy_c, val*W_11(dy_r)));
        IJV.push_back(Triplet<double>(f_n+dy_r,v_n + dy_c, val*W_12(dy_r)));

        IJV.push_back(Triplet<double>(3*f_n+dy_r,dy_c, val*W_21(dy_r)));
        IJV.push_back(Triplet<double>(3*f_n+dy_r,v_n + dy_c, val*W_22(dy_r)));
      }
    }
  } else {

    /*A = [W11*Dx, W12*Dx, W13*Dx;
           W11*Dy, W12*Dy, W13*Dy;
           W11*Dz, W12*Dz, W13*Dz;
           W21*Dx, W22*Dx, W23*Dx;
           W21*Dy, W22*Dy, W23*Dy;
           W21*Dz, W22*Dz, W23*Dz;
           W31*Dx, W32*Dx, W33*Dx;
           W31*Dy, W32*Dy, W33*Dy;
           W31*Dz, W32*Dz, W33*Dz;];*/
    IJV.reserve(9*(Dx.outerSize()+ Dy.outerSize() + Dz.outerSize()));
    for (int k = 0; k < Dx.outerSize(); k++) {
      for (SparseMatrix<double>::InnerIterator it(Dx,k); it; ++it) {
         int dx_r = it.row();
         int dx_c = it.col();
         double val = it.value();

         IJV.push_back(Triplet<double>(dx_r,dx_c, val*W_11(dx_r)));
         IJV.push_back(Triplet<double>(dx_r,v_n + dx_c, val*W_12(dx_r)));
         IJV.push_back(Triplet<double>(dx_r,2*v_n + dx_c, val*W_13(dx_r)));

         IJV.push_back(Triplet<double>(3*f_n+dx_r,dx_c, val*W_21(dx_r)));
         IJV.push_back(Triplet<double>(3*f_n+dx_r,v_n + dx_c, val*W_22(dx_r)));
         IJV.push_back(Triplet<double>(3*f_n+dx_r,2*v_n + dx_c, val*W_23(dx_r)));

         IJV.push_back(Triplet<double>(6*f_n+dx_r,dx_c, val*W_31(dx_r)));
         IJV.push_back(Triplet<double>(6*f_n+dx_r,v_n + dx_c, val*W_32(dx_r)));
         IJV.push_back(Triplet<double>(6*f_n+dx_r,2*v_n + dx_c, val*W_33(dx_r)));
      }
    }

    for (int k = 0; k < Dy.outerSize(); k++) {
      for (SparseMatrix<double>::InnerIterator it(Dy,k); it; ++it) {
         int dy_r = it.row();
         int dy_c = it.col();
         double val = it.value();

         IJV.push_back(Triplet<double>(f_n+dy_r,dy_c, val*W_11(dy_r)));
         IJV.push_back(Triplet<double>(f_n+dy_r,v_n + dy_c, val*W_12(dy_r)));
         IJV.push_back(Triplet<double>(f_n+dy_r,2*v_n + dy_c, val*W_13(dy_r)));

         IJV.push_back(Triplet<double>(4*f_n+dy_r,dy_c, val*W_21(dy_r)));
         IJV.push_back(Triplet<double>(4*f_n+dy_r,v_n + dy_c, val*W_22(dy_r)));
         IJV.push_back(Triplet<double>(4*f_n+dy_r,2*v_n + dy_c, val*W_23(dy_r)));

         IJV.push_back(Triplet<double>(7*f_n+dy_r,dy_c, val*W_31(dy_r)));
         IJV.push_back(Triplet<double>(7*f_n+dy_r,v_n + dy_c, val*W_32(dy_r)));
         IJV.push_back(Triplet<double>(7*f_n+dy_r,2*v_n + dy_c, val*W_33(dy_r)));
      }
    }

    for (int k = 0; k < Dz.outerSize(); k++) {
      for (SparseMatrix<double>::InnerIterator it(Dz,k); it; ++it) {
         int dz_r = it.row();
         int dz_c = it.col();
         double val = it.value();

         IJV.push_back(Triplet<double>(2*f_n + dz_r,dz_c, val*W_11(dz_r)));
         IJV.push_back(Triplet<double>(2*f_n + dz_r,v_n + dz_c, val*W_12(dz_r)));
         IJV.push_back(Triplet<double>(2*f_n + dz_r,2*v_n + dz_c, val*W_13(dz_r)));

         IJV.push_back(Triplet<double>(5*f_n+dz_r,dz_c, val*W_21(dz_r)));
         IJV.push_back(Triplet<double>(5*f_n+dz_r,v_n + dz_c, val*W_22(dz_r)));
         IJV.push_back(Triplet<double>(5*f_n+dz_r,2*v_n + dz_c, val*W_23(dz_r)));

         IJV.push_back(Triplet<double>(8*f_n+dz_r,dz_c, val*W_31(dz_r)));
         IJV.push_back(Triplet<double>(8*f_n+dz_r,v_n + dz_c, val*W_32(dz_r)));
         IJV.push_back(Triplet<double>(8*f_n+dz_r,2*v_n + dz_c, val*W_33(dz_r)));
      }
    }
  }
  A.setFromTriplets(IJV.begin(),IJV.end());
}

void WeightedGlobalLocal::buildRhs(const Eigen::SparseMatrix<double>& At) {
  VectorXd f_rhs(dim*dim*f_n); f_rhs.setZero();
  if (dim==2) {
    /*b = [W11*R11 + W12*R21; (formula (36))
         W11*R12 + W12*R22;
         W21*R11 + W22*R21;
         W21*R12 + W22*R22];*/
    for (int i = 0; i < f_n; i++) {
      f_rhs(i+0*f_n) = W_11(i) * Ri(i,0) + W_12(i)*Ri(i,1);
      f_rhs(i+1*f_n) = W_11(i) * Ri(i,2) + W_12(i)*Ri(i,3);
      f_rhs(i+2*f_n) = W_21(i) * Ri(i,0) + W_22(i)*Ri(i,1);
      f_rhs(i+3*f_n) = W_21(i) * Ri(i,2) + W_22(i)*Ri(i,3);
    }
  } else {
    /*b = [W11*R11 + W12*R21 + W13*R31;
         W11*R12 + W12*R22 + W13*R32;
         W11*R13 + W12*R23 + W13*R33;
         W21*R11 + W22*R21 + W23*R31;
         W21*R12 + W22*R22 + W23*R32;
         W21*R13 + W22*R23 + W23*R33;
         W31*R11 + W32*R21 + W33*R31;
         W31*R12 + W32*R22 + W33*R32;
         W31*R13 + W32*R23 + W33*R33;];*/
    for (int i = 0; i < f_n; i++) {
      f_rhs(i+0*f_n) = W_11(i) * Ri(i,0) + W_12(i)*Ri(i,1) + W_13(i)*Ri(i,2);
      f_rhs(i+1*f_n) = W_11(i) * Ri(i,3) + W_12(i)*Ri(i,4) + W_13(i)*Ri(i,5);
      f_rhs(i+2*f_n) = W_11(i) * Ri(i,6) + W_12(i)*Ri(i,7) + W_13(i)*Ri(i,8);
      f_rhs(i+3*f_n) = W_21(i) * Ri(i,0) + W_22(i)*Ri(i,1) + W_23(i)*Ri(i,2);
      f_rhs(i+4*f_n) = W_21(i) * Ri(i,3) + W_22(i)*Ri(i,4) + W_23(i)*Ri(i,5);
      f_rhs(i+5*f_n) = W_21(i) * Ri(i,6) + W_22(i)*Ri(i,7) + W_23(i)*Ri(i,8);
      f_rhs(i+6*f_n) = W_31(i) * Ri(i,0) + W_32(i)*Ri(i,1) + W_33(i)*Ri(i,2);
      f_rhs(i+7*f_n) = W_31(i) * Ri(i,3) + W_32(i)*Ri(i,4) + W_33(i)*Ri(i,5);
      f_rhs(i+8*f_n) = W_31(i) * Ri(i,6) + W_32(i)*Ri(i,7) + W_33(i)*Ri(i,8);
    }
  }
  VectorXd uv_flat(dim*v_n);
  for (int i = 0; i < dim; i++)
    for (int j = 0; j < v_n; j++)
      uv_flat(v_n*i+j) = m_state.V_o(j,i);

  rhs = (At*M.asDiagonal()*f_rhs + m_state.proximal_p * uv_flat);
}

/////// Implementation of Linesearch

Linesearch::Linesearch (SLIMData& param_state) : m_state(param_state) {
  // empty
}

double Linesearch::compute( const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
    Eigen::MatrixXd& cur_v, Eigen::MatrixXd& dst_v, WeightedGlobalLocal* energy, double cur_energy) {

    Eigen::MatrixXd d = dst_v - cur_v;

    double min_step_to_singularity = compute_max_step_from_singularities(cur_v,F,d);
    double max_step_size = min(1., min_step_to_singularity*0.8);

    return line_search(V,F,cur_v,d,max_step_size, energy, cur_energy);
}

double Linesearch::line_search(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                              Eigen::MatrixXd& uv, const Eigen::MatrixXd& d,
                              double step_size, WeightedGlobalLocal* energy, double cur_energy) {
  double old_energy;
  if (cur_energy > 0) {
    old_energy = cur_energy;
  } else {
    old_energy = energy->compute_energy(uv); // no energy was given -> need to compute the current energy
  }
  double new_energy = old_energy;
  int cur_iter = 0; int MAX_STEP_SIZE_ITER = 12;

  while (new_energy >= old_energy && cur_iter < MAX_STEP_SIZE_ITER) {
    Eigen::MatrixXd new_uv = uv + step_size * d;

    double cur_e = energy->compute_energy(new_uv);
    if ( cur_e >= old_energy) {
      step_size /= 2;
    } else {
      uv = new_uv;
      new_energy = cur_e;
    }
    cur_iter++;
  }
  return new_energy;
}

 double Linesearch::compute_max_step_from_singularities(const Eigen::MatrixXd& uv,
                                            const Eigen::MatrixXi& F,
                                            Eigen::MatrixXd& d) {
    double max_step = INFINITY;

    // The if statement is outside the for loops to avoid branching/ease parallelizing
    if (uv.cols() == 2) {
      for (int f = 0; f < F.rows(); f++) {
        double min_positive_root = get_min_pos_root_2D(uv,F,d,f);
        max_step = min(max_step, min_positive_root);
      }
    } else { // volumetric deformation
      for (int f = 0; f < F.rows(); f++) {
        double min_positive_root = get_min_pos_root_3D(uv,F,d,f);
        max_step = min(max_step, min_positive_root);
      }
    }
    return max_step;
 }

 double Linesearch::get_min_pos_root_2D(const Eigen::MatrixXd& uv,const Eigen::MatrixXi& F,
            Eigen::MatrixXd& d, int f) {
/*
      Finding the smallest timestep t s.t a triangle get degenerated (<=> det = 0)
      The following code can be derived by a symbolic expression in matlab:

      Symbolic matlab:
      U11 = sym('U11');
      U12 = sym('U12');
      U21 = sym('U21');
      U22 = sym('U22');
      U31 = sym('U31');
      U32 = sym('U32');

      V11 = sym('V11');
      V12 = sym('V12');
      V21 = sym('V21');
      V22 = sym('V22');
      V31 = sym('V31');
      V32 = sym('V32');

      t = sym('t');

      U1 = [U11,U12];
      U2 = [U21,U22];
      U3 = [U31,U32];

      V1 = [V11,V12];
      V2 = [V21,V22];
      V3 = [V31,V32];

      A = [(U2+V2*t) - (U1+ V1*t)];
      B = [(U3+V3*t) - (U1+ V1*t)];
      C = [A;B];

      solve(det(C), t);
      cf = coeffs(det(C),t); % Now cf(1),cf(2),cf(3) holds the coefficients for the polynom. at order c,b,a
    */

  int v1 = F(f,0); int v2 = F(f,1); int v3 = F(f,2);
  // get quadratic coefficients (ax^2 + b^x + c)
  #define U11 uv(v1,0)
  #define U12 uv(v1,1)
  #define U21 uv(v2,0)
  #define U22 uv(v2,1)
  #define U31 uv(v3,0)
  #define U32 uv(v3,1)

  #define V11 d(v1,0)
  #define V12 d(v1,1)
  #define V21 d(v2,0)
  #define V22 d(v2,1)
  #define V31 d(v3,0)
  #define V32 d(v3,1)


  double a = V11*V22 - V12*V21 - V11*V32 + V12*V31 + V21*V32 - V22*V31;
  double b = U11*V22 - U12*V21 - U21*V12 + U22*V11 - U11*V32 + U12*V31 + U31*V12 - U32*V11 + U21*V32 - U22*V31 - U31*V22 + U32*V21;
  double c = U11*U22 - U12*U21 - U11*U32 + U12*U31 + U21*U32 - U22*U31;

  return get_smallest_pos_quad_zero(a,b,c);
}

double Linesearch::get_smallest_pos_quad_zero(double a,double b, double c) {
  double t1,t2;
  if (a != 0) {
    double delta_in = pow(b,2) - 4*a*c;
    if (delta_in < 0) {
      return INFINITY;
    }
    double delta = sqrt(delta_in);
    t1 = (-b + delta)/ (2*a);
    t2 = (-b - delta)/ (2*a);
  } else {
    t1 = t2 = -b/c;
  }
  assert (std::isfinite(t1));
  assert (std::isfinite(t2));

  double tmp_n = min(t1,t2);
  t1 = max(t1,t2); t2 = tmp_n;
  if (t1 == t2) {
    return INFINITY; // means the orientation flips twice = doesn't flip
  }
  // return the smallest negative root if it exists, otherwise return infinity
  if (t1 > 0) {
    if (t2 > 0) {
      return t2;
    } else {
      return t1;
    }
  } else {
    return INFINITY;
  }
}

double Linesearch::get_min_pos_root_3D(const Eigen::MatrixXd& uv,const Eigen::MatrixXi& F,
            Eigen::MatrixXd& direc, int f) {
  /*
      Searching for the roots of:
        +-1/6 * |ax ay az 1|
                |bx by bz 1|
                |cx cy cz 1|
                |dx dy dz 1|
      Every point ax,ay,az has a search direction a_dx,a_dy,a_dz, and so we add those to the matrix, and solve the cubic to find the step size t for a 0 volume
      Symbolic matlab:
        syms a_x a_y a_z a_dx a_dy a_dz % tetrahedera point and search direction
        syms b_x b_y b_z b_dx b_dy b_dz
        syms c_x c_y c_z c_dx c_dy c_dz
        syms d_x d_y d_z d_dx d_dy d_dz
        syms t % Timestep var, this is what we're looking for


        a_plus_t = [a_x,a_y,a_z] + t*[a_dx,a_dy,a_dz];
        b_plus_t = [b_x,b_y,b_z] + t*[b_dx,b_dy,b_dz];
        c_plus_t = [c_x,c_y,c_z] + t*[c_dx,c_dy,c_dz];
        d_plus_t = [d_x,d_y,d_z] + t*[d_dx,d_dy,d_dz];

        vol_mat = [a_plus_t,1;b_plus_t,1;c_plus_t,1;d_plus_t,1]
        //cf = coeffs(det(vol_det),t); % Now cf(1),cf(2),cf(3),cf(4) holds the coefficients for the polynom
        [coefficients,terms] = coeffs(det(vol_det),t); % terms = [ t^3, t^2, t, 1], Coefficients hold the coeff we seek
  */
  int v1 = F(f,0); int v2 = F(f,1); int v3 = F(f,2); int v4 = F(f,3);
  #define a_x uv(v1,0)
  #define a_y uv(v1,1)
  #define a_z uv(v1,2)
  #define b_x uv(v2,0)
  #define b_y uv(v2,1)
  #define b_z uv(v2,2)
  #define c_x uv(v3,0)
  #define c_y uv(v3,1)
  #define c_z uv(v3,2)
  #define d_x uv(v4,0)
  #define d_y uv(v4,1)
  #define d_z uv(v4,2)

  #define a_dx direc(v1,0)
  #define a_dy direc(v1,1)
  #define a_dz direc(v1,2)
  #define b_dx direc(v2,0)
  #define b_dy direc(v2,1)
  #define b_dz direc(v2,2)
  #define c_dx direc(v3,0)
  #define c_dy direc(v3,1)
  #define c_dz direc(v3,2)
  #define d_dx direc(v4,0)
  #define d_dy direc(v4,1)
  #define d_dz direc(v4,2)

  // Find solution for: a*t^3 + b*t^2 + c*d +d = 0
  double a = a_dx*b_dy*c_dz - a_dx*b_dz*c_dy - a_dy*b_dx*c_dz + a_dy*b_dz*c_dx + a_dz*b_dx*c_dy - a_dz*b_dy*c_dx - a_dx*b_dy*d_dz + a_dx*b_dz*d_dy + a_dy*b_dx*d_dz - a_dy*b_dz*d_dx - a_dz*b_dx*d_dy + a_dz*b_dy*d_dx + a_dx*c_dy*d_dz - a_dx*c_dz*d_dy - a_dy*c_dx*d_dz + a_dy*c_dz*d_dx + a_dz*c_dx*d_dy - a_dz*c_dy*d_dx - b_dx*c_dy*d_dz + b_dx*c_dz*d_dy + b_dy*c_dx*d_dz - b_dy*c_dz*d_dx - b_dz*c_dx*d_dy + b_dz*c_dy*d_dx;
  double b = a_dy*b_dz*c_x - a_dy*b_x*c_dz - a_dz*b_dy*c_x + a_dz*b_x*c_dy + a_x*b_dy*c_dz - a_x*b_dz*c_dy - a_dx*b_dz*c_y + a_dx*b_y*c_dz + a_dz*b_dx*c_y - a_dz*b_y*c_dx - a_y*b_dx*c_dz + a_y*b_dz*c_dx + a_dx*b_dy*c_z - a_dx*b_z*c_dy - a_dy*b_dx*c_z + a_dy*b_z*c_dx + a_z*b_dx*c_dy - a_z*b_dy*c_dx - a_dy*b_dz*d_x + a_dy*b_x*d_dz + a_dz*b_dy*d_x - a_dz*b_x*d_dy - a_x*b_dy*d_dz + a_x*b_dz*d_dy + a_dx*b_dz*d_y - a_dx*b_y*d_dz - a_dz*b_dx*d_y + a_dz*b_y*d_dx + a_y*b_dx*d_dz - a_y*b_dz*d_dx - a_dx*b_dy*d_z + a_dx*b_z*d_dy + a_dy*b_dx*d_z - a_dy*b_z*d_dx - a_z*b_dx*d_dy + a_z*b_dy*d_dx + a_dy*c_dz*d_x - a_dy*c_x*d_dz - a_dz*c_dy*d_x + a_dz*c_x*d_dy + a_x*c_dy*d_dz - a_x*c_dz*d_dy - a_dx*c_dz*d_y + a_dx*c_y*d_dz + a_dz*c_dx*d_y - a_dz*c_y*d_dx - a_y*c_dx*d_dz + a_y*c_dz*d_dx + a_dx*c_dy*d_z - a_dx*c_z*d_dy - a_dy*c_dx*d_z + a_dy*c_z*d_dx + a_z*c_dx*d_dy - a_z*c_dy*d_dx - b_dy*c_dz*d_x + b_dy*c_x*d_dz + b_dz*c_dy*d_x - b_dz*c_x*d_dy - b_x*c_dy*d_dz + b_x*c_dz*d_dy + b_dx*c_dz*d_y - b_dx*c_y*d_dz - b_dz*c_dx*d_y + b_dz*c_y*d_dx + b_y*c_dx*d_dz - b_y*c_dz*d_dx - b_dx*c_dy*d_z + b_dx*c_z*d_dy + b_dy*c_dx*d_z - b_dy*c_z*d_dx - b_z*c_dx*d_dy + b_z*c_dy*d_dx;
  double c = a_dz*b_x*c_y - a_dz*b_y*c_x - a_x*b_dz*c_y + a_x*b_y*c_dz + a_y*b_dz*c_x - a_y*b_x*c_dz - a_dy*b_x*c_z + a_dy*b_z*c_x + a_x*b_dy*c_z - a_x*b_z*c_dy - a_z*b_dy*c_x + a_z*b_x*c_dy + a_dx*b_y*c_z - a_dx*b_z*c_y - a_y*b_dx*c_z + a_y*b_z*c_dx + a_z*b_dx*c_y - a_z*b_y*c_dx - a_dz*b_x*d_y + a_dz*b_y*d_x + a_x*b_dz*d_y - a_x*b_y*d_dz - a_y*b_dz*d_x + a_y*b_x*d_dz + a_dy*b_x*d_z - a_dy*b_z*d_x - a_x*b_dy*d_z + a_x*b_z*d_dy + a_z*b_dy*d_x - a_z*b_x*d_dy - a_dx*b_y*d_z + a_dx*b_z*d_y + a_y*b_dx*d_z - a_y*b_z*d_dx - a_z*b_dx*d_y + a_z*b_y*d_dx + a_dz*c_x*d_y - a_dz*c_y*d_x - a_x*c_dz*d_y + a_x*c_y*d_dz + a_y*c_dz*d_x - a_y*c_x*d_dz - a_dy*c_x*d_z + a_dy*c_z*d_x + a_x*c_dy*d_z - a_x*c_z*d_dy - a_z*c_dy*d_x + a_z*c_x*d_dy + a_dx*c_y*d_z - a_dx*c_z*d_y - a_y*c_dx*d_z + a_y*c_z*d_dx + a_z*c_dx*d_y - a_z*c_y*d_dx - b_dz*c_x*d_y + b_dz*c_y*d_x + b_x*c_dz*d_y - b_x*c_y*d_dz - b_y*c_dz*d_x + b_y*c_x*d_dz + b_dy*c_x*d_z - b_dy*c_z*d_x - b_x*c_dy*d_z + b_x*c_z*d_dy + b_z*c_dy*d_x - b_z*c_x*d_dy - b_dx*c_y*d_z + b_dx*c_z*d_y + b_y*c_dx*d_z - b_y*c_z*d_dx - b_z*c_dx*d_y + b_z*c_y*d_dx;
  double d = a_x*b_y*c_z - a_x*b_z*c_y - a_y*b_x*c_z + a_y*b_z*c_x + a_z*b_x*c_y - a_z*b_y*c_x - a_x*b_y*d_z + a_x*b_z*d_y + a_y*b_x*d_z - a_y*b_z*d_x - a_z*b_x*d_y + a_z*b_y*d_x + a_x*c_y*d_z - a_x*c_z*d_y - a_y*c_x*d_z + a_y*c_z*d_x + a_z*c_x*d_y - a_z*c_y*d_x - b_x*c_y*d_z + b_x*c_z*d_y + b_y*c_x*d_z - b_y*c_z*d_x - b_z*c_x*d_y + b_z*c_y*d_x;

  if (a==0) {
    return get_smallest_pos_quad_zero(b,c,d);
  }
  b/=a; c/=a; d/=a; // normalize it all
  std::vector<double> res(3);
  int real_roots_num = SolveP3(res,b,c,d);
  switch (real_roots_num) {
    case 1:
      return (res[0] >= 0) ? res[0]:INFINITY;
    case 2: {
      double max_root = max(res[0],res[1]); double min_root = min(res[0],res[1]);
      if (min_root > 0) return min_root;
      if (max_root > 0) return max_root;
      return INFINITY;
    }
    case 3:
    default: {
      std::sort(res.begin(),res.end());
      if (res[0] > 0) return res[0];
      if (res[1] > 0) return res[1];
      if (res[2] > 0) return res[2];
      return INFINITY;
    }
  }

}

#define TwoPi  6.28318530717958648
const double eps=1e-14;
//---------------------------------------------------------------------------
// x - array of size 3
// In case 3 real roots: => x[0], x[1], x[2], return 3
//         2 real roots: x[0], x[1],          return 2
//         1 real root : x[0], x[1] ± i*x[2], return 1
// http://math.ivanovo.ac.ru/dalgebra/Khashin/poly/index.html
int Linesearch::SolveP3(std::vector<double>& x,double a,double b,double c) { // solve cubic equation x^3 + a*x^2 + b*x + c
  double a2 = a*a;
    double q  = (a2 - 3*b)/9;
  double r  = (a*(2*a2-9*b) + 27*c)/54;
    double r2 = r*r;
  double q3 = q*q*q;
  double A,B;
    if(r2<q3) {
        double t=r/sqrt(q3);
    if( t<-1) t=-1;
    if( t> 1) t= 1;
        t=acos(t);
        a/=3; q=-2*sqrt(q);
        x[0]=q*cos(t/3)-a;
        x[1]=q*cos((t+TwoPi)/3)-a;
        x[2]=q*cos((t-TwoPi)/3)-a;
        return(3);
    } else {
        A =-pow(fabs(r)+sqrt(r2-q3),1./3);
    if( r<0 ) A=-A;
    B = A==0? 0 : B=q/A;

    a/=3;
    x[0] =(A+B)-a;
        x[1] =-0.5*(A+B)-a;
        x[2] = 0.5*sqrt(3.)*(A-B);
    if(fabs(x[2])<eps) { x[2]=x[1]; return(2); }
        return(1);
    }
}


/// Slim Implementation

SLIMData::SLIMData(Eigen::MatrixXd& V_in, Eigen::MatrixXi& F_in) : V(V_in), F(F_in) {
  proximal_p = 0.0001;

  v_num = V.rows();
  f_num = F.rows();
  igl::doublearea(V,F,M); M /= 2.;
  mesh_area = M.sum();
  mesh_improvement_3d = false; // whether to use a jacobian derived from a real mesh or an abstract regular mesh (used for mesh improvement)
  exp_factor = 1.0; // param used only for exponential energies (e.g exponential symmetric dirichlet)
}

Slim::Slim(SLIMData& m_state) :
      m_state(m_state), wGlobalLocal(NULL) {
  assert (m_state.F.cols() == 3 || m_state.F.cols() == 4);
  wGlobalLocal = new WeightedGlobalLocal(m_state);
}

void Slim::precompute() {
  wGlobalLocal->pre_calc();
  m_state.energy = wGlobalLocal->compute_energy(m_state.V_o)/m_state.mesh_area;
}

void Slim::solve(int iter_num) {
  for (int i = 0; i < iter_num; i++) {
    slim_iter();
  }
}

void Slim::slim_iter() {
  Linesearch linesearch(m_state);
  Eigen::MatrixXd dest_res;
  dest_res = m_state.V_o;
  wGlobalLocal->solve_weighted_proxy(dest_res);

  double old_energy = m_state.energy;

  m_state.energy = linesearch.compute(m_state.V,m_state.F, m_state.V_o, dest_res, wGlobalLocal,
                                         m_state.energy*m_state.mesh_area)/m_state.mesh_area;
}
