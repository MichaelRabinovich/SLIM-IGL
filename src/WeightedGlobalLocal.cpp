#include "WeightedGlobalLocal.h"

#include "igl/arap.h"
#include "igl/cat.h"
#include "igl/doublearea.h"
#include "igl/grad.h"
#include "igl/local_basis.h"
#include "igl/min_quad_with_fixed.h"
#include "igl/Timer.h"

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>

using namespace Eigen;

WeightedGlobalLocal::WeightedGlobalLocal(SLIMData& state, bool remeshing) : 
                                  m_state(state) {
}

void WeightedGlobalLocal::compute_map( const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
              Eigen::VectorXi& b, Eigen::MatrixXd& bc, Eigen::MatrixXd& uv) {

  update_weights_and_closest_rotations(V,F,uv);
  solve_weighted_arap(V,F,uv,b,bc);
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
  double exp_factor = m_state.exp_factor;
  for(int i=0; i <Ji.rows(); ++i ) {
    typedef Eigen::Matrix<double,2,2> Mat2;
    typedef Eigen::Matrix<double,2,1> Vec2;
    Mat2 ji,ri,ti,ui,vi; Vec2 sing; Vec2 closest_sing_vec;Mat2 mat_W;
    Mat2 fGrad; Vec2 m_sing_new;
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

        double in_exp = exp_factor*((pow(s1,2)+pow(s2,2))/(2*s1*s2));
        double exp_thing = exp(in_exp);

        s1_g *= exp_thing*exp_factor;
        s2_g *= exp_thing*exp_factor;

        m_sing_new << sqrt(s1_g/(2*(s1-1))), sqrt(s2_g/(2*(s2-1)));
        break;
    } case SLIMData::AMIPS_ISO_2D: {
        // Amips ISO energy in singular values: exp(5* (  0.5*(s1/s2 +s2/s1) + 0.25*( s1*s2 + 1/(s1*s2) )  ) )
        // Partial derivatives for s1 is: 5*( 0.25 * (s2-1/(s2*s1^2)) + 0.5*(1/s2 - s2/(s1^2))  )* exp(5* (  0.5*(s1/s2 +s2/s1) + 0.25*( s1*s2 + 1/(s1*s2) )  ) )
        double exp_thing = exp(exp_factor*(0.5*(s1/s2 + s2/s1) + 0.25*(s1*s2 + pow(s1*s2,-1))));
        double s1_g = exp_thing*exp_factor * (0.25 * (s2- (1./(s2*pow(s1,2)))) + 0.5 * ((1./s2) - s2/(pow(s1,2))) ); //(exp_factor/4)*(s2- (1./(s2*pow(s1,2))))*exp_thing + (exp_factor/2)*((1./s2) - s2/(pow(s1,2)))*exp_thing;
        double s2_g = exp_thing*exp_factor * (0.25 * (s1- (1./(s1*pow(s2,2)))) + 0.5 * ((1./s1) - s1/(pow(s2,2))) );
        
        double s1_zero = sqrt(2*pow(s2,2)+1)/sqrt(pow(s2,2)+2); double s2_zero = sqrt(2*pow(s1,2)+1)/sqrt(pow(s1,2)+2);
        m_sing_new << sqrt(s1_g/(2*(s1-s1_zero))), sqrt(s2_g/(2*(s2-s2_zero)));

        // change local step
        closest_sing_vec << s1_zero, s2_zero;
        ri = ui*closest_sing_vec.asDiagonal()*vi.transpose();
        break;
    } case SLIMData::EXP_symmd: {
        double s1_g = 2* (s1-pow(s1,-3)); 
        double s2_g = 2 * (s2-pow(s2,-3));

        double in_exp = exp_factor*(pow(s1,2)+pow(s1,-2)+pow(s2,2)+pow(s2,-2));
        double exp_thing = exp(in_exp);

        s1_g *= exp_thing*exp_factor;
        s2_g *= exp_thing*exp_factor;

        m_sing_new << sqrt(s1_g/(2*(s1-1))), sqrt(s2_g/(2*(s2-1)));
        break;
      }
    }

    if (abs(s1-1) < eps) m_sing_new(0) = 1; if (abs(s2-1) < eps) m_sing_new(1) = 1;
    mat_W = ui*m_sing_new.asDiagonal()*ui.transpose();

    W_11(i) = mat_W(0,0);
    W_12(i) = mat_W(0,1);
    W_21(i) = mat_W(1,0);
    W_22(i) = mat_W(1,1);

    // 2) Update local step (doesn't have to be a rotation, for instance in case of conformal energy)
    Ri(i,0) = ri(0,0); Ri(i,1) = ri(1,0); Ri(i,2) = ri(0,1); Ri(i,3) = ri(1,1);
   }
}

void WeightedGlobalLocal::solve_weighted_arap(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
        Eigen::MatrixXd& uv, Eigen::VectorXi& soft_b_p, Eigen::MatrixXd& soft_bc_p) {
  using namespace Eigen;

  Eigen::SparseMatrix<double> L;
  build_linear_system(L);

  // solve
  igl::min_quad_with_fixed_data<double> solver_data;
  bool ret_x = min_quad_with_fixed_precompute(
    L,Eigen::VectorXi(),Eigen::SparseMatrix<double>(),false,solver_data);
  Eigen::VectorXd Uc,Beq;
  Eigen::VectorXd neg_rhs = -1*rhs; //libigl solver expects a minus
  igl::min_quad_with_fixed_solve(
          solver_data,
          neg_rhs,Eigen::VectorXd(),Beq,
          Uc);

  uv.col(0) = Uc.block(0,0,m_state.v_num,1);
  uv.col(1) = Uc.block(m_state.v_num,0,m_state.v_num,1);
}

void WeightedGlobalLocal::pre_calc() {
  if (!has_pre_calc) {
    int f_n = m_state.f_num;

    if (m_state.F.cols() == 3) {
      dim = 2;
      Eigen::MatrixXd F1,F2,F3;
      igl::local_basis(m_state.V,m_state.F,F1,F2,F3);
      compute_surface_gradient_matrix(m_state.V,m_state.F,F1,F2,Dx,Dy);

      W_11.resize(f_n); W_12.resize(f_n); W_21.resize(f_n); W_22.resize(f_n);
    } else {
      dim = 3;
      compute_tet_grad_matrix(m_state.V,m_state.F,Dx,Dy,Dz,false /*remeshing, TODO: support me*/);

      W_11.resize(f_n);W_12.resize(f_n);W_13.resize(f_n);
      W_21.resize(f_n);W_22.resize(f_n);W_23.resize(f_n);
      W_31.resize(f_n);W_32.resize(f_n);W_33.resize(f_n);
    }

    Dx.makeCompressed();Dy.makeCompressed(); Dz.makeCompressed();
    Ri.resize(f_n, dim*dim); Ji.resize(f_n, dim*dim);
    rhs.resize(dim*m_state.v_num);

    first_solve = true;
    has_pre_calc = true;
  }
}

void WeightedGlobalLocal::buildA(Eigen::SparseMatrix<double>& A) {

  int v_n = m_state.v_num; int f_n = m_state.f_num;
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

  }
  A.setFromTriplets(IJV.begin(),IJV.end());
}

void WeightedGlobalLocal::build_linear_system(Eigen::SparseMatrix<double> &L) {
  int f_n = m_state.f_num; int v_n = m_state.v_num;

  // formula (35) in paper
  Eigen::SparseMatrix<double> A(dim*dim*f_n, dim*v_n);
  buildA(A);

  Eigen::SparseMatrix<double> At = A.transpose();
  At.makeCompressed();
  
  Eigen::VectorXd M(dim*dim*f_n);
  for (int i = 0; i < dim*dim; i++) {
    for (int j = 0; j < f_n; j++) {
      M(i*f_n + j) = m_state.M(j);
    }
  }

  Eigen::SparseMatrix<double> id_m(At.rows(),At.rows()); id_m.setIdentity();

  // add proximal penalty
  L = At*M.asDiagonal()*A + m_state.proximal_p * id_m; //add also a proximal term
  L.makeCompressed();

  // build rhs
  /*rhs = [W11*R11 + W12*R21;
         W11*R12 + W12*R22;
         W21*R11 + W22*R21;
         W21*R12 + W22*R22];*/
  VectorXd f_rhs(dim*dim*f_n); f_rhs.setZero();
  for (int i = 0; i < f_n; i++) {
    f_rhs(i+0*f_n) = W_11(i) * Ri(i,0) + W_12(i)*Ri(i,1);
    f_rhs(i+1*f_n) = W_11(i) * Ri(i,2) + W_12(i)*Ri(i,3);
    f_rhs(i+2*f_n) = W_21(i) * Ri(i,0) + W_22(i)*Ri(i,1);
    f_rhs(i+3*f_n) = W_21(i) * Ri(i,2) + W_22(i)*Ri(i,3);
  }
  VectorXd uv_flat = igl::cat<VectorXd>(1, m_state.V_o.col(0), m_state.V_o.col(1));
  rhs = (At*M.asDiagonal()*f_rhs + m_state.proximal_p * uv_flat);
  add_soft_constraints(L); //TODO: support me
}

void WeightedGlobalLocal::add_soft_constraints(Eigen::SparseMatrix<double> &L) {
  int v_n = m_state.v_num;
  for (int i = 0; i < m_state.b.rows(); i++) {
    int v_idx = m_state.b(i);
    rhs(v_idx) += m_state.soft_const_p * m_state.bc(i,0);
    rhs(v_n + v_idx) += m_state.soft_const_p * m_state.bc(i,1);

    L.coeffRef(v_idx, v_idx) += m_state.soft_const_p;
    L.coeffRef(2*v_idx, 2*v_idx) += m_state.soft_const_p;
  }
}

double WeightedGlobalLocal::compute_energy(const Eigen::MatrixXd& V,
                                                       const Eigen::MatrixXi& F,  
                                                       Eigen::MatrixXd& V_o) {
  compute_jacobians(V_o);
  return compute_energy_with_jacobians(V,F, Ji, V_o,m_state.M);
}

double WeightedGlobalLocal::compute_energy_with_jacobians(const Eigen::MatrixXd& V,
       const Eigen::MatrixXi& F, const Eigen::MatrixXd& Ji, Eigen::MatrixXd& uv, Eigen::VectorXd& areas) {

  int f_n = F.rows();

  double energy = 0;
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
      case SLIMData::AMIPS_ISO_2D: {
        energy += areas(i) * exp(m_state.exp_factor* (  0.5*( (s1/s2) +(s2/s1) ) + 0.25*( (s1*s2) + (1./(s1*s2)) )  ) );
        break;
      }
      case SLIMData::EXP_symmd: {
        energy += areas(i) * exp(m_state.exp_factor*(pow(s1,2) +pow(s1,-2) + pow(s2,2) + pow(s2,-2)));
        break;
      }
    }
    
  }
  return energy;
}
