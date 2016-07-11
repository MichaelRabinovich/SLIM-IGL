#include "LocalWeightedArapParametrizer.h"

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

LocalWeightedArapParametrizer::LocalWeightedArapParametrizer(SLIMData& state, bool remeshing) : 
                                  m_state(state) {
}

void LocalWeightedArapParametrizer::parametrize( const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
              Eigen::VectorXi& b, Eigen::MatrixXd& bc, Eigen::MatrixXd& uv) {

  update_weights_and_closest_rotations(V,F,uv);
  solve_weighted_arap(V,F,uv,b,bc);
}

void LocalWeightedArapParametrizer::compute_jacobians(const Eigen::MatrixXd& uv) {
  // Ji=[D1*u,D2*u,D1*v,D2*v];
  int k_idx = 0;
  for (int i = 0; i < f_n; i++) {
    const int col1 = dxj[k_idx]-1; const int col2 = dxj[k_idx+1]-1; const int col3 = dxj[k_idx+2]-1;

    const double a_x_k = a_x(k_idx); const double a_x_k_1 = a_x(k_idx+1); const double a_x_k_2 = a_x(k_idx+2);

    Ji(i,0) = a_x_k * uv(col1,0) +  a_x_k_1 * uv(col2,0) +  a_x_k_2* uv(col3,0); // Dx*u
    Ji(i,2) = a_x_k * uv(col1,1) +  a_x_k_1 * uv(col2,1) + a_x_k_2 * uv(col3,1); // Dx*v

    const double a_y_k = a_y(k_idx); const double a_y_k_1 = a_y(k_idx+1); const double a_y_k_2 = a_y(k_idx+2);
    Ji(i,1) = a_y_k * uv(col1,0) + a_y_k_1 * uv(col2,0) + a_y_k_2 * uv(col3,0); // Dy*u
    Ji(i,3) = a_y_k * uv(col1,1) + a_y_k_1 * uv(col2,1) + a_y_k_2 * uv(col3,1); // Dy*v
    
    k_idx +=3;
  }
  
}

void LocalWeightedArapParametrizer::update_weights_and_closest_rotations(const Eigen::MatrixXd& V,
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

void LocalWeightedArapParametrizer::solve_weighted_arap(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
        Eigen::MatrixXd& uv, Eigen::VectorXi& soft_b_p, Eigen::MatrixXd& soft_bc_p) {
  using namespace Eigen;

  /*b = [W11*R11 + W12*R21;
         W11*R12 + W12*R22;
         W21*R11 + W22*R21;
         W21*R12 + W22*R22];*/
  double h = m_state.proximal_p;

  get_At_AtMA_fast();
  
  // move to zero based
  for (int i = 0; i < aj.rows(); i++) {
    aj(i)--;
  }
  for (int i = 0; i < ai.rows(); i++) {
    ai(i)--;
  }

  igl::min_quad_with_fixed_data<double> solver_data;
  int nnz = aj.rows();
  SparseMatrix<double,RowMajor> mat = MappedSparseMatrix<double,Eigen::RowMajor>(2*v_n, 2*v_n, nnz, ai.data(), aj.data(), K.data());
  Eigen::SparseMatrix<double> Q(2*v_n,2*v_n);
  Q = mat.selfadjointView<Eigen::Upper>();

  bool ret_x = min_quad_with_fixed_precompute(
    Q,Eigen::VectorXi(),Eigen::SparseMatrix<double>(),false,solver_data);
  Eigen::VectorXd Uc,Beq;
  Eigen::VectorXd negRhs = -rhs;
  igl::min_quad_with_fixed_solve(
          solver_data,
          negRhs,Eigen::VectorXd(),Beq,
          Uc);

  uv.col(0) = Uc.block(0,0,v_n,1);
  uv.col(1) = Uc.block(v_n,0,v_n,1);

  // back to 1-based for Pardiso
  for (int i = 0; i < aj.rows(); i++) {
    aj(i)++;
  }
  for (int i = 0; i < ai.rows(); i++) {
    ai(i)++;
  }
  
}

void LocalWeightedArapParametrizer::pre_calc() {
  if (!has_pre_calc) {
    f_n = m_state.F.rows(); v_n = m_state.V.rows();
    W_11.resize(f_n); W_12.resize(f_n); W_21.resize(f_n); W_22.resize(f_n);
    Ri.resize(f_n, 4); Ji.resize(f_n, 4);
    Eigen::MatrixXd F1,F2,F3;

    igl::local_basis(m_state.V,m_state.F,F1,F2,F3);
    Eigen::SparseMatrix<double> Dx,Dy;
    compute_surface_gradient_matrix(m_state.V,m_state.F,F1,F2,Dx,Dy);

    first_solve = true;
    
    build_dx_k_maps(v_n, m_state.F, ai,aj, inst1, inst2, inst4, inst1_idx,inst2_idx,inst4_idx);
    

    K.resize(aj.rows());
    Eigen::VectorXi ai_t; Eigen::VectorXi aj_t;
    dx_to_csr(Dx, dxi, dxj, a_x); dx_to_csr(Dy, ai_t, aj_t, a_y);

    int ax_size = a_x.rows();
    w11Dx.resize(ax_size); w12Dx.resize(ax_size); w11Dy.resize(ax_size); w12Dy.resize(ax_size);
    w21Dx.resize(ax_size); w22Dx.resize(ax_size); w21Dy.resize(ax_size); w22Dy.resize(ax_size);

    rhs.resize(2*v_n);

    has_pre_calc = true;
  }
}

void LocalWeightedArapParametrizer::get_At_AtMA_fast() {
  using namespace Eigen;

  rhs.setZero();
  int A_idx = 0;
  for (int i = 0; i < f_n; i++) {
    int nnz_in_line = dxi[i+1] - dxi[i];
    for (int j = 0; j < nnz_in_line; j++) {
      double f1_i = m_state.M(i) * (W_11(i) * Ri(i,0) + W_12(i)*Ri(i,1));
      double f2_i = m_state.M(i) * (W_11(i) * Ri(i,2) + W_12(i)*Ri(i,3));
      double f3_i = m_state.M(i) * (W_21(i) * Ri(i,0) + W_22(i)*Ri(i,1));
      double f4_i = m_state.M(i) * (W_21(i) * Ri(i,2) + W_22(i)*Ri(i,3));

      int dest_k = dxj[A_idx]-1;
      rhs(dest_k) += a_x(A_idx) * W_11(i) * f1_i + a_y(A_idx) * W_11(i) * f2_i  + a_x(A_idx) * W_21(i) * f3_i + a_y(A_idx) * W_21(i) * f4_i;
      rhs(dest_k+v_n) += a_x(A_idx) * W_21(i) * f1_i + a_y(A_idx) * W_21(i) * f2_i  + a_x(A_idx) * W_22(i) * f3_i + a_y(A_idx) * W_22(i) * f4_i;

      A_idx++;
    }
  }

  for (int i = 0; i < f_n; i++) {
      W_11(i)*=sqrt(m_state.M(i));
      W_12(i)*=sqrt(m_state.M(i));
      W_21(i)*=sqrt(m_state.M(i));
      W_22(i)*=sqrt(m_state.M(i));
  }

  multiply_dx_by_W(a_x, W_11, w11Dx); multiply_dx_by_W(a_x, W_12, w12Dx);
  multiply_dx_by_W(a_y, W_11, w11Dy); multiply_dx_by_W(a_y, W_12, w12Dy);
  multiply_dx_by_W(a_x, W_21, w21Dx); multiply_dx_by_W(a_x, W_22, w22Dx);
  multiply_dx_by_W(a_y, W_21, w21Dy); multiply_dx_by_W(a_y, W_22, w22Dy);
  
  K.setZero();

  // K1
  add_dx_mult_dx_to_K(w11Dx,w11Dx, K, inst1,inst1_idx); add_dx_mult_dx_to_K(w11Dy,w11Dy, K, inst1,inst1_idx); 
  add_dx_mult_dx_to_K(w21Dx,w21Dx, K, inst1,inst1_idx); add_dx_mult_dx_to_K(w21Dy,w21Dy, K, inst1,inst1_idx);

  // K2
  add_dx_mult_dx_to_K(w11Dx,w12Dx, K, inst2,inst2_idx); add_dx_mult_dx_to_K(w11Dy,w12Dy, K, inst2,inst2_idx); 
  add_dx_mult_dx_to_K(w21Dx,w22Dx, K, inst2,inst2_idx); add_dx_mult_dx_to_K(w21Dy,w22Dy, K, inst2,inst2_idx);

  // K4
  add_dx_mult_dx_to_K(w12Dx,w12Dx, K, inst4,inst4_idx); add_dx_mult_dx_to_K(w12Dy,w12Dy, K, inst4,inst4_idx); 
  add_dx_mult_dx_to_K(w22Dx,w22Dx, K, inst4,inst4_idx); add_dx_mult_dx_to_K(w22Dy,w22Dy, K, inst4,inst4_idx);

  add_proximal_penalty();
}

void LocalWeightedArapParametrizer::add_proximal_penalty() {
  double h = m_state.proximal_p; 
    // add proximal penalty
    for (int i = 0; i < v_n; i++) {
      rhs(i) += h * m_state.V_o(i,0);
      rhs(v_n + i) += h * m_state.V_o(i,1);
    }
  int k_idx = 0;
  for (int i = 0; i < ai.rows()-1; i++) {
    int nnz = ai[i+1] - ai[i];
    for (int j = 0; j < nnz; j++) {
      int col = aj[k_idx];
      if ((i+1)==col) {
        K[k_idx] += h;
      }
      k_idx++;
    }
  }
}

double LocalWeightedArapParametrizer::compute_energy(const Eigen::MatrixXd& V,
                                                       const Eigen::MatrixXi& F,  
                                                       Eigen::MatrixXd& V_o) {
  compute_jacobians(V_o);
  return compute_energy_with_jacobians(V,F, Ji, V_o,m_state.M);
}

double LocalWeightedArapParametrizer::compute_energy_with_jacobians(const Eigen::MatrixXd& V,
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
