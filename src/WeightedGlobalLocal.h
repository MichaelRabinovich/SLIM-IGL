#ifndef LOCAL_WEIGHTED_ARAP_PARAMETRIZER_H
#define LOCAL_WEIGHTED_ARAP_PARAMETRIZER_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>
#include <set>
#include <vector>

#include "Energy.h"
#include "FastLsBuildUtils.h"
#include "SLIMData.h"
#include "geometric_utils.h"

#include "igl/arap.h"

class WeightedGlobalLocal : public Energy {

public:
  WeightedGlobalLocal(SLIMData& state, bool remeshing = false);

  void compute_map( const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    Eigen::VectorXi& soft_b,
    Eigen::MatrixXd& soft_bc,
    Eigen::MatrixXd& V_o);

  virtual double compute_energy(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                 Eigen::MatrixXd& V_o);

  void pre_calc();

  void compute_jacobians(const Eigen::MatrixXd& V_o);

  Eigen::MatrixXd Ri,Ji;
  Eigen::VectorXd W_11; Eigen::VectorXd W_12; Eigen::VectorXd W_21; Eigen::VectorXd W_22;
  Eigen::SparseMatrix<double> Dx,Dy;
private:

  void update_weights_and_closest_rotations(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& uv);
  void solve_weighted_arap(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& uv, Eigen::VectorXi& b,
      Eigen::MatrixXd& bc);

  void get_At_AtMA_fast();
  void add_soft_constraints();
  void add_proximal_penalty();

  double compute_energy_with_jacobians(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, 
    const Eigen::MatrixXd& Ji, Eigen::MatrixXd& V_o, Eigen::VectorXd& areas);
  
  SLIMData& m_state;
  Eigen::VectorXd w11Dx,w12Dx,w11Dy,w12Dy,w21Dx,w22Dx,w21Dy,w22Dy;
  Eigen::VectorXd rhs;

  // Cached data for the matrix system calculations (the sparsity pattern is constant)
  Eigen::VectorXd a_x,a_y;
  Eigen::VectorXi dxi,dxj;
  Eigen::VectorXi ai,aj;
  Eigen::VectorXd K;
  instruction_list inst1,inst2,inst4;
  std::vector<int> inst1_idx,inst2_idx,inst4_idx;

  int f_n,v_n;

  bool first_solve;
  bool has_pre_calc = false;
};

#endif // #ifndef LOCAL_WEIGHTED_ARAP_PARAMETRIZER_H