#ifndef WEIGHTED_GLOBAL_LOCAL_H
#define WEIGHTED_GLOBAL_LOCAL_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>
#include <set>
#include <vector>

#include "SLIMData.h"
#include "geometric_utils.h"

#include "igl/arap.h"

// Computes the weights and solve the linear system for the quadratic proxy specified in the paper
// The output of this is used to generate a search direction that will be fed to the Linesearch class
class WeightedGlobalLocal {

public:
  WeightedGlobalLocal(SLIMData& state);

  // Compute necessary information to before solving the proxy quadratic
  void pre_calc();
  
  // Solve the weighted proxy global step
  // Output:
  //    V_new #V by dim list of mesh positions
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

#endif // WEIGHTED_GLOBAL_LOCAL_H