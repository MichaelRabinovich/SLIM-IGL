#ifndef SLIM_DATA_H
#define SLIM_DATA_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <set>
#include <string>

#include "igl/arap.h"
#include "igl/Timer.h"

struct SLIMData {

public:

  SLIMData(Eigen::MatrixXd& V, Eigen::MatrixXi& F);

  void save(const std::string filename);
  void load(const std::string filename);

  // Global Information
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  Eigen::VectorXd M;
  Eigen::MatrixXd V_o; // output vertices

  int v_num;
  int f_num;

  double energy;

  enum SLIM_ENERGY {
    ARAP,
    LOG_ARAP,
    SYMMETRIC_DIRICHLET,
    CONFORMAL,
    EXP_CONFORMAL,
    EXP_SYMMETRIC_DIRICHLET
  };
  SLIM_ENERGY slim_energy;

  // soft constraints
  Eigen::VectorXi b;
  Eigen::MatrixXd bc;
  double soft_const_p;

  double proximal_p;
  double exp_factor; // used for exponential energies, ignored otherwise
  bool mesh_improvement_3d; // only supported for 3d

  // INTERNAL
  double mesh_area;
  double avg_edge_length;
};

#endif // SLIM_DATA_H
