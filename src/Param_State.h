#ifndef PARAM_STATE_H
#define PARAM_STATE_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <set>
#include <string>

#include "igl/arap.h"
#include "igl/Timer.h"

struct Param_State {

public:

  Param_State(){}

  void save(const std::string filename);
  void load(const std::string filename);

  // Global Information
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  Eigen::VectorXd M;
  Eigen::MatrixXd uv;

  int v_num;
  int f_num;

  // viewing params
  double mesh_area;
  double avg_edge_length;

  // result measurements
  double energy;

  enum Param_Method {
    GLOBAL_ARAP_IRLS = 0,
    symmd,
    REMESHING,
    REMESHING_3D,
    BAR_DEFORMATION_3D,
    EQUALITY_CONSTRAINTS_DEMO,
    SEAMLESS_DEMO
  };
  Param_Method method;

  enum GLOBAL_LOCAL_ENERGY {
  ARAP,
  LOG_ARAP,
  SYMMETRIC_DIRICHLET,
  CONFORMAL,
  EXP_CONFORMAL,
  AMIPS_ISO_2D,
  EXP_symmd
  };
  GLOBAL_LOCAL_ENERGY global_local_energy;

  // constraints (and or stiffness)
  Eigen::VectorXi b;
  Eigen::MatrixXd bc;
  double proximal_p;
  double soft_const_p;
  double exp_factor;
};

#endif // PARAM_STATE_H
