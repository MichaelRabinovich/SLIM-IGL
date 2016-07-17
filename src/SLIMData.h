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

  void save(const std::string filename); // serialize the structure
  void load(const std::string filename); // deserialize the structure

  // Input
  Eigen::MatrixXd V; // #V by 3 list of mesh vertex positions
  Eigen::MatrixXi F; // #F by 3/3 list of mesh faces (triangles/tets)  
  enum SLIM_ENERGY {
    ARAP,
    LOG_ARAP,
    SYMMETRIC_DIRICHLET,
    CONFORMAL,
    EXP_CONFORMAL,
    EXP_SYMMETRIC_DIRICHLET
  };
  SLIM_ENERGY slim_energy;

  // Optional Input
    // soft constraints
    Eigen::VectorXi b;
    Eigen::MatrixXd bc;
    double soft_const_p;

  double exp_factor; // used for exponential energies, ignored otherwise
  bool mesh_improvement_3d; // only supported for 3d

  // Output
  Eigen::MatrixXd V_o; // #V by dim list of mesh vertex positions (dim = 2 for parametrization, 3 otherwise)
  double energy; // objective value

  // INTERNAL
  Eigen::VectorXd M;
  double mesh_area;
  double avg_edge_length;
  int v_num;
  int f_num;
  double proximal_p;
};

#endif // SLIM_DATA_H
