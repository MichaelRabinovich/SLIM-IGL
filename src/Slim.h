#ifndef SLIM_H
#define SLIM_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <string>

#include <igl/jet.h>
#include <igl/readOBJ.h>
#include <igl/facet_components.h>
#include <igl/slice.h>

class WeightedGlobalLocal;

// Compute a SLIM map as derived in "Scalable Locally Injective Maps" [Rabinovich et al. 2016].
struct SLIMData { 

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

  WeightedGlobalLocal* wGlobalLocal;
};
  // Initialize a SLIM deformation
  // Inputs:
  //	SLIMData structure, which should include:
  //		V  #V by 3 list of mesh vertex positions
  //		F  #F by 3/3 list of mesh faces (triangles/tets)
  //		SLIM_ENERGY Energy to optimize
  //		For other parameters see @SLIMData
//SLIMData();

// Compute necessary information to start using a SLIM deformation
void slim_precompute(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::VectorXi b, Eigen::MatrixXd& bc, double soft_const_p,
    Eigen::MatrixXd& V_init, SLIMData& slimData);

// Run iter_num iterations of SLIM
// Outputs:
//    V_o (in SLIMData): #V by dim list of mesh vertex positions
void slim_solve(SLIMData& data, int iter_num);

#endif // SLIM_H
