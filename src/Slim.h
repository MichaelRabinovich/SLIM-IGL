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

struct SLIMData {

public:

  SLIMData(Eigen::MatrixXd& V, Eigen::MatrixXi& F);

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

// Compute a SLIM map as derived in "Scalable Locally Injective Maps" [Rabinovich et al. 2016].
class Slim {

public:

  // Initialize a SLIM deformation
  // Inputs:
  //	SLIMData structure, which should include:
  //		V  #V by 3 list of mesh vertex positions
  //		F  #F by 3/3 list of mesh faces (triangles/tets)
  //		SLIM_ENERGY Energy to optimize
  //		For other parameters see @SLIMData
  Slim(SLIMData& m_state);

  // Compute necessary information to start using a SLIM deformation
  void precompute();

  // Run iter_num iterations of SLIM
  // Outputs:
  //		V_o (in SLIMData): #V by dim list of mesh vertex positions
  void solve(int iter_num);

private:

  void slim_iter();

  WeightedGlobalLocal* wGlobalLocal;
  SLIMData& m_state;
};

#endif // SLIM_H
