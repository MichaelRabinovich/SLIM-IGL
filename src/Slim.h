#ifndef SLIM_H
#define SLIM_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <string>

#include "WeightedGlobalLocal.h"

#include <igl/jet.h>
#include <igl/readOBJ.h>
#include <igl/facet_components.h>
#include <igl/slice.h>

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
