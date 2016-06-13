#ifndef SLIM_H
#define SLIM_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <string>

#include "LocalWeightedArapParametrizer.h"

#include <igl/jet.h>
#include <igl/readOBJ.h>
#include <igl/facet_components.h>
#include <igl/slice.h>

class Slim {

public:

  Slim(SLIMData* m_state);

  void precompute();
  void solve(Eigen::MatrixXd& outV, int iter_num);

private:

  void single_line_search_arap();

  LocalWeightedArapParametrizer* WArap_p;
  SLIMData* m_state;
};

#endif // SLIM_H
