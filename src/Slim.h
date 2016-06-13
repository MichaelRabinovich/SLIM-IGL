#ifndef SLIM_H
#define SLIM_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <string>


#include "LocalWeightedArapParametrizer.h"
#include "ParametrizationAlgorithm.h"

#include <igl/jet.h>
#include <igl/readOBJ.h>
#include <igl/facet_components.h>
#include <igl/slice.h>

class Slim {

public:

  Slim(Param_State* m_state);

  void precompute();
  void solve(Eigen::MatrixXd& outV, int iter_num);

private:

  void single_line_search_arap();
  void get_linesearch_params(Eigen::MatrixXd& dest_res, Energy** param_energy);

  LocalWeightedArapParametrizer* WArap_p;
  Param_State* m_state;
};

#endif // SLIM_H
