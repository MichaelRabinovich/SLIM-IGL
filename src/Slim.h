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

  Slim(SLIMData& m_state);

  void precompute();
  void solve(int iter_num);

private:

  void slim_iter();

  LocalWeightedArapParametrizer* WArap_p;
  SLIMData& m_state;
};

#endif // SLIM_H
