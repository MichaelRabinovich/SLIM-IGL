#include "Slim.h"

#include "SLIMData.h"
#include "geometric_utils.h"
#include "Linesearch.h"


#include <igl/boundary_loop.h>
#include <igl/cotmatrix.h>
#include <igl/edge_lengths.h>
#include <igl/local_basis.h>
#include <igl/readOBJ.h>
#include <igl/repdiag.h>
#include <igl/vector_area_matrix.h>
#include <iostream>

using namespace std;

Slim::Slim(SLIMData& m_state) :
      m_state(m_state), wGlobalLocal(NULL) {
  assert (m_state.F.cols() == 3 || m_state.F.cols() == 4);
  wGlobalLocal = new WeightedGlobalLocal(m_state);
}

void Slim::precompute() {
  wGlobalLocal->pre_calc();
  m_state.energy = wGlobalLocal->compute_energy(m_state.V, m_state.F, m_state.V_o)/m_state.mesh_area;
}

void Slim::solve(int iter_num) {
  for (int i = 0; i < iter_num; i++) {
    slim_iter();
  }
}

void Slim::slim_iter() {
  Linesearch linesearch(m_state);
  Eigen::MatrixXd dest_res;
  dest_res = m_state.V_o;
  wGlobalLocal->compute_map(m_state.V,m_state.F, m_state.b,m_state.bc, dest_res);

  double old_energy = m_state.energy;

  m_state.energy = linesearch.compute(m_state.V,m_state.F, m_state.V_o, dest_res, wGlobalLocal,
                                         m_state.energy*m_state.mesh_area)/m_state.mesh_area;
}
