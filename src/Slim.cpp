#include "Slim.h"

#include "SLIMData.h"
#include "eigen_stl_utils.h"
#include "geometric_utils.h"
#include "LinesearchParametrizer.h"


#include <igl/boundary_loop.h>
#include <igl/cotmatrix.h>
#include <igl/edge_lengths.h>
#include <igl/local_basis.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/readOBJ.h>
#include <igl/repdiag.h>
#include <igl/vector_area_matrix.h>
#include <iostream>

#undef NDEBUG
#include <assert.h>
#define NDEBUG

using namespace std;

Slim::Slim(SLIMData& m_state) : 
      m_state(m_state), WArap_p(NULL) {
  assert (m_state.F.cols() == 3);
  
  WArap_p = new LocalWeightedArapParametrizer(m_state);
}

void Slim::precompute() {
  WArap_p->pre_calc();
  m_state.energy = WArap_p->compute_energy(m_state.V, m_state.F, m_state.V_o)/m_state.mesh_area;
}

void Slim::solve(int iter_num) {
  for (int i = 0; i < iter_num; i++) {
    cout << "iter number " << i << endl; // todo: remove me
    slim_iter();
  }
}

void Slim::slim_iter() {
  // weighted arap for riemannian metric
  LinesearchParametrizer linesearchParam(m_state);
  Eigen::MatrixXd dest_res;
  dest_res = m_state.V_o;
  WArap_p->parametrize(m_state.V,m_state.F, m_state.b,m_state.bc, dest_res);

  double old_energy = m_state.energy;

  m_state.energy = linesearchParam.parametrize(m_state.V,m_state.F, m_state.V_o, dest_res, WArap_p, m_state.energy*m_state.mesh_area)/m_state.mesh_area;
}
