#include "Slim.h"

#include "Param_State.h"
#include "eigen_stl_utils.h"
#include "parametrization_utils.h"
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

Slim::Slim(Param_State* m_state) : 
      m_state(m_state), WArap_p(NULL) {
  assert (m_state->F.cols() == 3);
  
  WArap_p = new LocalWeightedArapParametrizer(m_state);
}

void Slim::precompute() {
  WArap_p->pre_calc();
}

void Slim::solve(Eigen::MatrixXd& outV, int iter_num) {
  for (int i = 0; i < iter_num; i++) {
    single_line_search_arap();
  }
}

void Slim::get_linesearch_params(Eigen::MatrixXd& dest_res,
                                                        Energy** param_energy) {
  dest_res = m_state->uv;
  WArap_p->parametrize(m_state->V,m_state->F, m_state->b,m_state->bc, dest_res);
  *param_energy = WArap_p;
}

void Slim::single_line_search_arap() {
  // weighted arap for riemannian metric
  LinesearchParametrizer linesearchParam(m_state);
  Eigen::MatrixXd dest_res;
  Energy* param_energy = NULL;
  get_linesearch_params(dest_res, &param_energy);

  Eigen::MatrixXd old_uv = m_state->uv;
  double old_energy = m_state->energy;

  m_state->energy = linesearchParam.parametrize(m_state->V,m_state->F, m_state->uv, dest_res, param_energy, m_state->energy*m_state->mesh_area)/m_state->mesh_area;
}
