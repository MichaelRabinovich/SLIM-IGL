#ifndef geometric_utils_H
#define geometric_utils_H

#include "igl/igl_inline.h"
#ifdef HAS_GUI
#include <igl/viewer/Viewer.h>
#endif

#include <igl/arap.h>
#include <Eigen/Core>
#include <set>
#include <tuple>
#include <vector>

#include "SLIMData.h"

using namespace std;

void compute_surface_gradient_matrix(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                                     const Eigen::MatrixXd& F1, const Eigen::MatrixXd& F2,
                                     Eigen::SparseMatrix<double>& D1, Eigen::SparseMatrix<double>& D2);

bool tutte_on_circle(const Eigen::MatrixXd& V,
              const Eigen::MatrixXi& F,
              Eigen::MatrixXd& uv);

void map_vertices_to_circle_area_normalized(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::VectorXi& bnd,
  Eigen::MatrixXd& UV);

int get_euler_char(const Eigen::MatrixXd& V,
              const Eigen::MatrixXi& F);

#endif // geometric_utils_H
