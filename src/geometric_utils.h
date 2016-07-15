#ifndef GEOMETRIC_UTILS_H
#define GEOMETRIC_UTILS_H

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

int get_euler_char(const Eigen::MatrixXd& V,
              const Eigen::MatrixXi& F);


void get_flips(const Eigen::MatrixXd& V,
               const Eigen::MatrixXi& F,
               const Eigen::MatrixXd& uv,
               std::vector<int>& flip_idx);

int count_flips(const Eigen::MatrixXd& V,
              const Eigen::MatrixXi& F,
              const Eigen::MatrixXd& uv);

void compute_surface_gradient_matrix(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                                     const Eigen::MatrixXd& F1, const Eigen::MatrixXd& F2,
                                     Eigen::SparseMatrix<double>& D1, Eigen::SparseMatrix<double>& D2);

void compute_tet_grad_matrix(const Eigen::MatrixXd& V, const Eigen::MatrixXi& T,
                            Eigen::SparseMatrix<double>& Dx, Eigen::SparseMatrix<double>& Dy, Eigen::SparseMatrix<double>& Dz,
                            bool uniform);

#endif // GEOMETRIC_UTILS_H
