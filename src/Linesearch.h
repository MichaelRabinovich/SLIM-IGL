#ifndef LINESEARCH_H
#define LINESEARCH_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>
#include <set>
#include <vector>

#include "Energy.h"
#include "SLIMData.h"

class Linesearch {

public:
  // does precomputation if it was not already done
  Linesearch(SLIMData& param_state);

  double compute( const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    Eigen::MatrixXd& cur_uv,
    Eigen::MatrixXd& dst_uv,
    Energy* energy,
    double cur_energy = -1);

  double compute_max_step_from_singularities(const Eigen::MatrixXd& uv,
                                            const Eigen::MatrixXi& F,
                                            Eigen::MatrixXd& d);

private:
  double line_search(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                              Eigen::MatrixXd& uv, const Eigen::MatrixXd& d,
                              double step_size, Energy* energy, double cur_energy);

  double get_min_pos_root_2D(const Eigen::MatrixXd& uv,const Eigen::MatrixXi& F,
            Eigen::MatrixXd& direc, int f);

  double get_min_pos_root_3D(const Eigen::MatrixXd& uv,const Eigen::MatrixXi& F,
            Eigen::MatrixXd& direc, int f);

  double get_smallest_pos_quad_zero(double a,double b, double c);
  int SolveP3(std::vector<double>& x,double a,double b,double c);

  SLIMData& m_state;
};

#endif // LINESEARCH_H
