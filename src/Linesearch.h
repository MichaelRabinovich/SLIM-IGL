#ifndef LINESEARCH_H
#define LINESEARCH_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>
#include <set>
#include <vector>

#include "WeightedGlobalLocal.h"
#include "SLIMData.h"

class Linesearch {

public:
  Linesearch(SLIMData& param_state);

  // A simple backtracking linesearch
  // Input:
  //    V #V by 3 list of mesh positions (original mesh)
  //    F #F by simplex-size list of triangle|tet indices into V
  //    cur_v #V by dim list of the current mesh positions
  //    dst_v #V by dim list the destination mesh positions (d = cur_v - dst_v)
  //    energy A class used to evaluate the current objective value
  //    cur_energy (OPTIONAL) The current objective funcational at cur_v
  //
  // Output:
  //    cur_v A new set of vertices such that energy(cur_v) < previous energy
  //    Returns the new objective value
  double compute( const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    Eigen::MatrixXd& cur_v,
    Eigen::MatrixXd& dst_v,
    WeightedGlobalLocal* energy,
    double cur_energy = -1);

private:

  double compute_max_step_from_singularities(const Eigen::MatrixXd& uv,
                                            const Eigen::MatrixXi& F,
                                            Eigen::MatrixXd& d);

  double line_search(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                              Eigen::MatrixXd& uv, const Eigen::MatrixXd& d,
                              double step_size, WeightedGlobalLocal* energy, double cur_energy);

  double get_min_pos_root_2D(const Eigen::MatrixXd& uv,const Eigen::MatrixXi& F,
            Eigen::MatrixXd& direc, int f);

  double get_min_pos_root_3D(const Eigen::MatrixXd& uv,const Eigen::MatrixXi& F,
            Eigen::MatrixXd& direc, int f);

  double get_smallest_pos_quad_zero(double a,double b, double c);
  int SolveP3(std::vector<double>& x,double a,double b,double c);

  SLIMData& m_state;
};

#endif // LINESEARCH_H
