#include "SLIMData.h"

#include "igl/serialize.h"

using namespace std;

SLIMData::SLIMData(Eigen::MatrixXd& V_in, Eigen::MatrixXi& F_in) : V(V_in), F(F_in) {
  proximal_p = 0.0001;

  v_num = V.rows();
  f_num = F.rows();
  igl::doublearea(V,F,M); M /= 2.;
  mesh_area = M.sum();
}

void SLIMData::save(const std::string filename) {
   
   igl::serialize(V, "V", filename, true);
   igl::serialize(F,"F",filename);
   igl::serialize(M,"M",filename);
   igl::serialize(V_o,"V_o",filename);
   igl::serialize(v_num,"v_num",filename);
   igl::serialize(f_num,"f_num",filename);

   igl::serialize(mesh_area,"mesh_area",filename);
   igl::serialize(avg_edge_length,"avg_edge_length",filename);
   
   igl::serialize(energy,"energy", filename);

   igl::serialize(b,"b",filename);
   igl::serialize(bc,"bc",filename);
}

void SLIMData::load(const std::string filename) {
   igl::deserialize(V,"V",filename);
   igl::deserialize(F,"F",filename);
   igl::deserialize(M,"M",filename);
   igl::deserialize(V_o,"V_o",filename);
   igl::deserialize(v_num,"v_num",filename);
   igl::deserialize(f_num,"f_num",filename);
   
   igl::deserialize(mesh_area,"mesh_area",filename);
   igl::deserialize(avg_edge_length,"avg_edge_length",filename);
   
   igl::deserialize(energy,"energy", filename);

   igl::deserialize(b,"b",filename);
   igl::deserialize(bc,"bc",filename);
}