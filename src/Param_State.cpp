#include "Param_State.h"

#include "igl/serialize.h"

using namespace std;

void Param_State::save(const std::string filename) {
   
   igl::serialize(V, "V", filename, true);
   igl::serialize(F,"F",filename);
   igl::serialize(M,"M",filename);
   igl::serialize(uv,"uv",filename);
   igl::serialize(v_num,"v_num",filename);
   igl::serialize(f_num,"f_num",filename);

   igl::serialize(mesh_area,"mesh_area",filename);
   igl::serialize(avg_edge_length,"avg_edge_length",filename);
   
   igl::serialize(energy,"energy", filename);
   igl::serialize(method,"method",filename);

   igl::serialize(b,"b",filename);
   igl::serialize(bc,"bc",filename);
}

void Param_State::load(const std::string filename) {
   igl::deserialize(V,"V",filename);
   igl::deserialize(F,"F",filename);
   igl::deserialize(M,"M",filename);
   igl::deserialize(uv,"uv",filename);
   igl::deserialize(v_num,"v_num",filename);
   igl::deserialize(f_num,"f_num",filename);
   
   igl::deserialize(mesh_area,"mesh_area",filename);
   igl::deserialize(avg_edge_length,"avg_edge_length",filename);
   
   igl::deserialize(energy,"energy", filename);
   igl::deserialize(method,"method",filename);

   igl::deserialize(b,"b",filename);
   igl::deserialize(bc,"bc",filename);
}