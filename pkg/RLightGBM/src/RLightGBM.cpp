#include "RLightGBM.h"

#include <Rcpp.h>
#include <Rdefines.h>


using namespace Rcpp;


// [[Rcpp::export(name="lgbm.data.create")]]
SEXP RLGBM_CreateDatasetFromMat(NumericMatrix x, CharacterVector params = "num_model_predict=-1") {
  //DatesetHandle out;

  const LightGBM::Dataset* handle = CreateDatasetFromMat(&x[0], C_API_DTYPE_FLOAT64, x.nrow(), x.ncol(),
                                0, (Rcpp::as<std::string>(params)).c_str(), nullptr);

  //RUN_OR_DIE(
  //    LGBM_CreateDatasetFromMat(&x[0], C_API_DTYPE_FLOAT64, x.nrow(), x.ncol(),
  //                              0, "", nullptr, &out));

  return Rcpp::wrap(*handle);
}


/////////////////////
// Rcpp extensions //
/////////////////////

namespace Rcpp {
template<> SEXP wrap(const ::LightGBM::Booster& b) {
  return XPTR(const ::LightGBM::Booster, &b, true);
}

template<> SEXP wrap(const ::LightGBM::Dataset& d) {
  return XPTR(const ::LightGBM::Dataset, &d, true);
}

//// TODO return xptr in order to avoid mem leak or remove this at all...
//template<> NetParameter as(SEXP n) {
//  S4 msg(n);
//
//  Rcpp::XPtr < ::google::protobuf::Message, PreserveStorage, my_delete_finalizer< google::protobuf::Message > > ptr = msg.slot("pointer");
//
//  //fprintf(stderr, "pointer %p\n", ptr.get());
//
//  //  const NetParameter* source =
//  //      ::google::protobuf::internal::dynamic_cast_if_available<
//  //          const NetParameter*>(ptr.get());
//  //
//  //  fprintf(stderr, "pointer %p\n", source);
//
//  NetParameter* net = new NetParameter();
//  net->CopyFrom(*ptr);
//
//  return *net;
//}
}

