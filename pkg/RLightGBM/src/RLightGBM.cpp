#include "RLightGBM.h"

#include <Rcpp.h>
#include <Rdefines.h>

#define COL_MAJOR 0
#define ROW_MAJOR 1

using namespace Rcpp;

// [[Rcpp::export(name="lgbm.data.create")]]
Rcpp::XPtr<::LightGBM::Dataset> RLGBM_CreateDatasetFromMat(
    NumericMatrix x, CharacterVector params = "num_iteration_predict=-1") {

  return Rcpp::XPtr < ::LightGBM::Dataset
      > (CreateDatasetFromMat(&x[0], C_API_DTYPE_FLOAT64, x.nrow(), x.ncol(),
                              COL_MAJOR,
                              (Rcpp::as < std::string > (params)).c_str(),
                              nullptr));
}

// [[Rcpp::export(name="lgbm.data.create.CSR")]]
Rcpp::XPtr<::LightGBM::Dataset> RLGBM_CreateDatasetFromCSR(
    IntegerVector i, IntegerVector p, NumericVector x, IntegerVector dim,
    CharacterVector params = "num_iteration_predict=-1") {
  R_CHECK(p.length() == dim[0]+1);

  return Rcpp::XPtr < ::LightGBM::Dataset
      > (CreateDatasetFromCSR(&p[0], C_API_DTYPE_INT32, &i[0], &x[0],
                              C_API_DTYPE_FLOAT64, p.length(), x.length(),
                              dim[1],
                              (Rcpp::as < std::string > (params)).c_str(),
                              nullptr));
}

// [[Rcpp::export(name="lgbm.data.create.CSC")]]
Rcpp::XPtr<::LightGBM::Dataset> RLGBM_CreateDatasetFromCSC(
    IntegerVector i, IntegerVector p, NumericVector x, IntegerVector dim,
    CharacterVector params = "num_iteration_predict=-1") {
  R_CHECK(p.length() == dim[1]+1);


  return Rcpp::XPtr < ::LightGBM::Dataset
      > (CreateDatasetFromCSC(&p[0], C_API_DTYPE_INT32, &i[0], &x[0],
                              C_API_DTYPE_FLOAT64, p.length(), x.length(),
                              dim[0],
                              (Rcpp::as < std::string > (params)).c_str(),
                              nullptr));
}

// [[Rcpp::export(name="lgbm.data.setField")]]
void RLGBM_DatasetSetField(Rcpp::XPtr<::LightGBM::Dataset> data_handle,
                           CharacterVector field_name, NumericVector x) {

  const int n = x.size();

  std::vector<float> x32(n);

  for (int i = 0; i < n; i++) {
    x32[i] = static_cast<float>(x[i]);
  }

  bool is_success = data_handle->SetFloatField(
      (Rcpp::as < std::string > (field_name)).c_str(), &x32[0], n);

  if (!is_success) {
    Rcpp::stop(
        "Unable to set field: " + (Rcpp::as < std::string > (field_name)));
  }
}

// [[Rcpp::export(name="lgbm.booster.create.str")]]
Rcpp::XPtr<::LightGBM::Booster> RLGBM_CreateBoosterFromString(
    Rcpp::XPtr<::LightGBM::Dataset> data_handle, CharacterVector params =
        "task=train") {

  return Rcpp::XPtr < ::LightGBM::Booster
      > (new LightGBM::Booster(data_handle,
                               (Rcpp::as < std::string > (params)).c_str()));
}

// [[Rcpp::export(name="lgbm.booster.create")]]
Rcpp::XPtr<::LightGBM::Booster> RLGBM_CreateBooster(
    Rcpp::XPtr<::LightGBM::Dataset> data_handle, List config) {

  std::unordered_map < std::string, std::string > params;
  CharacterVector keys = config.names();

  for (int i = 0; i < config.size(); i++) {
    std::string key = Rcpp::as < std::string > (keys[i]);
    params[key] = Rcpp::as < std::string > (config[key]);
  }

  return Rcpp::XPtr < ::LightGBM::Booster
      > (new LightGBM::Booster(data_handle, params));
}

// [[Rcpp::export(name="lgbm.booster.load")]]
Rcpp::XPtr<::LightGBM::Booster> RLGBM_CreateBoosterFromFile(
    CharacterVector filename) {

  return Rcpp::XPtr < ::LightGBM::Booster
      > (new LightGBM::Booster((Rcpp::as < std::string > (filename)).c_str()));
}

// [[Rcpp::export(name="lgbm.booster.train")]]
void RLGBM_TrainBooster(Rcpp::XPtr<::LightGBM::Booster> booster_handle,
                        int num_iters) {

  for (int i = 0; i < num_iters; i++) {
    booster_handle->TrainOneIter();
  }

  if ( booster_handle->NumberOfSubModels() ==  0) {
    Rcpp::warning("Something went wrong: The resulting model contains 0 trees!");
  }
}

// [[Rcpp::export(name="lgbm.booster.predict")]]
NumericVector RLGBM_PredictFromMat(
    Rcpp::XPtr<::LightGBM::Booster> booster_handle, NumericMatrix x,
    int predict_type = 0, int num_used_iterations = -1) {

  auto get_row_fun = RowPairFunctionFromDenseMatric(&x[0], x.nrow(), x.ncol(),
                                                    C_API_DTYPE_FLOAT64,
                                                    COL_MAJOR);
  const int num_class = booster_handle->NumberOfClasses();

  NumericVector out(num_class * x.nrow());

  int64_t out_len;

  booster_handle->Predict(num_used_iterations, predict_type, x.nrow(),
                          get_row_fun, &out[0], &out_len);

  R_CHECK(out_len == num_class * x.nrow());

  return out;
}

// [[Rcpp::export(name="lgbm.booster.predict.CSR")]]
NumericVector RLGBM_PredictFromCSR(
    Rcpp::XPtr<::LightGBM::Booster> booster_handle, IntegerVector i,
    IntegerVector p, NumericVector x, IntegerVector dim, int predict_type = 0,
    int num_used_iterations = -1) {
  R_CHECK(p.length() == dim[0] + 1);

  auto get_row_fun = RowFunctionFromCSR(&p[0], C_API_DTYPE_INT32, &i[0], &x[0],
  C_API_DTYPE_FLOAT64, p.length(), x.length());

  const int num_class = booster_handle->NumberOfClasses();

  NumericVector out(num_class * dim[0]);

  int64_t out_len;

  booster_handle->Predict(num_used_iterations, predict_type, dim[0],
                          get_row_fun, &out[0], &out_len);

  R_CHECK(out_len == num_class * dim[0]);

  return out;
}

//// [[Rcpp::export(name="lgbm.booster.predict.CSC")]]
//SEXP RLGBM_PredictFromCSC(SEXP booster_handle, IntegerVector i, IntegerVector p,
//                          NumericVector x, IntegerVector dim, int predict_type,
//                          int num_used_iterations) {
//
//  Rcpp::XPtr < ::LightGBM::Booster >ptr(booster_handle);
//
//  ptr->PrepareForPrediction(num_used_iterations, predict_type);
//
//  auto get_row_fun = RowFunctionFromCSC(&p[0], C_API_DTYPE_INT32, &i[0], &x[0],
//                                          C_API_DTYPE_FLOAT64, p.length(),
//                                          dim[0] * dim[1]);
//  const int num_class = ptr->NumberOfClasses();
//
//  if (num_class < 1) {
//    Rcpp::warning("Number of classes is < 1");
//  }
//
//  NumericVector out(num_class * x.nrow());
//
//#pragma omp parallel for schedule(guided)
//  for (int i = 0; i < dim[0]; ++i) {
//    auto one_row = get_row_fun(i);
//    auto predicton_result = ptr->Predict(one_row);
//    for (int j = 0; j < num_class; ++j) {
//      out[i * num_class + j] = predicton_result[j];
//    }
//  }
//
//  return out;
//}

// [[Rcpp::export(name="lgbm.booster.save")]]
void RLGBM_SaveBoosterModel(Rcpp::XPtr<::LightGBM::Booster> booster_handle,
                            CharacterVector filename, int num_used_iterations = -1) {

  booster_handle->SaveModelToFile(
      num_used_iterations, (Rcpp::as < std::string > (filename)).c_str());
}

// [[Rcpp::export(name="lgbm.booster.dump")]]
CharacterVector RLGBM_DumpBoosterModel(Rcpp::XPtr<::LightGBM::Booster> booster_handle,
                            int num_iteration = -1) {

  return booster_handle->DumpModel(num_iteration);
}

