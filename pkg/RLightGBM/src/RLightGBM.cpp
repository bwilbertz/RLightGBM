#include "RLightGBM.h"

#include <Rcpp.h>
#include <Rdefines.h>

#define COL_MAJOR 0
#define ROW_MAJOR 1

using namespace Rcpp;

// [[Rcpp::export(name="lgbm.data.create")]]
Rcpp::XPtr<::LightGBM::Dataset> RLGBM_CreateDatasetFromMat(
    NumericMatrix x, CharacterVector params = "num_model_predict=-1") {

  return Rcpp::XPtr < ::LightGBM::Dataset
      > (CreateDatasetFromMat(&x[0], C_API_DTYPE_FLOAT64, x.nrow(), x.ncol(),
                              COL_MAJOR,
                              (Rcpp::as < std::string > (params)).c_str(),
                              nullptr));
}

// [[Rcpp::export(name="lgbm.data.create.CSR")]]
Rcpp::XPtr<::LightGBM::Dataset> RLGBM_CreateDatasetFromCSR(
    IntegerVector i, IntegerVector p, NumericVector x, IntegerVector dim,
    CharacterVector params = "num_model_predict=-1") {
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
    CharacterVector params = "num_model_predict=-1") {
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

  std::vector<const LightGBM::Dataset*> p_valid_datas;
  std::vector < std::string > p_valid_names;

  return Rcpp::XPtr < ::LightGBM::Booster
      > (new LightGBM::Booster(data_handle, p_valid_datas, p_valid_names,
                               (Rcpp::as < std::string > (params)).c_str()));
}

// [[Rcpp::export(name="lgbm.booster.create")]]
Rcpp::XPtr<::LightGBM::Booster> RLGBM_CreateBooster(
    Rcpp::XPtr<::LightGBM::Dataset> data_handle, List config) {

  std::vector<const LightGBM::Dataset*> p_valid_datas;
  std::vector < std::string > p_valid_names;

  std::unordered_map < std::string, std::string > params;
  CharacterVector keys = config.names();

  for (int i = 0; i < config.size(); i++) {
    std::string key = Rcpp::as < std::string > (keys[i]);
    params[key] = Rcpp::as < std::string > (config[key]);
  }

  return Rcpp::XPtr < ::LightGBM::Booster
      > (new LightGBM::Booster(data_handle, p_valid_datas, p_valid_names,
                               params));
}

// [[Rcpp::export(name="lgbm.booster.load")]]
Rcpp::XPtr<::LightGBM::Booster> RLGBM_CreateBoosterFromFile(
    CharacterVector filename) {

  return Rcpp::XPtr < ::LightGBM::Booster
      > (new LightGBM::Booster((Rcpp::as < std::string > (filename)).c_str()));
}

// [[Rcpp::export(name="lgbm.booster.train")]]
void RLGBM_TrainBooster(Rcpp::XPtr<::LightGBM::Booster> booster_handle,
                        int num_iters, int eval_iters = 0) {

  for (int i = 0; i < num_iters; i++) {
    booster_handle->TrainOneIter(
        eval_iters > 0 ? (i + 1) % eval_iters == 0 : false);
  }

  if ( booster_handle->NumberOfSubModels() ==  0) {
    Rcpp::warning("Something went wrong: The resulting model contains 0 trees!");
  }
}

void preparePrediction(Rcpp::XPtr<::LightGBM::Booster> booster_handle,
                       int num_used_iterations, int predict_type) {
  const int num_models = booster_handle->NumberOfSubModels();

  if (num_models == 0) {
    Rcpp::stop("Trying to run prediction on a model containing 0 trees!");
  }

  const int num_class = booster_handle->NumberOfClasses();

  if (num_class < 1) {
    Rcpp::warning("Number of classes is < 1");
  } else {
    if (num_used_iterations < 0) {
      num_used_iterations = num_models / num_class;
    } else if (num_used_iterations * num_class > num_models) {
      num_used_iterations = num_models / num_class;
      Rcpp::warning(
          "num_used_iterations exceeding number of trained trees. Reducing num_used_iterations to %d.",
          num_used_iterations);

    }
  }

  // PrepareForPrediction calls SetNumUsedModel, which has special logic and divides by num_class again...
  booster_handle->PrepareForPrediction(num_used_iterations * num_class,
                                       predict_type);
}

// [[Rcpp::export(name="lgbm.booster.predict")]]
NumericVector RLGBM_PredictFromMat(
    Rcpp::XPtr<::LightGBM::Booster> booster_handle, NumericMatrix x,
    int predict_type = 0, int num_used_iterations = -1) {

  preparePrediction(booster_handle, num_used_iterations, predict_type);

  auto get_row_fun = RowPairFunctionFromDenseMatric(&x[0], x.nrow(), x.ncol(),
                                                    C_API_DTYPE_FLOAT64,
                                                    COL_MAJOR);
  const int num_class = booster_handle->NumberOfClasses();


  NumericVector out(num_class * x.nrow());

#pragma omp parallel for schedule(guided)
  for (int i = 0; i < x.nrow(); ++i) {
    auto one_row = get_row_fun(i);
    auto predicton_result = booster_handle->Predict(one_row);
    for (int j = 0; j < num_class; ++j) {
      out[i * num_class + j] = predicton_result[j];
    }
  }

  return out;
}

// [[Rcpp::export(name="lgbm.booster.predict.CSR")]]
NumericVector RLGBM_PredictFromCSR(
    Rcpp::XPtr<::LightGBM::Booster> booster_handle, IntegerVector i,
    IntegerVector p, NumericVector x, IntegerVector dim, int predict_type = 0,
    int num_used_iterations = -1) {
  R_CHECK(p.length() == dim[0] + 1);

  preparePrediction(booster_handle, num_used_iterations, predict_type);

  auto get_row_fun = RowFunctionFromCSR(&p[0], C_API_DTYPE_INT32, &i[0], &x[0],
                                        C_API_DTYPE_FLOAT64, p.length(),
                                        x.length());

  const int num_class = booster_handle->NumberOfClasses();

  NumericVector out(num_class * dim[0]);

#pragma omp parallel for schedule(guided)
  for (int i = 0; i < dim[0]; ++i) {
    auto one_row = get_row_fun(i);
    auto predicton_result = booster_handle->Predict(one_row);
    for (int j = 0; j < num_class; ++j) {
      out[i * num_class + j] = predicton_result[j];
    }
  }

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

