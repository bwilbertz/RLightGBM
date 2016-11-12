/*
 * RLightGBM.h
 *
 *  Created on: 11.11.2016
 *      Author: benedikt
 */

#ifndef RLIGHTGBM_H_
#define RLIGHTGBM_H_


#include <LightGBM/api.hpp>
#include <LightGBM/c_api.h>

#define XPTR_TYPE(X)  Rcpp::XPtr < X >
#define XPTR(X,Y, Z)  XPTR_TYPE(X) (Y, Z)

#define XPTR_BOOSTER(X) Rcpp::XPtr < void, Rcpp::PreserveStorage, delete_booster< void > > (X)
#define XPTR_DATA(X) Rcpp::XPtr < void, Rcpp::PreserveStorage, delete_dataset< void > > (X)

#define THROW_LGBM_ERROR Rcpp::stop (LGBM_GetLastError())

#define RUN_OR_DIE(X) if (X == -1) THROW_LGBM_ERROR

#include <RcppCommon.h>

namespace Rcpp {
template<> SEXP wrap(const ::LightGBM::Booster&);

template<> SEXP wrap(const ::LightGBM::Dataset&);

//template<> ::LightGBM::Booster* as(SEXP);

//template<> ::LightGBM::Dataset* as(SEXP);
}

#endif /* RLIGHTGBM_H_ */
