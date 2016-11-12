/*
 * RLightGBM.h
 *
 *  Created on: 11.11.2016
 *      Author: benedikt
 */

#ifndef RLIGHTGBM_H_
#define RLIGHTGBM_H_

#include <LightGBM/api.hpp>

#define XPTR_TYPE(X)  Rcpp::XPtr < X >
#define XPTR(X,Y, Z)  XPTR_TYPE(X) (Y, Z)

#define R_CHECK(condition)                                   \
  if (!(condition)) Rcpp::stop("Check failed: " #condition \
     " at %s, line %d .\n", __FILE__,  __LINE__);


#endif /* RLIGHTGBM_H_ */
