# TODO: Add comment
# 
# Author: benedikt
###############################################################################


library(RLightGBM)

x <- matrix(rnorm(30), nrow = 10)
str(x)

p <- lgbm.data.create(x)

str(p)

p


