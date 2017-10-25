# RLightGBM, R interface to Light Gradient Boosting Machine library

LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:

- Faster training speed and higher efficiency
- Lower memory usage
- Better accuracy
- Parallel learning supported
- Capable of handling large-scale data

For more details on LightGBM, please refer to [https://github.com/Microsoft/LightGBM](https://github.com/Microsoft/LightGBM).

## Getting Started

Low-level Interface:
```R
library(RLightGBM)
data(example.binary)

# parameters
num_iterations <- 100

config <- list(objective = "binary", 
		metric="binary_logloss,auc",
		learning_rate = 0.1,
		num_leaves = 63,
		tree_learner = "serial",
		feature_fraction = 0.8,
		bagging_freq = 5,
		bagging_fraction = 0.8,
		min_data_in_leaf = 50,
		min_sum_hessian_in_leaf = 5.0,
		verbosity = -1)

# create data handle and booster
handle.data <- lgbm.data.create(x)
lgbm.data.setField(handle.data, "label", y)

handle.booster <- lgbm.booster.create(handle.data, lapply(config, as.character))

# train for num_iterations iterations
lgbm.booster.train(handle.booster, num_iterations)

# predict
y.pred <- lgbm.booster.predict(handle.booster, x.test)

# test accuracy
sum(y.test == (y.pred > 0.5)) / length(y.test)

# save model (can be loaded again via lgbm.booster.load(filename))
lgbm.booster.save(handle.booster, filename = "/tmp/model.txt")
```

Training and prediction using [Caret](http://caret.r-forge.r-project.org/):
```R
library(caret)
library(RLightGBM)
data(iris)

model <-caretModel.LGBM()

fit <- train(Species ~ ., data = iris, method=model, verbosity = -1)
print(fit)

y.pred <- predict(fit, iris[,1:4])

##
## training using sparse matrices
##
library(Matrix)

model.sparse <- caretModel.LGBM.sparse()

# generate a sparse matrix
mat <- Matrix(as.matrix(iris[,1:4]), sparse = T)

fit <- train(data.frame(idx = 1:nrow(iris)), iris$Species, method = model.sparse, matrix = mat, verbosity = -1)
print(fit)

```

## Installation

Run these lines on Ubuntu 14.04 or later:
```sh
git clone --recursive https://github.com/bwilbertz/RLightGBM.git

cd RLightGBM
R CMD build --no-build-vignettes pkg/RLightGBM
R CMD INSTALL RLightGBM_0.1.tar.gz
```
Please note that `devtools::install_github` cannot be used for installation due to devtools not supporting git submodules.

For windows installation, run the final installation step from within R using `devtools` and `RTools`:
```R
library(devtools)

find_rtools()

install.packages("RLightGBM_0.1.tar.gz", type="source", repos=NULL)
```

## Disclaimer

This package was written in order to run some testing of LightGBM from `R` using Caret. 
There are parts which could have been done more elegant (e.g. returning proper S4 objects instead of raw pointers, more sanity checks, etc) 
and not every feature of LightGBM was tested, 
but basic functionality and support for sparse matrices are working well and without any overhead. 