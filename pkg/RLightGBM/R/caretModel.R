# TODO: Add comment
# 
# Author: benedikt
###############################################################################


caretModel.LGBM <- function() {
	m <- list()
	
	m$label <- "LightGBM, Light Gradient Boosting Machine"
	
	m$library <- c("RLightGBM", "plyr")
	
	m$type <- c("Regression", "Classification")
	
	m$parameters <- data.frame(parameter = c("num_iteration", "learning_rate", "num_leaves", "min_gain_to_split", "feature_fraction", 
					"min_sum_hessian_in_leaf", "min_data_in_leaf", "bagging_fraction", "lambda_l2"),
			class = rep("numeric", 9),
			label = c("num_iteration", "learning_rate", "num_leaves", "min_gain_to_split", "feature_fraction", 
					"min_sum_hessian_in_leaf", "min_data_in_leaf", "bagging_fraction", "lambda_l2"))
	
	m$grid <- function (x, y, len = NULL, search = "grid") {
		if (search == "grid") {
			out <- expand.grid(num_leaves = 2^seq(1, len), 
					num_iteration = floor((1:len) * 50), 
					learning_rate = c(0.1, 0.3), 
					min_gain_to_split = 0, 
					feature_fraction = c(0.6, 0.8),
					min_data_in_leaf = c(1),
					min_sum_hessian_in_leaf = c(1), 
					bagging_fraction = seq(0.5, 1, length = len), 
					lambda_l2 = c(1.0))
		}
		else {
			out <- data.frame(
					num_iteration = sample(1:1000, size = len, replace = TRUE), 
					num_leaves = as.Integer(2^sample(1:10, replace = TRUE, size = len)), 
					learning_rate = runif(len, min = 0.001, max = 0.6), 
					min_gain_to_split = runif(len, min = 0, max = 10), 
					feature_fraction = runif(len, min = 0.3, max = 0.7),
					min_data_in_leaf = as.integer(runif(len, min=1, max = 100)),
					min_sum_hessian_in_leaf = sample(0:20, size = len, replace = TRUE), 
					bagging_fraction = runif(len, min = 0.25, max = 1), 
					lambda_l2 = runif(len, min = 0.0, max = 1))
			out$num_iteration <- floor(out$num_iteration)
			out <- out[!duplicated(out), ]
		}
		out
	}
	
	m$loop <- function (grid) {
		loop <- ddply(grid, c("learning_rate", "num_leaves", "learning_rate", "feature_fraction", "min_gain_to_split",
						"min_sum_hessian_in_leaf", "min_data_in_leaf", "bagging_fraction", "lambda_l2"), function(x) c(num_iteration = max(x$num_iteration)))
		submodels <- vector(mode = "list", length = nrow(loop))
		for (i in seq(along = loop$num_iteration)) {
			index <- which(grid$num_leaves == loop$num_leaves[i] & 
							grid$learning_rate == loop$learning_rate[i] & 
							grid$min_gain_to_split == loop$min_gain_to_split[i] & 
							grid$feature_fraction == loop$feature_fraction[i] & 
							grid$min_sum_hessian_in_leaf == loop$min_sum_hessian_in_leaf[i] & 
							grid$min_data_in_leaf == loop$min_data_in_leaf[i] &
							grid$bagging_fraction == loop$bagging_fraction[i] &
							grid$lambda_l2 == loop$lambda_l2[i])
			trees <- grid[index, "num_iteration"]
			submodels[[i]] <- data.frame(num_iteration = trees[trees != 
									loop$num_iteration[i]])
		}
		list(loop = loop, submodels = submodels)
	}
	
	m$fit <- function (x, y, wts, param, lev, last, classProbs, ...) {
		if (class(x)[1] != "lgbm.data") 
			x <- as.matrix(x)
		
		config <- list(learning_rate = param$learning_rate, 
				num_leaves = param$num_leaves, 
				min_gain_to_split = param$min_gain_to_split, 
				feature_fraction = param$feature_fraction, 
				min_sum_hessian_in_leaf = param$min_sum_hessian_in_leaf,
				min_data_in_leaf = param$min_data_in_leaf, 
				bagging_fraction = param$bagging_fraction,
				lambda_l2 = param$lambda_l2,
				...)
		
		if ( param$bagging_fraction < 1.0) config$bagging_freq <- 5
		
		
		if (is.factor(y)) {
			if (length(lev) == 2) {
				y <- ifelse(y == lev[1], 1, 0)
				dat <- lgbm.data.create(x)
				lgbm.data.setField(dat, "label", y)
				
				config$application <- "binary"
				config$num_class <- 1
				
				out <- lgbm.booster.create(dat, lapply(config, as.character))
				lgbm.booster.train(out, param$num_iteration)
			}
			else {
				y <- as.numeric(y) - 1
				dat <- lgbm.data.create(x)
				lgbm.data.setField(dat, "label", y)
				
				config$application <- "multiclass"
				config$num_class <- length(lev)
				
				out <- lgbm.booster.create(dat, lapply(config, as.character))
				lgbm.booster.train(out, param$num_iteration)
			}
		}
		else {
			dat <- lgbm.data.create(x)
			lgbm.data.setField(dat, "label", y)
			
			config$application <- "regression"
			config$num_class <- 1
			
			out <- lgbm.booster.create(dat, lapply(config, as.character))
			lgbm.booster.train(out, param$num_iteration)
		}
		
		list(booster.handle = out, num_models = param$num_iteration, num_classes = config$num_class)
	}
	
	m$predict <- function (modelFit, newdata, submodels = NULL) {
		if (class(newdata)[1] != "lgbm.data") 
			newdata <- as.matrix(newdata)
		
		out <- lgbm.booster.predict(modelFit$booster.handle, newdata, 0L, modelFit$num_models)
		if (modelFit$problemType == "Classification") {
			if (length(modelFit$obsLevels) == 2) {
				out <- ifelse(out >= 0.5, modelFit$obsLevels[1], 
						modelFit$obsLevels[2])
			}
			else {
				out <- matrix(out, ncol = length(modelFit$obsLevels), 
						byrow = TRUE)
				out <- modelFit$obsLevels[apply(out, 1, which.max)]
			}
		}
		if (!is.null(submodels)) {
			tmp <- vector(mode = "list", length = nrow(submodels) + 
							1)
			tmp[[1]] <- out
			for (j in seq(along = submodels$num_iteration)) {
				tmp_pred <- lgbm.booster.predict(modelFit$booster.handle, newdata, 0L, submodels$num_iteration[j])
				if (modelFit$problemType == "Classification") {
					if (length(modelFit$obsLevels) == 2) {
						tmp_pred <- ifelse(tmp_pred >= 0.5, modelFit$obsLevels[1], 
								modelFit$obsLevels[2])
					}
					else {
						tmp_pred <- matrix(tmp_pred, ncol = length(modelFit$obsLevels), 
								byrow = TRUE)
						tmp_pred <- modelFit$obsLevels[apply(tmp_pred, 
										1, which.max)]
					}
				}
				tmp[[j + 1]] <- tmp_pred
			}
			out <- tmp
		}
		out
	}
	
	m$prob <- function (modelFit, newdata, submodels = NULL) {
		if (class(newdata)[1] != "lgbm.data") 
			newdata <- as.matrix(newdata)
		
		out <- lgbm.booster.predict(modelFit$booster.handle, newdata, 0L, modelFit$num_models)
		if (length(modelFit$obsLevels) == 2) {
			out <- cbind(out, 1 - out)
			colnames(out) <- modelFit$obsLevels
		}
		else {
			out <- matrix(out, ncol = length(modelFit$obsLevels), 
					byrow = TRUE)
			colnames(out) <- modelFit$obsLevels
		}
		out <- as.data.frame(out)
		if (!is.null(submodels)) {
			tmp <- vector(mode = "list", length = nrow(submodels) + 
							1)
			tmp[[1]] <- out
			for (j in seq(along = submodels$num_iteration)) {
				tmp_pred <- lgbm.booster.predict(modelFit$booster.handle, newdata, 0L, submodels$num_iteration[j])
				if (length(modelFit$obsLevels) == 2) {
					tmp_pred <- cbind(tmp_pred, 1 - tmp_pred)
					colnames(tmp_pred) <- modelFit$obsLevels
				}
				else {
					tmp_pred <- matrix(tmp_pred, ncol = length(modelFit$obsLevels), 
							byrow = TRUE)
					colnames(tmp_pred) <- modelFit$obsLevels
				}
				tmp_pred <- as.data.frame(tmp_pred)
				tmp[[j + 1]] <- tmp_pred
			}
			out <- tmp
		}
		out
	}
	
	m$predictors <- function (x, ...) {
		## imp <- xgb.importance(x$xNames, model = x)
		## x$xNames[x$xNames %in% imp$Feature]
		stop("model$predictors not implemented yet.")
	}
	
	m$varImp <- function (object, numTrees = NULL, ...) {
		## imp <- xgb.importance(object$xNames, model = object)
		## imp <- as.data.frame(imp)[, 1:2]
		## rownames(imp) <- as.character(imp[, 1])
		## imp <- imp[, 2, drop = FALSE]
		## colnames(imp) <- "Overall"
		## imp
		stop("model$varImp not implemented yet.")
	}
	
	m$levels <- function (x) x$obsLevels
	
	m$tags <- c("Tree-Based Model", "Boosting", "Ensemble Model", "Implicit Feature Selection")
	
	m$sort <- function (x) {
		x[order(x$num_iteration, x$num_leaves, x$learning_rate, x$min_gain_to_split, x$feature_fraction, 
						x$min_sum_hessian_in_leaf, x$bagging_fraction, x$lambda_l2), ]
	}
	
	return(m)
}

caretModel.LGBM.sparse <- function() {
	m <- caretModel.LGBM()
	
	m$label <- "LightGBM, Light Gradient Boosting Machine for sparse input"
	
	m$library <- c("RLightGBM", "plyr", "SparseM")

	m$fit <- function (x, y, wts, param, lev, last, classProbs, matrix, force.CSR = F, ...) {
		if ( is.null(x$idx) ) stop("Input x must contain a column named 'idx' counting the rows of matrix.")
		#if (class(matrix) != "dgCMatrix") stop("Expect matrix to be of class dgCMatrix.")
		
		config <- list(learning_rate = param$learning_rate, 
				num_leaves = param$num_leaves, 
				min_gain_to_split = param$min_gain_to_split, 
				feature_fraction = param$feature_fraction, 
				min_sum_hessian_in_leaf = param$min_sum_hessian_in_leaf,
				min_data_in_leaf = param$min_data_in_leaf,
				bagging_fraction = param$bagging_fraction,
				lambda_l2 = param$lambda_l2,
				...)
		
		if ( param$bagging_fraction < 1.0) config$bagging_freq <- 5
		
		m <- matrix[x$idx,]
		
		if ( force.CSR && ! is.matrix.csr(m) ) m <- as.matrix.csr(m, nrow = nrow(x))
		
		if ( is.matrix.csr(m) )  
			dat <- lgbm.data.create.CSR(m@ja-1, m@ia-1, m@ra, m@dimension)
		else if ( is.matrix.csc(m) )
			dat <- lgbm.data.create.CSC(m@ja-1, m@ia-1, m@ra, m@dimension)
		else if ( class(matrix) == "dgCMatrix" )
			dat <- lgbm.data.create.CSC(m@i, m@p, m@x, m@Dim)
		else
			stop(paste("Unsupported sparse matrix type:", class(matrix)))
		
		if (is.factor(y)) {
			if (length(lev) == 2) {
				y <- ifelse(y == lev[1], 1, 0)				
				lgbm.data.setField(dat, "label", y)
				
				config$application <- "binary"
				config$num_class <- 1
				
				out <- lgbm.booster.create(dat, lapply(config, as.character))
				lgbm.booster.train(out, param$num_iteration)
			}
			else {
				y <- as.numeric(y) - 1				
				lgbm.data.setField(dat, "label", y)
				
				config$application <- "multiclass"
				config$num_class <- length(lev)
				
				out <- lgbm.booster.create(dat, lapply(config, as.character))
				lgbm.booster.train(out, param$num_iteration)
			}
		}
		else {			
			lgbm.data.setField(dat, "label", y)
			
			config$application <- "regression"
			config$num_class <- 1
			
			out <- lgbm.booster.create(dat, lapply(config, as.character))
			lgbm.booster.train(out, param$num_iteration)
		}
		
		list(booster.handle = out, num_models = param$num_iteration, num_classes = config$num_class, matrix = matrix)
	}
	
	m$predict <- function (modelFit, newdata, submodels = NULL) {
		m <- as.matrix.csr(modelFit$matrix[newdata$idx,], nrow=length(newdata$idx))
		
		out <- lgbm.booster.predict.CSR(modelFit$booster.handle, m@ja-1, m@ia-1, m@ra, m@dimension, 0L, modelFit$num_models)
		if (modelFit$problemType == "Classification") {
			if (length(modelFit$obsLevels) == 2) {
				out <- ifelse(out >= 0.5, modelFit$obsLevels[1], 
						modelFit$obsLevels[2])
			}
			else {
				out <- matrix(out, ncol = length(modelFit$obsLevels), 
						byrow = TRUE)
				out <- modelFit$obsLevels[apply(out, 1, which.max)]
			}
		}
		if (!is.null(submodels)) {
			tmp <- vector(mode = "list", length = nrow(submodels) + 
							1)
			tmp[[1]] <- out
			for (j in seq(along = submodels$num_iteration)) {
				tmp_pred <- lgbm.booster.predict.CSR(modelFit$booster.handle, m@ja-1, m@ia-1, m@ra, m@dimension, 0L, submodels$num_iteration[j])
				if (modelFit$problemType == "Classification") {
					if (length(modelFit$obsLevels) == 2) {
						tmp_pred <- ifelse(tmp_pred >= 0.5, modelFit$obsLevels[1], 
								modelFit$obsLevels[2])
					}
					else {
						tmp_pred <- matrix(tmp_pred, ncol = length(modelFit$obsLevels), 
								byrow = TRUE)
						tmp_pred <- modelFit$obsLevels[apply(tmp_pred, 
										1, which.max)]
					}
				}
				tmp[[j + 1]] <- tmp_pred
			}
			out <- tmp
		}
		out
	}
	
	m$prob <- function (modelFit, newdata, submodels = NULL) {
		m <- as.matrix.csr(modelFit$matrix[newdata$idx,], nrow=length(newdata$idx))
		
		out <- lgbm.booster.predict.CSR(modelFit$booster.handle, m@ja-1, m@ia-1, m@ra, m@dimension, 0L, modelFit$num_models)
		if (length(modelFit$obsLevels) == 2) {
			out <- cbind(out, 1 - out)
			colnames(out) <- modelFit$obsLevels
		}
		else {
			out <- matrix(out, ncol = length(modelFit$obsLevels), 
					byrow = TRUE)
			colnames(out) <- modelFit$obsLevels
		}
		out <- as.data.frame(out)
		if (!is.null(submodels)) {
			tmp <- vector(mode = "list", length = nrow(submodels) + 
							1)
			tmp[[1]] <- out
			for (j in seq(along = submodels$num_iteration)) {
				tmp_pred <- lgbm.booster.predict.CSR(modelFit$booster.handle, m@ja-1, m@ia-1, m@ra, m@dimension, 0L, submodels$num_iteration[j])
				if (length(modelFit$obsLevels) == 2) {
					tmp_pred <- cbind(tmp_pred, 1 - tmp_pred)
					colnames(tmp_pred) <- modelFit$obsLevels
				}
				else {
					tmp_pred <- matrix(tmp_pred, ncol = length(modelFit$obsLevels), 
							byrow = TRUE)
					colnames(tmp_pred) <- modelFit$obsLevels
				}
				tmp_pred <- as.data.frame(tmp_pred)
				tmp[[j + 1]] <- tmp_pred
			}
			out <- tmp
		}
		out
	}
	
	
	return(m)
}

