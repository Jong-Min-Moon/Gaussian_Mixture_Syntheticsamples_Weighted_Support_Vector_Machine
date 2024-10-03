library(smotefamily) # for smote algorithms


get_idx_confusion_mat <- function(confusion_mat){
  idx.truth.neg <- (1:2)[rownames(confusion_mat) == "-1"]
  idx.truth.pos <- (1:2)[rownames(confusion_mat) == "1"]
  idx.pred.neg <- (1:2)[colnames(confusion_mat) == "-1"]
  idx.pred.pos <- (1:2)[colnames(confusion_mat) == "1"]
  return(list(
    truth.neg = idx.truth.neg,
    truth.pos = idx.truth.pos,
    pred.neg = idx.pred.neg,
    pred.pos = idx.pred.pos
  ))
}

get_four_counts <- function(confusion_mat){
  idx <- get_idx_confusion_mat(confusion_mat)
  TP <- confusion_mat[idx$truth.pos, idx$pred.pos]
  TN <- confusion_mat[idx$truth.neg, idx$pred.neg]
  FP <- confusion_mat[idx$truth.neg, idx$pred.pos]
  FN <- confusion_mat[idx$truth.pos, idx$pred.neg]
  return(list(
    TP = TP,
    TN = TN,
    FP = FP,
    FN = FN
  ))
}

get_TPR <- function(confusion_mat){
  four_counts <- get_four_counts(confusion_mat)
  TPR <- four_counts$TP / (four_counts$TP + four_counts$FN)
  return(TPR)
}

get_TNR <- function(confusion_mat){
  four_counts <- get_four_counts(confusion_mat)
  TNR <- four_counts$TN / (four_counts$TN + four_counts$FP)
  return(TNR)
}

get_precision <- function(confusion_mat){
  four_counts <- get_four_counts(confusion_mat)
  precision <- four_counts$TP / (four_counts$TP + four_counts$FP)
  return(precision)
}

get_gmean <- function(confusion_mat){
  gmean <- sqrt(get_TPR(confusion_mat) * get_TNR(confusion_mat))
  return(gmean)
} 

get_accuracy <- function(confusion_mat){
  four_counts <- get_four_counts(confusion_mat)
  ACC <- (four_counts$TP + four_counts$TN) / (four_counts$TP + four_counts$TN + four_counts$FP + four_counts$FN)
  return(ACC)
}


#get_fmeasure <- function(confusion_mat){
#  precision = get_precision(confusion_mat)
#  recall = get_TPR(confusion_mat)
#  beta = 2
#  f2_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
#  return(f2_score)
#}
get_fmeasure <- function(confusion_mat) {
  # Set beta to 2
  four_counts <- get_four_counts(confusion_mat)
  beta = 2
  
  # Compute the F2 score using the given formula
  f2_score = ((1 + beta^2) * four_counts$TP) /
  (
    (1 + beta^2)*four_counts$TP
    +
    beta^2 * four_counts$FN
    +
    four_counts$FP)
  
  return(f2_score)
}

data_onefold_split <-function(dataset, k.fold.obj, foldnum){
  indices.train <- k.fold.obj != foldnum
  indices.test  <- k.fold.obj == foldnum
  data.train <- dataset[indices.train,]
  data.test  <- dataset[indices.test ,]
  return(list(train = data.train, test = data.test))
}

standardize_split_obj <- function(data_split_obj, d){
  data.train <- data_split_obj$train
  data.test  <- data_split_obj$test
  d <- ncol(data.train)-1
  preProcValues <- preProcess(data.train[1:d],  method = c("center", "scale")) #learn standarization parameters
  data.train.standarized <- predict(preProcValues, data.train) #standardize training set
  data.test.standarized <- predict(preProcValues, data.test) #standardize validation set
  return(list(
    train = data.train.standarized,
    test = data.test.standarized
  ))
}

get_weight_vec <- function(c_pos, c_neg, dataset){
  n_og <- sum(dataset$z == 0)
  if (n_og > 0){
    n <- nrow(dataset)
    n_neg <- sum(dataset$y == -1)
    pi_neg <- n_neg / n_og
    pi_pos <- 1 - pi_neg
    pi_neg_s <- n_neg / n
    pi_pos_s <- 1 - pi_neg_s
  
    L_neg <- c_pos * pi_pos_s * pi_neg
    L_pos <- c_neg * pi_neg_s * pi_pos
  
    n_pos_s <- sum((dataset$y == 1) & (dataset$z == 1))
    n_pos <- sum((dataset$y == 1) & (dataset$z == 0))
    OR <- (n_pos + n_pos_s) / n_pos
  
    L_pos_0 <- L_pos * OR / 2
    L_pos_1 <- L_pos * OR / 2 / (OR-1)
  
    weight_vec <- L_neg * (dataset$y == -1) + L_pos_0 * ((dataset$y == 1) & (dataset$z == 0)) + L_pos_1 * ((dataset$y == 1) & (dataset$z == 1))
    print(L_neg)
    print(L_pos_0)
    print(L_pos_1)
  }
  if (n_og == 0){
    L_neg <- c_pos
    L_pos <- c_neg
    weight_vec <- L_neg * (dataset$y == -1) +  L_pos * (dataset$y == 1)
    print(L_neg)
    print(L_pos)
  }
  
  return(weight_vec)
}
get.SMOTE.oversample <- function(data, OR){
  
  n_pos   <- sum(data$y == 1)
  n_pos_s <- n_pos * (OR-1)
  
 
  
  smote.samples = SMOTE(
    X = data[ -which(colnames(data) == "y") ],
    target = data["y"])$syn_data
  
  # 2.2.2. Then, we concatenate several SMOTE results.
  for (i in 1:ceiling(OR-1) ){  
    smote.samples_new =  SMOTE(
    X = data[ -which(colnames(data) == "y") ],
    target = data["y"])$syn_data
    smote.samples <- rbind(smote.samples, smote.samples_new)
  }   
  
  # 2.2.3. Finally, we choose as much as we want.
  idxChosen <- sample(1:dim(smote.samples)[1], n_pos_s, replace = FALSE)
  smote.samples.selected <- smote.samples[idxChosen , ]
  
  # 2.2.4. smote function changes the datatype and name of the target variable; So we fix them.
  smote.samples.selected["class"] <- factor(smote.samples.selected[["class"]], levels = c("-1", "1")); 
  colnames(smote.samples.selected) <- c( colnames(data) )
  
  smote.samples.selected$z <- 1
  data$z  <- 0
  
  data.augmented <- rbind(data,smote.samples.selected)
  data.augmented$y <- factor(data.augmented$y, levels = c("-1", "1")) # turn the y variable into a factor
  data.augmented$z <- factor(data.augmented$z, levels = c("0", "1")) # turn the z variable into a factor
  return(data.augmented)  
}

get.BLSMOTE.oversample <- function(data, OR){
  n_pos   <- sum(data$y == 1)
  n_pos_s <- n_pos * (OR-1)

  smote.samples = BLSMOTE(
    X = data[ -which(colnames(data) == "y") ],
    target = data["y"], #smote function only samples from the positive class
    dupSize = 0, K = 5, C = ceiling(n_pos / 4))$syn_data
  
  # 2.2.2. Then, we concatenate several SMOTE results.
  for (i in 1:ceiling(OR-1) ){  
    smote.samples_new = BLSMOTE(
    X = data[ -which(colnames(data) == "y") ],
    target = data["y"],
    dupSize = 0, K = 5, C = ceiling(n_pos / 4))$syn_data
    smote.samples <- rbind(smote.samples, smote.samples_new)
  }   
  
  # 2.2.3. Finally, we choose as much as we want.
  idxChosen <- sample(1:dim(smote.samples)[1], n_pos_s, replace = FALSE)
  smote.samples.selected <- smote.samples[idxChosen , ]
  
  # 2.2.4. smote function changes the datatype and name of the target variable; So we fix them.
  smote.samples.selected["class"] <- factor(smote.samples.selected[["class"]], levels = c("-1", "1")); 
  colnames(smote.samples.selected) <- c( colnames(data) )
  
  smote.samples.selected$z <- 1
  data$z  <- 0
  
  data.augmented <- rbind(data,smote.samples.selected)
  data.augmented$y <- factor(data.augmented$y, levels = c("-1", "1")) # turn the y variable into a factor
  data.augmented$z <- factor(data.augmented$z, levels = c("0", "1")) # turn the z variable into a factor
  return(data.augmented)  
}

get.DBSMOTE.oversample <- function(data, OR){
  
  n_pos   <- sum(data$y == 1)
  n_pos_s <- n_pos * (OR-1)

  
  smote.samples = DBSMOTE(
    X = data[ -which(colnames(data) == "y") ],
    target = data["y"],
    dupSize = 0)$syn_data

  # 2.2.2. Then, we concatenate several SMOTE results.
  for (i in 1:ceiling(OR-1) ){  
    smote.samples_new = DBSMOTE(
      X = data[ -which(colnames(data) == "y") ],
      target = data["y"],
      dupSize = 0)$syn_data
    smote.samples <- rbind(smote.samples, smote.samples_new)
  }   
  
  # 2.2.3. Finally, we choose as much as we want.
  idxChosen <- sample(1:dim(smote.samples)[1], n_pos_s, replace = FALSE)
  smote.samples.selected <- smote.samples[idxChosen , ]
  
  # 2.2.4. smote function changes the datatype and name of the target variable; So we fix them.
  smote.samples.selected["class"] <- factor(smote.samples.selected[["class"]], levels = c("-1", "1")); 
  colnames(smote.samples.selected) <- c( colnames(data) )
  
  smote.samples.selected$z <- 1
  data$z  <- 0
  
  data.augmented <- rbind(data,smote.samples.selected)
  data.augmented$y <- factor(data.augmented$y, levels = c("-1", "1")) # turn the y variable into a factor
  data.augmented$z <- factor(data.augmented$z, levels = c("0", "1")) # turn the z variable into a factor
  return(data.augmented)  
}













get.BLSMOTE.oversample <- function(data, OR){
  
  n_pos   <- sum(data$y == 1)
  n_pos_s <- n_pos * (OR-1)
  
  smote.samples = BLSMOTE(
    X = data[ -which(colnames(data) == "y") ],
    target = data["y"],
    dupSize = 0, K = 5, C = ceiling(n_pos / 4))$syn_data
  
  # 2.2.2. Then, we concatenate several SMOTE results.
  for (i in 1:ceiling(OR-1) ){  
    smote.samples_new = BLSMOTE(
      X = data[ -which(colnames(data) == "y") ],
      target = data["y"],
      dupSize = 0)$syn_data
    smote.samples <- rbind(smote.samples, smote.samples_new)
  }   
  
  # 2.2.3. Finally, we choose as much as we want.
  idxChosen <- sample(1:dim(smote.samples)[1], n_pos_s, replace = FALSE)
  smote.samples.selected <- smote.samples[idxChosen , ]
  
  # 2.2.4. smote function changes the datatype and name of the target variable; So we fix them.
  smote.samples.selected["class"] <- factor(smote.samples.selected[["class"]], levels = c("-1", "1")); 
  colnames(smote.samples.selected) <- c( colnames(data) )
  
  smote.samples.selected$z <- 1
  data$z  <- 0
  
  data.augmented <- rbind(data,smote.samples.selected)
  data.augmented$y <- factor(data.augmented$y, levels = c("-1", "1")) # turn the y variable into a factor
  data.augmented$z <- factor(data.augmented$z, levels = c("0", "1")) # turn the z variable into a factor
  return(data.augmented)  
}

get.DBSMOTE.oversample <- function(data, OR){
  
  n_pos   <- sum(data$y == 1)
  n_pos_s <- n_pos * (OR-1)

  
  smote.samples = DBSMOTE(
    X = data[ -which(colnames(data) == "y") ],
    target = data["y"],
    dupSize = 0)$syn_data

  # 2.2.2. Then, we concatenate several SMOTE results.
  for (i in 1:ceiling(OR-1) ){  
    smote.samples_new = DBSMOTE(
      X = data[ -which(colnames(data) == "y") ],
      target = data["y"],
      dupSize = 0)$syn_data
    smote.samples <- rbind(smote.samples, smote.samples_new)
  }   
  
  # 2.2.3. Finally, we choose as much as we want.
  idxChosen <- sample(1:dim(smote.samples)[1], n_pos_s, replace = FALSE)
  smote.samples.selected <- smote.samples[idxChosen , ]
  
  # 2.2.4. smote function changes the datatype and name of the target variable; So we fix them.
  smote.samples.selected["class"] <- factor(smote.samples.selected[["class"]], levels = c("-1", "1")); 
  colnames(smote.samples.selected) <- c( colnames(data) )
  
  smote.samples.selected$z <- 1
  data$z  <- 0
  
  data.augmented <- rbind(data,smote.samples.selected)
  data.augmented$y <- factor(data.augmented$y, levels = c("-1", "1")) # turn the y variable into a factor
  data.augmented$z <- factor(data.augmented$z, levels = c("0", "1")) # turn the z variable into a factor
  return(data.augmented)  
}












get.gmm.oversample <-function(gmm.model, data, OR){
  # generate synthetic samples from the learned GMM, and add them to the training dataset
  data$z  <- 0
  n_pos   <- sum(data$y == 1)
  n_pos_s <- n_pos * (OR-1)
  
  #1. extract parameters from the GMM result
  G <- gmm.model$G; # number of clusters, determined by BIC
  d <- gmm.model$d; # number of predictor variables. In our data, d=2
  prob <- gmm.model$parameters$pro # learned mixing coefficients pi_1, ... pi_G
  means <- gmm.model$parameters$mean # learned cluster mean vectors mu_1, ... mu_G
  vars <- gmm.model$parameters$variance$sigma #learned cluter covariance matrix Sigma_1, ... Sigma_G
  data.gmm <- data.frame(matrix(NA, n_pos_s, d + 2)) #initialize a matrix for storing synthetic minority  samples
  colnames(data.gmm ) <- colnames(data)
  
  #2. Generate synthetic minority samples from the learned Guassian mixture
  gmc.index <- sample(x = 1:G, size = n_pos_s, replace = T, prob = prob) #randomly assign group, according to the learned group membership probability.
  for(i in 1 : n_pos_s) {
    data.gmm [i, ] <- c(
      rmvnorm(1, mean = means[ , gmc.index[i]],sigma=vars[,,gmc.index[i]]),
      1,1
    )
  }
  data.augmented <- rbind(data,data.gmm)
  data.augmented$y <- factor(data.augmented$y, levels = c("-1", "1")) # turn the y variable into a factor
  data.augmented$z <- factor(data.augmented$z, levels = c("0", "1")) # turn the z variable into a factor
  return(data.augmented)
} # end of  function get.gmm.oversample


get_confusion_matrix <- function(svm_model, split_obj, d){
  svm.cmat <- table("truth" = split_obj$test$y, "pred" = predict(svm_model, split_obj$test[1:d]))
}
