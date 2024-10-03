rcode_dir <- getwd( ) # Please change this directory to the appropriate one.
utildir = paste0(rcode_dir, '/util')


OR = 4.72 #change this value
rep = 1 #used as the seed number
datafile = paste0(rcode_dir, '/army_fire.csv')
data = 'army'
n_pos = 40

library(ROCit)
library(caret)
library(e1071)
source( paste0(utildir, "/util_gswsvm.R") )
method <- "smote"
d = 4
n_fold <- 5

#db insert
#library(DBI)
#library(RSQLite)
#table_name = 'comparison_real'


#####################################
# Preliminary step
#####################################


#tuning parameters range
param.set.c <- 4 ^ (-5:5)
param.set.gamma = 4 ^ (-5:5)
# arrays for saving tuning result
tune.mat.gcwsvm <- array(0, dim = c(
  length(param.set.c),
  length(param.set.gamma)
))

result.mat <- matrix(nrow = 1, ncol = 7)
colnames(result.mat) <- c("ACC", "TPR", "TNR", "GME", "AUC", "PRE", "FME")
result.mat <- data.frame(result.mat)
result.mat[1,] <- c(0,0,0,0,0,0,0)

# read data


###################################
# Main part
###################################

################################################
# For loop level 1: monte carlo simulation. Since data is fixed, this means trying independent splits


################################################
cat(rep, "th rep", "\n")
set.seed(rep) # for reproducible result
dataset <- read.csv(datafile)
dataset$y <- factor(dataset$y, levels = c("-1", "1") );
################################################
# For loop level 2: 5-fold CV for performance metric evaluation
k.fold.test <- createFolds(dataset$y, k = n_fold, list = FALSE, returnTrain = FALSE)
for (foldnum.test in 1:n_fold) {
  ################################################
  cat( rep, "th rep, ", foldnum.test, "th test fold\n ", sep = "")
  set.seed(foldnum.test) # for reproducible result
  train_test_split_now <- data_onefold_split(dataset, k.fold.test, foldnum.test) # training-test set split
  ################################################
  # for loop level 3: 5-fold CV for hyperparameter tuning
  k.fold.tune <- createFolds(train_test_split_now$train$y, k = n_fold, list = FALSE, returnTrain = FALSE)
  for (foldnum.tune in 1:n_fold) {
    ################################################
    cat( rep, "th rep, ", foldnum.test, "th test fold, ", foldnum.tune, "th tune fold\n", sep = "")
    tunetrain_tunevalid_split_now <- data_onefold_split(train_test_split_now$train, k.fold.tune, foldnum.tune) # [tuning] training-validation set split
    tunetrain_tunevalid_split_now <- standardize_split_obj(tunetrain_tunevalid_split_now, d) # [tuning] standardization
    ################################################
      tunetrain_tunevalid_split_now_train_dbsmote <- get.SMOTE.oversample(tunetrain_tunevalid_split_now$train, OR)
      for (i in 1:length(param.set.c)){ #hyperparameter tuning: loop over C
        for (j in 1:length(param.set.gamma)){ #hyperparameter tuning: loop over gamma
        # fit weighted svm model
        
        model.gcwsvm.now <- e1071::svm(data = tunetrain_tunevalid_split_now_train_dbsmote[1:(d+1)], y ~ .,
                                gamma = param.set.gamma[j],
                                cost = param.set.c[i],
                                kernel = "radial", scale = FALSE)
        # calculate g-mean
        svm.cmat <- get_confusion_matrix(model.gcwsvm.now, tunetrain_tunevalid_split_now, d)
        gme <- get_gmean(svm.cmat) #G-mean
        tune.mat.gcwsvm[i,j] <- tune.mat.gcwsvm[i,j] + (gme / 5) # mean G-mean over 5 folds
        } #end of gamma loop
      }# end of c loop
    
  }# end of tuning
  print("tuning done")
  ############################################################################################################
  # Now we fit the models with best hyperparameters and evaluate performance of the models.
  ############################################################################################################
  
  # standardization
  train_test_split_now <- standardize_split_obj(train_test_split_now, d) # [tuning] standardization
  
  # Fit best GC-WSVM
  param.best.gcwsvm <- which(tune.mat.gcwsvm == max(tune.mat.gcwsvm), arr.ind = TRUE)
  
  cat("best parameter for gcwsvm: c = ",  
      param.set.c[param.best.gcwsvm[1]],
      ", gamma = ",
      param.set.gamma[param.best.gcwsvm[2]]
  )
  train_test_split_now$train <- get.SMOTE.oversample( train_test_split_now$train,  OR)
  model.gcwsvm.best <- svm(data = train_test_split_now$train[1:(d+1)], y ~ .,
                           gamma = param.set.gamma[param.best.gcwsvm[2]],
                           cost = param.set.c[param.best.gcwsvm[1]],
                           kernel = "radial",
                           scale = FALSE
  )
  
  cmat.gcwsvm <- get_confusion_matrix(model.gcwsvm.best, train_test_split_now, d)
  print(cmat.gcwsvm)
  result.mat$ACC <- result.mat$ACC  + get_accuracy(cmat.gcwsvm)/5
  result.mat$PRE <- result.mat$PRE  + get_precision(cmat.gcwsvm)/5
  result.mat$TPR <- result.mat$TPR  + get_TPR(cmat.gcwsvm)/5
  result.mat$FME <- result.mat$FME  + get_fmeasure(cmat.gcwsvm)/5
  result.mat$GME <- result.mat$GME  + get_gmean(cmat.gcwsvm)/5
  result.mat$TNR <- result.mat$TNR  + get_TNR(cmat.gcwsvm)/5
  
  pred <- predict(model.gcwsvm.best, train_test_split_now$test[1:d], decision.values = TRUE)
  decision.values <- attr(pred, "decision.values")[,]
  roc_empirical <- rocit(score = decision.values, class = train_test_split_now$test$y,
                         negref = "1") 
  result.mat$AUC <- result.mat$AUC + (roc_empirical$AUC)/5
} # end of for loop level 2 (5-fold CV for performance metric evaluation)


# db insert
#print(result.mat)
#conn <- dbConnect(RSQLite::SQLite(), "/home1/jongminm/forest_fire_ROK_army/experiment/gswsvm.db")
#query <- paste0('INSERT INTO ', table_name, '(rep, accuracy, precision, tpr, tnr, fmeasure, gmean, auc, method, over_ratio, imbal_ratio, dimension, n_pos)', ' VALUES (?, ?, ?, ?, ?, ?, ?,?, ?, ?, ?, ?, ?)')
#dbExecute(conn, query, params = c(rep, result.mat$ACC, result.mat$PRE, result.mat$TPR, result.mat$TNR, result.mat$FME, result.mat$GME, result.mat$AUC, method, OR, IR, d, n_pos) )
#dbDisconnect(conn)

