rcode_dir <- getwd( ) # Please change this directory to the appropriate one.
utildir = paste0(rcode_dir, '/util')

# simulation setting parameters, same as the description in Section 4.1.1
# change n_pos value to try different settings
OR = 5
var_neg = 1/25
var_pos = 1/25
n_pos = 60
IR = (3000-n_pos)/n_pos 
method <- "clustersvm"
d = 21
n_fold <- 5

# repetition number is used as the seed number
rep = 1

library(ROCit)
library(caret)
library(e1071)
library(mclust)
library(SwarmSVM) # for clusterSVM
source( paste0(utildir, "/shifted_tri_waveform.R") ) #setting 1 of section 4.1.1
source( paste0(utildir, "/util_gswsvm.R") )


#packages required for sqlite3. comment them if unnecessary.
#library(DBI)
#library(RSQLite)
#table_name = 'mixture_neg'



#####################################
# Preliminary step
#####################################


#tuning parameters range
param.set.c <- c(1, 5, 10, 20, 50, 100)
param.set.gamma = c(1, 5, 10, 20, 50, 100)

# arrays for saving tuning result
tune.mat.gcwsvm <- array(0, dim = c(
  length(param.set.c),
  length(param.set.gamma)))

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
dataset <- get_waveform_data_mytri(
  IR = IR, positive_sample_size = n_pos, var_pos = var_pos, var_neg = var_neg
)
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
    gmc.model <- Mclust(tunetrain_tunevalid_split_now$train[1:d], modelNames = c("EII")) #learn GMM
    param.clusterSVM.k <- gmc.model$G
      for (i in 1:length(param.set.c)){ #hyperparameter tuning: loop over C
        for (j in 1:length(param.set.gamma)){ #hyperparameter tuning: loop over gamma
        # fit weighted svm model
        model.gcwsvm.now <- clusterSVM(
          x = tunetrain_tunevalid_split_now$train[1:d],
          y = (tunetrain_tunevalid_split_now$train)$y,
          lambda = param.set.gamma[j],
          cost = param.set.c[i],
          centers = param.clusterSVM.k,
          seed = 512, verbose = 0) 

        # calculate g-mean
        predict_clustersvm_tune <- predict(model.gcwsvm.now, tunetrain_tunevalid_split_now$test[1:d])$predictions
        svm.cmat <- table(
          "truth" = tunetrain_tunevalid_split_now$test$y,
          "pred" = predict_clustersvm_tune
          )


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
      param.set.gamma[param.best.gcwsvm[2]],
      "\n"
  )

  gmc.model.best <- Mclust(train_test_split_now$train[1:d], modelNames = c("EII")) #learn GMM
  model.gcwsvm.best <- clusterSVM(
          x = train_test_split_now$train[1:d],
          y = (train_test_split_now$train)$y,
          lambda = param.set.gamma[param.best.gcwsvm[2]],
          cost = param.set.c[param.best.gcwsvm[1]],
          centers = gmc.model.best$G,
          seed = 512, verbose = 0) 
  predict_clustersvm_test <- predict(model.gcwsvm.best, train_test_split_now$test[1:d], decisionValues = TRUE)
  cmat.gcwsvm <- table(
          "truth" = train_test_split_now$test$y,
          "pred" = predict_clustersvm_test$predictions
          )
  print(cmat.gcwsvm)
  result.mat$ACC <- result.mat$ACC  + get_accuracy(cmat.gcwsvm)/5
  result.mat$PRE <- result.mat$PRE  + get_precision(cmat.gcwsvm)/5
  result.mat$TPR <- result.mat$TPR  + get_TPR(cmat.gcwsvm)/5
  result.mat$FME <- result.mat$FME  + get_fmeasure(cmat.gcwsvm)/5
  result.mat$GME <- result.mat$GME  + get_gmean(cmat.gcwsvm)/5
  result.mat$TNR <- result.mat$TNR  + get_TNR(cmat.gcwsvm)/5
  
  pred <- predict(model.gcwsvm.best, train_test_split_now$test[1:d])
  decision.values <- predict_clustersvm_test$decisionValues[,1]
  roc_empirical <- rocit(score = decision.values, class = train_test_split_now$test$y,
                         negref = "1") 
  result.mat$AUC <- result.mat$AUC + (roc_empirical$AUC)/5
} # end of for loop level 2 (5-fold CV for performance metric evaluation)



print(result.mat)
conn <- dbConnect(RSQLite::SQLite(), "/home1/jongminm/forest_fire_ROK_army/experiment/gswsvm.db")
query <- paste0('INSERT INTO ', table_name, '(rep, accuracy, precision, tpr, tnr, fmeasure, gmean, auc, method, over_ratio, imbal_ratio, dimension, n_pos)', ' VALUES (?, ?, ?, ?, ?, ?, ?,?, ?, ?, ?, ?, ?)')

dbExecute(conn, query, params = c(rep, result.mat$ACC, result.mat$PRE, result.mat$TPR, result.mat$TNR, result.mat$FME, result.mat$GME, result.mat$AUC, method, OR, IR, d, n_pos) )
dbDisconnect(conn)

