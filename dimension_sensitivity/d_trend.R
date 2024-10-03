

library("mclust")

library("mvtnorm")

library("LOMAR")
library("dad")


sigma_square = 1/25
n_mixture_true = 4
get_magnitude <- function(d){
  m = 1/2/sqrt(d)
}


generate_gaussian_mixture <- function(n, n_mixture_true, mu_1, mu_2, mu_3, mu_4, cov){
  n_vec = rmultinom(1,n , rep(1, n_mixture_true) )
  data_1 = rmvnorm(n = n_vec[1], mean = mu_1, sigma = cov)
  data_2 = rmvnorm(n = n_vec[2], mean = mu_2, sigma = cov)
  data_3 = rmvnorm(n = n_vec[3], mean = mu_3, sigma = cov)
  data_4 = rmvnorm(n = n_vec[4], mean = mu_4, sigma = cov)
  data = rbind(data_1, data_2, data_3, data_4)
  return(data)
}

get_mean_vec <- function(d){
  m <-  get_magnitude(d)
  mu_0 = m*rep(1,d)
  nu_0 = m*c(rep(1,d/2), rep(-1, d/2))
  mu_1 = mu_0
  mu_2 = -mu_1
  mu_3 = nu_0
  mu_4 = -mu_3
  return( t(cbind(mu_1, mu_2, mu_3, mu_4)))
}

generate_data_gmm <-function(d, n, sigma_square){
	mean_vec = get_mean_vec(d)
	cov = sigma_square*diag(d)
	data_min <- generate_gaussian_mixture(n, n_mixture_true, mean_vec[1,], mean_vec[2,], mean_vec[3,], mean_vec[4,], cov)
	return(data_min)
}



#####################################
# Preliminary step
#####################################
replication <- 200



###################################
# Main part
###################################

################################################
d_vec = 2*seq(1,16)
n_vec = c(50,100,200,300,400, 500)
result_K = matrix(nrow = length(n_vec), ncol = length(d_vec))
result_distance = matrix(nrow = length(n_vec), ncol = length(d_vec))
result_mean = matrix(nrow = length(n_vec), ncol = length(d_vec))


for (idx_d in 1:length(d_vec)){
	d = d_vec[idx_d]
	for (idx_n in 1:length(n_vec)){
		n = n_vec[idx_n]
		result_mat = matrix(nrow = replication, ncol = 2)
		colnames(result_mat) = c("K", "distance")
		result_mat = data.frame(result_mat)
		for (rep in 1:replication){
  			cat(rep, "th rep", "\n")
  			set.seed(rep*10) # for reproducible result
  			dataset <- generate_data_gmm(d,n, sigma_square)
  			gmc.model <- Mclust(dataset, modelNames = c("EII")) #learn GMM
  			n_mixture_est <- gmc.model$G
  			var_est <- gmc.model$parameters$variance$scale
    		result_mat$K[rep] <- n_mixture_est
    		
			covariance_mat <- array(sigma_square * diag(d), c(d, d, n_mixture_true) )
			covariance_mat_est <- array(var_est * diag(d), c(d, d, n_mixture_est) )
			mean_mat_original <- get_mean_vec(d)
			mean_mat_estimated <- t(matrix(gmc.model$parameters$mean, ncol = n_mixture_est))
			result_mat$distance[rep] = GMM_Wd(
			  mean_mat_original,
			  mean_mat_estimated,
			  covariance_mat,
			  covariance_mat_est)$d
		}# for loop over replication
	result_summary = apply(result_mat, 2, mean)
	result_K[idx_n, idx_d] = mean(result_mat$K)
	result_distance[idx_n, idx_d] = mean(result_mat$distance)
	}
}

colnames(result_distance) <- d_vec
rownames(result_distance) <- n_vec
result_distance

