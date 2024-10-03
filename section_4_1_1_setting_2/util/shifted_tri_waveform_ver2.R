#### PART 1. the main data-generating function
get_waveform_data_mytri <- function(IR, positive_sample_size, var_pos, var_neg){
  data <- rbind(
    get_waveform_data_neg_mytri(floor(positive_sample_size * IR), var_neg),
    get_waveform_data_pos_mytri(positive_sample_size, var_pos)
  )
  data$y <- as.factor(data$y)
  return(data)
}

#### PART 2. negative class and positive class data generators,
####         used by get_waveform_data_mytri
get_waveform_data_pos_mytri <- function(sample_size, var){
  data <- get_mixture(sample_size, c(-8,-5,-2, 1,4,7,10), var)
                                    #these values are specified in the paper
  data$y <- 1
  return(data)
}

get_waveform_data_neg_mytri <- function(sample_size, var){
  data <- get_mixture_neg(sample_size, c(-10,-7,-4, -1, 2,5,8), var)
                                        #these values are specified in the paper
  data$y <- -1
  return(data)
}

#### PART 3. inner functions of get_waveform_data_pos_mytri and get_waveform_data_neg_mytri
get_mixture <- function(sample_size, shift_vec, var){
  n_mixture <- length(shift_vec)
  mixture_indicator_vec <- sample(length(shift_vec), sample_size, replace = TRUE)
  mixture_indicator_mat <- duplicate_col(mixture_indicator_vec, 21) 
  
  data_array <- array(dim= c(sample_size, 21, n_mixture))
  for (i in 1:n_mixture){
    data_array[,,i] <- duplicate_row(my_triangle_shift(1:21, shift_vec[i]), sample_size)
  }
  
  gaussian_noise <- matrix(sqrt(var)*rnorm(sample_size * 21), nrow = sample_size)
  data_mat <- matrix(0, nrow = sample_size, ncol = 21)
  for (i in 1:n_mixture){
    data_mat <- data_mat + (mixture_indicator_mat==i) * data_array[,,i]
  }
  data_mat <- data_mat + gaussian_noise
  
  data_df = data.frame(cbind(data_mat, rep(1, sample_size)))
  colnames(data_df)[22] <- "y"
  return(data_df)
}

get_mixture_neg <- function(sample_size, shift_vec, var){
  # does not create a mixture; the mean vector is the mixture of seven vectors
  n_mixture <- length(shift_vec)
  data_array <- array(dim= c(sample_size, 21, n_mixture))
  for (i in 1:n_mixture){
    data_array[,,i] <- duplicate_row(my_triangle_shift(1:21, shift_vec[i]), sample_size)
  }
  
  gaussian_noise <- matrix(sqrt(var)*rnorm(sample_size * 21), nrow = sample_size)
  data_mat <- matrix(0, nrow = sample_size, ncol = 21)
  for (i in 1:n_mixture){
    data_mat <- data_mat + (1/length(shift_vec)) * data_array[,,i] #elementwise multiplication
  }
  data_mat <- data_mat + gaussian_noise
  
  data_df = data.frame(cbind(data_mat, rep(1, sample_size)))
  colnames(data_df)[22] <- "y"
  return(data_df)
}

#### PART 4. Basic  operations
duplicate_row <- function(vec, sample_size){
  mat = matrix(rep(vec, sample_size), nrow = sample_size, byrow = TRUE)
  return(mat)
}

duplicate_col <- function(vec, dimension){
  mat = matrix(rep(vec, dimension), ncol = dimension)
  return(mat)
}

my_triangle <- function(i){
  dim = length(i)
  raw_wave = (6-3*abs(i-11))/10
  waveform = pmax(rep(0,dim),raw_wave)
  return(waveform)
}

my_triangle_shift <- function(i, shift){
  return(my_triangle(i-shift))
}

get_one_subgroup <-function(sample_size, shift_1, shift_2, var){
  set.seed(abs(shift_1*shift_2)+shift_1)
  convex_coef_vec <- runif(sample_size)
  convex_coef_mat <- duplicate_col(convex_coef_vec, 21)

  h_1_mat <- duplicate_row(my_triangle_shift(1:21, shift_1), sample_size)
  h_2_mat <- duplicate_row(my_triangle_shift(1:21, shift_2), sample_size)
 
  gaussian_noise <- sqrt(var)*rnorm(sample_size * 21)
  gaussian_noise <- matrix(gaussian_noise, nrow = sample_size)
  sample_mat <- convex_coef_mat * h_1_mat + (1-convex_coef_mat) * h_2_mat + gaussian_noise
  return(sample_mat)
}