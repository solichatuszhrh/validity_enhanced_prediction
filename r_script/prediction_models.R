### THIS SCRIPT CONTAINS CODES TO GET THE PREDICTION MODELS FROM THREE DIFFERENT MODELS, WHICH ARE NEURAL NETWORK, RANDOM FOREST, AND EXTREME GRADIENT BOOSTING.
### THE PREDICTION VALUES ARE BASED ON THREE SCHWARTZ HUMAN VALUES (CHOSEN ARBITRARILY): UNIVERSALISM, TRADITION, AND POWER.
### THE FULL DATASET CONTAINS 2029 RESPONDENTS BUT AFTER EXCLUDING NA VALUES (THE RESPONDENTS WHO WERE NOT PARTICIPATED IN THE SECOND PART: 163 RESPONDENTS), 1866 RESPONDENTS ARE LEFT.
### ONLY "POWER" THAT HAS COMPLETE VALUES BECAUSE THE FIRST PART IS FROM QUESTION 1 TO QUESTION 30 AND THE SECOND PART IS FROM QUESTION 31 TO QUESTION 57.
### TRAIN-TEST SPLIT WAS APPLIED TO 1866 RESPONDENTS WITH 80:20 RATIO.


### INPUT
### INPUT DATA IS THREE HUMAN VALUES (UNIVERSALISM, TRADITION, AND POWER)
### THE HUMAN VALUES ARE THE SUM OF SEVERAL QUESTIONS (VALUES OF 57 QUESTIONS REFER TO SVS 57, VERSION OF THE TEST INTRODUCED IN THE 1990S)
### 57 QUESTIONS OF SURVEY DATA REPRESENTING 10 SCHWARTZ HUMAN VALUES
### 1. POWER (q3, q12, q27)
### 2. ACHIEVEMENT (q34, q43, q55, q39)
### 3. UNIVERSALISM (q1, q17, q24, q26, q29, q30, q35, q38)
### 4. HEDONISM (q4, q50, q57)
### 5. STIMULATION (q9, q25, q37)
### 6. SECURITY (q13, q15, q56, q8, q22)
### 7. BENEVOLENCE (q33, q45, q49, q52, q54)
### 8. TRADITION (q36, q44, q51, q32)
### 9. CONFORMITY (q11, q20, q40, q47)
### 10. SELF-DIRECTION (q5, q16, q31, q41, q53)

### OUTPUT
### OUTPUT DATA IS PREDICTION VALUES FOR THOSE THREE HUMAN VALUES

### ANALYSIS
### THE FACEBOOK METHOD DATA IS THE PREDICTION VALUES FROM NEURAL NETWORK, RANDOM FOREST, AND/OR EXTREME GRADIENT BOOSTING
### THE PREDICITON VARIABLES ARE 600 LDA (LATENT DIRICHLET ALLOCATION)



### Dataset
# Convert data to matrix format
data <- combine[,V2:V601]  # Features matrix (600 LDA values)
labels <- combine[,c("universalism","tradition","power")] # Target vector (3 human values)

# Split data into training and testing sets
set.seed(42)
train_index <- sample(1:nrow(combine), size = 0.8 * nrow(combine))
train_data <- as.matrix(data[train_index, ])
train_labels <- as.matrix(labels[train_index])
test_data <- as.matrix(data[-train_index, ])
test_labels <- as.matrix(labels[-train_index])

### Calculate the MSE
mse <- function(true, predicted) {
  mean((true - predicted)^2)
}


### Neural Networks
library(neuralnet)

nn_results <- list()  # To store models and predictions

for (i in seq_len(ncol(train_labels))) {
  # Extract each outcome variable
  target_train <- train_labels[, i]
  target_test <- test_labels[, i]
  
  # Combine features and target into a single data frame
  nn_train_data <- data.frame(train_data, target = target_train)
  
  # Define the formula for neural network
  nn_formula <- as.formula(paste("target ~", paste(colnames(train_data), collapse = "+")))
  
  # Train the neural network
  nn_model <- neuralnet(nn_formula, data = nn_train_data, hidden = c(10, 5), linear.output = TRUE)
  
  # Make predictions
  nn_predictions <- compute(nn_model, test_data)$net.result
  
  # Store the MSE and predictions
  nn_results[[i]] <- list(
    model = nn_model,
    predictions = nn_predictions,
    mse = mse(target_test, nn_predictions)
  )
}


### Random Forest
library(randomForest)

rf_results <- list()  # To store models and predictions

for (i in seq_len(ncol(train_labels))) {
  # Extract each outcome variable
  target_train <- train_labels[, i]
  target_test <- test_labels[, i]
  
  # Train the random forest model
  rf_model <- randomForest(x = train_data, y = target_train, ntree = 500, mtry = 3, importance = TRUE)
  
  # Make predictions
  rf_predictions <- predict(rf_model, test_data)
  
  # Store the MSE and predictions
  rf_results[[i]] <- list(
    model = rf_model,
    predictions = rf_predictions,
    mse = mse(target_test, rf_predictions)
  )
}


### Extreme Gradient Boosting
library(xgboost)

xgb_results <- list()  # To store models and predictions

for (i in seq_len(ncol(train_labels))) {
  # Extract each outcome variable
  target_train <- as.numeric(train_labels[, i])
  target_test <- as.numeric(test_labels[, i])
  
  # Convert data to DMatrix
  dtrain <- xgb.DMatrix(data = train_data, label = target_train)
  dtest <- xgb.DMatrix(data = test_data, label = target_test)
  
  # Set model parameters
  params <- list(objective = "reg:squarederror", max_depth = 6, eta = 0.1, subsample = 0.8, colsample_bytree = 0.8)
  num_rounds <- 100
  
  # Train the model
  xgb_model <- xgb.train(params = params, data = dtrain, nrounds = num_rounds)
  
  # Make predictions
  xgb_predictions <- predict(xgb_model, dtest)
  
  # Store the MSE and predictions
  xgb_results[[i]] <- list(
    model = xgb_model,
    predictions = xgb_predictions,
    mse = mse(target_test, xgb_predictions)
  )
}


### Compare results
for (i in seq_len(ncol(train_labels))) {
  cat("\nOutcome Variable", i, "\n")
  cat("Neural Network MSE:", round(nn_results[[i]]$mse, 2), "\n")
  cat("Random Forest MSE:", round(rf_results[[i]]$mse, 2), "\n")
  cat("XGBoost MSE:", round(xgb_results[[i]]$mse, 2), "\n")
  cat("Baseline MSE:", mean(test_labels[i]),"\n")
}


### Combine dataset
full_data <- cbind(test_data,test_labels)
v_names <- paste0("V", 1:600)
colnames(full_data) <- c(v_names, "universalism", "tradition", "power")
full_data <- as.data.frame(full_data)

# Add prediciton values
full_data$uni_xgb <- as.numeric(xgb_results[[1]]$predictions)
full_data$tra_xgb <- as.numeric(xgb_results[[2]]$predictions)
full_data$pow_xgb <- as.numeric(xgb_results[[3]]$predictions)
full_data$uni_nn <- as.numeric(nn_results[[1]]$predictions)
full_data$tra_nn <- as.numeric(nn_results[[2]]$predictions)
full_data$pow_nn <- as.numeric(nn_results[[3]]$predictions)
full_data$uni_rf <- as.numeric(rf_results[[1]]$predictions)
full_data$tra_rf <- as.numeric(rf_results[[2]]$predictions)
full_data$pow_rf <- as.numeric(rf_results[[3]]$predictions)
