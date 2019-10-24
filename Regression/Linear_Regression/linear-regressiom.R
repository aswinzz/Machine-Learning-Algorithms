LinearRegression <- function(formula, data, train_size, target){
  ## split dataset into train and test  
  set.seed(99)
  n <- nrow(data)
  id <- sample(1:n, size = train_size*n, replace = FALSE)
  train_df <- data[id, ] 
  test_df <- data[-id, ]
  
  ## train model
  train_form <- as.formula(formula)
  lm_model <- lm(train_form, data = train_df)
  
  ## predictions
  train_p <- predict(lm_model)
  test_p <- predict(lm_model, newdata = test_df)
  
  ## evaluate and print result
  train_mae <- mean(abs(train_p - train_df[[target]]))
  test_mae <- mean(abs(test_p - test_df[[target]]))
  cat("Train MAE:", train_mae)
  cat("\nTest MAE:", test_mae)
}

LinearRegression("medv ~ .", Boston, 0.80, "medv")
