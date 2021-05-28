# Load package(s)
library(tidymodels) 
library(tidyverse) 
library(skimr) 
library(corrplot) 
library(lubridate) 

# Handle common conflicts
tidymodels_conflicts()

# Seed
set.seed(123) 

# Load data ----

# Convert character into factors 
train <- read_csv("data/train.csv") %>% 
  mutate_if(is.character, as.factor)
test <- read_csv("data/test.csv") 



# Visualize important continuous variables 
train %>%
  select(-c(where(is.factor), where(is.character), id)) %>%
  cor(use = "pairwise") %>%
  corrplot::corrplot(method = "circle")

# Folds
folds <- vfold_cv(train, v = 5, repeats = 3, strata = money_made_inv) 

# Recipe 
bank_recipe2 <- recipe(money_made_inv ~ bc_util + int_rate + loan_amnt + 
                          out_prncp_inv + tot_cur_bal, data = train) %>% 
  step_other(all_nominal(), -all_outcomes(), threshold = 0.1) %>% 
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
  step_normalize(all_numeric(), -all_outcomes()) %>% 
  step_zv(all_predictors(), -all_outcomes()) 

# Prep and Bake
prep(bank_recipe2) %>% 
  bake(new_data = NULL) 


# Random Forest model 
rf_model2 <- rand_forest(mode = "regression",
                        min_n = tune(), 
                        mtry = tune()) %>% 
  set_engine("ranger")

# Parameters
rf_params2 <- parameters(rf_model2) %>% 
  update(mtry = mtry(range = c(2, 5)))


# Grid
rf_grid2 <- grid_regular(rf_params2, levels = 10)

# Workflow
rf_workflow2 <- workflow() %>% 
  add_model(rf_model2) %>% 
  add_recipe(bank_recipe2) 

# Tune and Fit
rf_tune2 <-  rf_workflow2 %>% 
  tune_grid(resamples = folds, 
            grid = rf_grid2) 

save(rf_tune2, rf_workflow2, file = "model_info/rf_tuned2.rda") 


rf_workflow_tuned2 <- rf_workflow2 %>% 
  finalize_workflow(select_best(rf_tune2, metric = "rmse")) 

# Results
rf_results2 <- fit(rf_workflow_tuned2, train) 

# Prediction
rf_results2 <- rf_results2 %>% 
  predict(new_data = test) %>% 
  bind_cols(Id = test$id) %>% 
  rename(Predicted = .pred) 

rf_results2 <- rf_results2[, c(2,1)] 

write_csv(rf_results2, file = "rf_output2.csv") 