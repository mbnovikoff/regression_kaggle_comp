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
# Make the outcome variable a factor in the data set 
# import data
train <- read_csv("data/train.csv") %>% 
  mutate_if(is.character, as.factor)
 test <- read_csv("data/test.csv")

train %>%
  select(-c(where(is.factor), where(is.character), id)) %>%
  cor(use = "pairwise") %>%
  corrplot::corrplot(method = "circle")

folds <- vfold_cv(train,  v = 5, repeats = 3)

bank_recipe <- recipe(money_made_inv ~ out_prncp_inv + loan_amnt + acc_now_delinq + num_tl_120dpd_2m + avg_cur_bal + tot_cur_bal + num_tl_30dpd + dti + int_rate + application_type
                      + tot_coll_amt + term + home_ownership + annual_inc + grade, data = train) %>%
  
  #term + out_prncp_inv + int_rate + application_type +
  #+  loan_amnt + tot_coll_amt 
  #+ annual_inc + avg_cur_bal + home_ownership + dti + grade, data = train) %>%
  #step_rm(id, purpose, earliest_cr_line, emp_title, last_credit_pull_d) %>% 
  step_dummy(all_nominal())

#we canuse prep and juice to verify that our recipe is working and transforming our results as we want it to  
prep(bank_recipe) %>% 
  bake(new_data = NULL)

rf_model <- rand_forest(mode = "regression",
                        min_n = tune(),
                        mtry = tune()) %>% 
  set_engine("ranger")

rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(range = c(1, 15)))

rf_grid <- grid_regular(rf_params, levels = 10)

rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(bank_recipe)

# Tuning/fitting ----
rf_tuned <- rf_workflow %>% 
  tune_grid(folds, grid = rf_grid)

rf_workflow_tuned <- rf_workflow %>% 
  finalize_workflow(select_best(rf_tuned, metric = "rmse"))

save(rf_tuned, rf_workflow, file = "model_info/rf_tuned.rda") 

rf_results <- fit(rf_workflow_tuned, train)

rf_pred <- predict(rf_results, test) %>% 
  bind_cols(Id = test$id) %>% 
  rename(Predicted = .pred)
rf_pred <- rf_pred[, c(2,1)]
write_csv(rf_pred, "rf_output.csv")
