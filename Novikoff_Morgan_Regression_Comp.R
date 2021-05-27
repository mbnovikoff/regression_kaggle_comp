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

# Visualize important continous variables 
train %>%
  select(-c(where(is.factor), where(is.character), id)) %>%
  cor(use = "pairwise") %>%
  corrplot::corrplot(method = "circle")

# Folds
folds <- vfold_cv(train,  v = 5, repeats = 3)

# Recipe
bank_recipe <- recipe(money_made_inv ~ out_prncp_inv + loan_amnt + acc_now_delinq 
                      + num_tl_120dpd_2m + avg_cur_bal + tot_cur_bal + num_tl_30dpd 
                      + dti + int_rate + application_type
                      + tot_coll_amt + term + home_ownership + annual_inc + grade, 
                       data = train) %>%
               step_dummy(all_nominal())

# Prep and Bake  
prep(bank_recipe) %>% 
  bake(new_data = NULL)

# Random Forest Model
rf_model <- rand_forest(mode = "regression",
                        min_n = tune(),
                        mtry = tune()) %>% 
            set_engine("ranger")

# Parameters
rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(range = c(1, 15)))

# Grid
rf_grid <- grid_regular(rf_params, levels = 10)

# Workflow
rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(bank_recipe)

# Tune and Fit
rf_tuned <- rf_workflow %>% 
  tune_grid(folds, grid = rf_grid)

rf_workflow_tuned <- rf_workflow %>% 
  finalize_workflow(select_best(rf_tuned, metric = "rmse"))

save(rf_tuned, rf_workflow, file = "model_info/rf_tuned.rda") 

# Results 
rf_results <- fit(rf_workflow_tuned, train)

# Final Output
rf_pred <- predict(rf_results, test) %>% 
  bind_cols(Id = test$id) %>% 
  rename(Predicted = .pred)
rf_pred <- rf_pred[, c(2,1)]
write_csv(rf_pred, "rf_output.csv")
