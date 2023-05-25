# Housing Price Prediction Project
Housing prices play a crucial role in finances, economic indicators, public policy decisions, and overall societal well-being. In this project, linear regression, random forest, and XGBoost models were trained to predict the av_total (assessed value) of properties in the greater Boston area.
## Load Library
```
library(tidymodels)
library(tidyverse)
library(janitor)
library(vip)
library(skimr)
```
## Import Data
```
boston <- read_csv("boston_train.csv") %>% clean_names()
kaggle <- read_csv("boston_holdout.csv") %>% clean_names()
zips <- read_csv("zips.csv") %>% clean_names()

skim(boston)
```
## Explore Data
```
b_analysis <- boston %>%
  mutate_if(is.character, as.factor) %>%
  mutate(built_decade = as.factor(yr_built - yr_built %% 10),
         remod_decade = as.factor(yr_remod - yr_remod %% 10))

mean_func <- function(col)
  b_analysis %>%
  group_by(!!as.name(col)) %>%
  summarise(mean_av_total = mean(av_total)) %>%
  arrange(desc(mean_av_total)) %>%
  print(b_analysis)

category_cols <- c("own_occ", "built_decade", "remod_decade")

for (col in category_cols) {
  mean_func(col)
}
```
## Data Partition
```
set.seed(123)
split <- initial_split(boston, prop = 0.7)
train <- training(split) 
test <- testing(split)

kfold_splits <- vfold_cv(train, v = 5)
```
## Create Recipe
```
boston_recipe <- recipe(av_total ~ land_sf + living_area + yr_built + num_floors + r_ovrall_cnd + r_int_cnd + r_fplace +
                        median_income + r_ext_cnd + r_ext_fin + r_bldg_styl + r_full_bth + r_half_bth + r_bth_style + 
                        r_kitch_style + population + pop_density, data = train) %>%
  step_mutate(home_age = 2022 - yr_built) %>% 
  step_rm(yr_built) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_nzv(all_predictors()) 

bake(boston_recipe %>% prep(), train %>% sample_n(1000))
```
## Build Models
### Linear Regression
```
lm_model <- linear_reg(mixture = 1, penalty = 0.001) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

lm_workflow <- workflow() %>%
  add_recipe(boston_recipe) %>%
  add_model(lm_model) %>%
  fit(train)

tidy(lm_workflow) %>%
  mutate_if(is.numeric, round, 4)

lm_workflow %>%
  pull_workflow_fit() %>%
  vi() %>% 
  mutate(Importance = if_else(Sign == "NEG", -Importance, Importance)) %>% 
  ggplot(aes(reorder(Variable, Importance), Importance, fill = Sign)) +
  geom_col() + coord_flip() + labs(title = "Linear Model Importance")

bind_cols(
  predict(lm_workflow, train, type = "numeric"), train) %>% 
  mutate(part = "train") -> score_lm_train

bind_cols(
  predict(lm_workflow, test), test) %>% mutate(part = "test") -> score_lm_test

bind_rows(score_lm_train, score_lm_test) %>% 
  group_by(part) %>% 
  metrics(av_total, .pred) %>%
  pivot_wider(id_cols = part, names_from = .metric, values_from = .estimate)
```
### Random Forest
```
rf_model <- rand_forest(trees = tune(), min_n = tune()) %>%
  set_mode("regression") %>%
  set_engine("ranger", importance = "permutation")

rf_workflow <- workflow() %>%
  add_recipe(boston_recipe) %>%
  add_model(rf_model)

rf_grid <- grid_regular(trees(c(25, 250)), min_n(c(5, 10)), levels = 4)

rf_grid_search <- tune_grid(rf_workflow, resamples = kfold_splits, grid = rf_grid)
rf_grid_search %>% collect_metrics()

rf_grid_search %>%
  collect_metrics() %>%
  ggplot(aes(min_n, mean, color = .metric)) +
  geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err),
                alpha = 0.5) +
  geom_line(size = 1.5) +
  facet_wrap(~.metric, scales = "free", nrow = 2) +
  scale_x_log10() +
  theme(legend.position = "none") +
  labs(title = "Random Forest min_n")

rf_grid_search %>%
  collect_metrics() %>%
  ggplot(aes(trees, mean, color = .metric)) +
  geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err),
                alpha = 0.5) +
  geom_line(size = 1.5) +
  facet_wrap(~.metric, scales = "free", nrow = 2) +
  scale_x_log10() +
  theme(legend.position = "none") +
  labs(title = "Random Forest Number of Trees")

* Final Fit *
lowest_rf_rmse <- rf_grid_search %>%
  select_best("rmse")

rf_final <- finalize_workflow(rf_workflow, lowest_rf_rmse) %>%
  fit(train)

* Evaluation *
rf_final %>%
  pull_workflow_fit() %>%
  vip() + labs(title = "Random Forest Importance")

bind_cols(predict(rf_final, test), test) %>%
  mutate(error = av_total - .pred) %>% 
  slice_min(order_by = error, n = 10)

bind_cols(
  predict(rf_final, train), train) %>% 
  metrics(av_total, .pred)

bind_cols(
  predict(rf_final, test), test) %>% 
  metrics(av_total, .pred)
```
