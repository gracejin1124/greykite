# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC #### Overview
# MAGIC 
# MAGIC We will be generating sales forecasts. The following are the metadata parameters:
# MAGIC - Frequency of series: Weekly
# MAGIC - Horizon: 12 weeks
# MAGIC - Granularity: Undecided
# MAGIC - Frequency of generation: Undecided
# MAGIC 
# MAGIC This will be done in three phases:
# MAGIC 1. Model Selection
# MAGIC   - We specify a broad set of models/hyperparameters called the set of candidate models. For each series, the best model, based off some undecided metric, will be chosen via tscv. The model selection will be run for all series once a year, or, for individual series, if the forecasting accuracy decreases significantly.
# MAGIC 2. Training
# MAGIC   - For each series, once a model has been chosen, the model must be retrained on the entire series so that it can be used to generate forecasts. This will be done right before forecasting as training GLMs is relatively cheap.
# MAGIC 3. Forecasting
# MAGIC   - 12 weeks of forecasts for each series will be generated ever "Frequency of generation" weeks

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Imports

# COMMAND ----------

#https://docs.microsoft.com/en-us/azure/databricks/kb/notebooks/cmd-c-on-object-id-p0
import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

import pandas as pd
import numpy as np
import plotly

from greykite.common import constants as cst
from greykite.framework.templates.autogen.forecast_config import (ComputationParam,
                                                                        EvaluationMetricParam,
                                                                        EvaluationPeriodParam,
                                                                        MetadataParam, ForecastConfig,
                                                                        ModelComponentsParam)
from greykite.sklearn.cross_validation import RollingTimeSeriesSplit
from greykite.framework.benchmark.benchmark_class import BenchmarkForecastConfig
from greykite.common.evaluation import EvaluationMetricEnum

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Query

# COMMAND ----------

jdbcHostname = "tdprod1cop1.ngco.com"
jdbcDatabase = "RLDMPROD_V"
jdbcPort = 1025
username = 'aspuri'
password = 'welcome1'
encryptdata = "ON"
driver = "com.teradata.jdbc.TeraDriver"
url = 'jdbc:teradata://{}/DBS_PORT={},DATABASE={},USER={},PASSWORD={},ENCRYPTDATA={}'.format(jdbcHostname, jdbcPort, jdbcDatabase, username, password, encryptdata)

# COMMAND ----------

query = f"""(
SELECT
CAL_WK_END_DT week_end_dt
, CAST(SUM(x0.PRRTD_PSTD_SL_AMT) AS float) sales
, CAST((x1.CORP_YR_NUM + (x1.CORP_PD_NUM/13.0)) AS float) year_period
, CAST((x1.CORP_YR_NUM + (x1.CORP_QTR_NUM/4.0)) AS float) year_quarter
FROM RLDMPROD_V.RTL_QLFY_PRRTD_PSTD_SL_DLY x0
LEFT JOIN RLDMPROD_V.CAL_DT x1
ON x0.TRANS_DT = x1.CAL_DT
LEFT JOIN CNSLD_ARTCL x2
ON TO_NUMBER(x0.SCAN_CD) = x2.SCAN_CD
LEFT JOIN CNSLD_MCH_HIER x3
ON x2.LCL_MCH_0_CD = x3.MCH_0_CD
LEFT JOIN RLDMPROD_V.SITE_HIER x4
ON x0.SITE_NUM = x4.SITE_NUM
--only keep currently open sites
WHERE x0.SITE_NUM in (
    SELECT
    SITE_NUM
    FROM
    RLDMPROD_V.SITE_HIER S
    WHERE (S.STR_CLS_DT > DATE
    OR S.STR_CLS_DT IS NULL)
    AND S.STR_OPN_DT < DATE
    AND
      (
          (S.BAN_SHRT_DESC_E in ('Fortinos','Your Ind Grocer', 'Valu-Mart', 'Atl Your Ind Grocer', 'nofrills', 'Provigo') AND PROFL_SITE_NUM in ('L160'))
          OR (S.BAN_SHRT_DESC_E in ('Loblaw', 'Zehrs', 'Atlantic Superstore', 'Superstore', 'Wholesale Club', 'Dominion', 'Maxi', 'Retail RCWC', 'Extra Foods') AND PROFL_SITE_NUM in ('L130'))
      ) 
    --this list discludes Joe Fresh and Liquorstore
    --Your Ind Grocer | Ontario has 5 extra sites...4987,4988,4989,4990,4991
    --Provigo | Quebec has 4 extra sites...4549, 4913, 4914, 4934
    --Extra Foods | West  has 3 extra sites...4982, 4983, 4984
    AND x0.SITE_NUM NOT IN ('4987', '4988', '4989', '4990', '4991', '4549', '4913', '4914', '4934', '4982', '4983', '4984')
)
AND x3.MCH_2_ENG_DESC = 'Grocery'
AND x4.CUST_GRP_ENG_DESC = 'Market'
AND x1.CAL_YR_NUM > 2017
AND (
    (x1.CAL_YR_NUM <> YEAR(DATE)) OR
    (x1.CORP_WK_NUM <> (SELECT CORP_WK_NUM FROM RLDMPROD_V.CAL_DT WHERE CAL_DT = DATE))
    )
GROUP BY CAL_WK_END_DT, year_period, year_quarter
) A
ORDER BY week_end_dt
"""

df = spark.read.format("jdbc")\
    .option("driver", driver)\
    .option("url", url)\
    .option("dbtable", query)\
    .option("user", username)\
    .option("password", password)\
    .load()\
    .toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Pipeline & Model Specifications

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC In the following cell we include the parameters common to all the model specifications

# COMMAND ----------

# The forecast horizon is 12 weeks. So all CV folds will have a test set of exactly 12 weeks.
forecast_horizon = 12


# The prediction interval coverage.
# Since linear regression assumptions will rarely ever be met for time series data, the prediction intervals are generated via bootstrap simulations
# Currently, we set the coverage to None. i.e. no prediction interval. This is to save computation
coverage = None


# The manual specification for anomalies.
# When training, anomalies will be replaced with NA, and training will subsequently ignore these values.
  # since we're using generalized linear models (and not ARIMA's for example) removing values is not a breaking issue.
# Note that this date range was chosen subjectively to account for the massive 2020 COVID spike in sales.
anomaly_info = {
     "value_col": 'sales',
     "anomaly_df": pd.DataFrame({
       cst.START_DATE_COL: ["2020-03-14"],
       cst.END_DATE_COL: ["2020-05-23"],
       cst.ADJUSTMENT_DELTA_COL: [np.nan]
     }),
     "adjustment_delta_col": cst.ADJUSTMENT_DELTA_COL,
}


# The metadata specification.
# The data has weekly frequency (where the weeks start on Saturday).
# The end date of each week is used as the time column.
metadata = MetadataParam(
     time_col = 'week_end_dt', 
     value_col = 'sales',
     freq = 'W-SAT',
     anomaly_info = anomaly_info
)


# The events to pass to the models as regressors.
# Note that since the data is weekly, the dates for these events must match up with the end dates of the weeks.
# Currently, this includes easter, thanksgiving and 1 unit of lead and lag for each.
events = dict(
    holiday_lookup_countries = None,
        daily_event_df_dict = {
        "easter": pd.DataFrame({
            "date": ["2018-03-31", "2019-04-20", "2020-04-11", "2021-04-03", "2020-04-16"],
            "event_name": "is_event"
        }),
        "easterlead": pd.DataFrame({
            "date":["2018-04-07", "2019-04-27", "2020-04-18", "2021-04-10", "2020-04-23"],
            "event_name": "is_event2"
        }),
        "thanksgiving": pd.DataFrame({
            "date": ["2018-10-06", "2019-10-12", "2020-10-10", "2021-10-09", "2020-10-08"],
            "event_name": "is_event3"
        }),
        "thanksgivinglead": pd.DataFrame({
            "date":["2018-10-13", "2019-10-19", "2020-10-17", "2021-10-06", "2020-10-15"],
            "event_name": "is_event4"
        }),
    }
)  


# The cross-validation config within the pipeline.
# However, since we're performing cross validation using GreyKite's benchmark function, we set this to no cross validation (1 split).
# We still specify this, as otherwise, each model will perform its own cross validation within the already existing benchmark cross validation.
evaluation_period = EvaluationPeriodParam(
    test_horizon = 0,
    cv_max_splits = 1,
    periods_between_train_test = 0
)


# The cross validation metrics to compute
evaluation_metric = EvaluationMetricParam(
    cv_selection_metric = EvaluationMetricEnum.MeanAbsolutePercentError.name,
    cv_report_metrics = [
        EvaluationMetricEnum.MedianAbsolutePercentError.name,
        EvaluationMetricEnum.MeanAbsoluteError.name,
        EvaluationMetricEnum.MedianAbsoluteError.name,
        EvaluationMetricEnum.RootMeanSquaredError.name
    ],
    relative_error_tolerance = 0.01
)


# The computational metrics (e.g. how many cores to use)
# This doesn't integrate with databricks, so its doing nothing right now
computation = ComputationParam(
    n_jobs = -1,
    verbose = 1
)


# Defining the benchmark cross validation folds
# The minimum train period is set to 128 weeks so that training for all splits always includes data after the 2020 COVID spike
# We set 3 weeks between splits so that each fold does not end training on the same week of the month
  # for example, if we had 4 weeks between splits, and the first split started at the start of January, then the next split would start at the start of February, and then the next at the start of March, etc.
  # the fact that each split is always at the start of the month could introduce bias
tscv = RollingTimeSeriesSplit(
    forecast_horizon = forecast_horizon,
    min_train_periods = 128, # 1 week after anomaly ends, so we have 1 lag
    expanding_window = True,
    use_most_recent_splits = True,
    periods_between_splits = 3,
    max_splits = 18
)

# The cross validation metrics to compute
metric_dict = {
     "MAPE": EvaluationMetricEnum.MeanAbsolutePercentError,
     "RMSE": EvaluationMetricEnum.RootMeanSquaredError
 }

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We now specify each of the individual models. Note that we are currently not using any autoregressive terms, however, this may change depending on the granluarity (as this determines the amount of computation available for each series). Also note that the autoregressive effects, if any, will likely be negligable as 12 weeks is a large horizon.

# COMMAND ----------


evaluation_metric = EvaluationMetricParam(
    cv_selection_metric = EvaluationMetricEnum.MeanAbsolutePercentError.name,
    cv_report_metrics = [
        EvaluationMetricEnum.MedianAbsolutePercentError.name,
        EvaluationMetricEnum.MeanAbsoluteError.name,
        EvaluationMetricEnum.MedianAbsoluteError.name,
        EvaluationMetricEnum.RootMeanSquaredError.name
    ],
    relative_error_tolerance = 0.01
)


# COMMAND ----------

# Silverkite_0 is meant to model heavily noisy series.
# Seasonality: None | Changepoints: None | Growth: Linear | AR: None | Algo: OLS | Events: None
Silverkite_0 = ForecastConfig(
    forecast_horizon = forecast_horizon,  
    coverage = coverage,        
    metadata_param = metadata,
    computation_param = computation,
    evaluation_period_param = evaluation_period,
    evaluation_metric_param = evaluation_metric,
    model_components_param = ModelComponentsParam(
                                events = {'holiday_lookup_countries': None},                                 
                                custom = {'min_admissible_value': 0,
                                         'fit_algorithm_dict': {'fit_algorithm': 'statsmodels_ols'}
                                         },
                                growth = {'growth_term': 'linear'},
                                seasonality = {'yearly_seasonality': False,
                                               'quarterly_seasonality': False,
                                               'monthly_seasonality': False},
                                changepoints = {'changepoints_dict': None},
                                autoregression = {'autoreg_dict': None}
    )
)


# Silverkite_1 is meant to model series with a normal amount of seasonality, and no changepoints.
# Seasonality: Normal | Changepoints: None | Growth: Linear | AR: None | Algo: OLS | Events: Easter,Thanksgiving +-1
Silverkite_1 = ForecastConfig(
    forecast_horizon = forecast_horizon,  
    coverage = coverage,        
    metadata_param = metadata,
    computation_param = computation,
    evaluation_period_param = evaluation_period,
    model_components_param = ModelComponentsParam(
                                events = events,
                                custom = {'min_admissible_value': 0,
                                         'fit_algorithm_dict': {'fit_algorithm': 'statsmodels_ols'}
                                         },
                                growth = {'growth_term': 'linear'},
                                seasonality = {'yearly_seasonality': 15,
                                               'quarterly_seasonality': False,
                                               'monthly_seasonality': False},
                                changepoints = {'changepoints_dict': None},
                                autoregression = {'autoreg_dict': None}
    )
)


# Silverkite_2 is meant to model series with a normal amount of seasonality, and a light amount of changepoints.
# Seasonality: Normal | Changepoints: Light | Growth: Linear | AR: None | Algo: OLS | Events: Easter,Thanksgiving +-1
Silverkite_2 = ForecastConfig(
    forecast_horizon = forecast_horizon,  
    coverage = coverage,        
    metadata_param = metadata,
    computation_param = computation,
    evaluation_period_param = evaluation_period,
    model_components_param = ModelComponentsParam(
                                events = events,
                                custom = {'min_admissible_value': 0,
                                         'fit_algorithm_dict': {'fit_algorithm': 'statsmodels_ols'}
                                         },
                                growth = {'growth_term': 'linear'},
                                seasonality = {'yearly_seasonality': 15,
                                               'quarterly_seasonality': False,
                                               'monthly_seasonality': False},
                                changepoints = {'changepoints_dict':
                                                {'method': 'auto',
                                                 'resample_freq': '7D',
                                                 'regularization_strength': 0.6,
                                                 'potential_changepoint_distance': '14D',
                                                 'no_changepoint_distance_from_end': '180D', 
                                                 'yearly_seasonality_order': 15, 
                                                 'yearly_seasonality_change_freq': None}
                                               },
                                autoregression = {'autoreg_dict': None}
    )
)


# Silverkite_3 is meant to model series with a heavy amount of seasonality, and a normal amount changepoints. Since the number of regressors is high, we use ridge regression.
# Seasonality: Heavy | Changepoints: Normal | Growth: Linear | AR: None | Algo: Ridge | Events: Easter,Thanksgiving +-1
Silverkite_3 = ForecastConfig(
    forecast_horizon = forecast_horizon,  
    coverage = coverage,        
    metadata_param = metadata,
    computation_param = computation,
    evaluation_period_param = evaluation_period,
    model_components_param = ModelComponentsParam(
                                events = events,
                                custom = {'min_admissible_value': 0,
                                         'fit_algorithm_dict': {'fit_algorithm': 'ridge'}
                                         },
                                growth = {'growth_term': 'linear'},
                                seasonality = {'yearly_seasonality': 25,
                                               'quarterly_seasonality': False,
                                               'monthly_seasonality': False},
                                changepoints = {'changepoints_dict':
                                                {'method': 'auto',
                                                 'resample_freq': '7D',
                                                 'regularization_strength': 0.5,
                                                 'potential_changepoint_distance': '14D',
                                                 'no_changepoint_distance_from_end': '180D', 
                                                 'yearly_seasonality_order': 15, 
                                                 'yearly_seasonality_change_freq': '365D'}
                                               },
                                autoregression = {'autoreg_dict': None}
    )
)


# Silverkite_4 is meant to model series with a heavy amount of seasonality, and a light amount changepoints. Since the number of regressors is high, we use ridge regression.
# Seasonality: Heavy | Changepoints: Light | Growth: Linear | AR: None | Algo: Ridge | Events: Easter,Thanksgiving +-1
Silverkite_4 = ForecastConfig(
    forecast_horizon = forecast_horizon,  
    coverage = coverage,        
    metadata_param = metadata,
    computation_param = computation,
    evaluation_period_param = evaluation_period,
    model_components_param = ModelComponentsParam(
                                events = events,
                                custom = {'min_admissible_value': 0,
                                         'fit_algorithm_dict': {'fit_algorithm': 'ridge'}
                                         },
                                growth = {'growth_term': 'linear'},
                                seasonality = {'yearly_seasonality': 25,
                                               'quarterly_seasonality': False,
                                               'monthly_seasonality': False},
                                changepoints = {'changepoints_dict':
                                                {'method': 'auto',
                                                 'resample_freq': '7D',
                                                 'regularization_strength': 0.6,
                                                 'potential_changepoint_distance': '14D',
                                                 'no_changepoint_distance_from_end': '180D', 
                                                 'yearly_seasonality_order': 15, 
                                                 'yearly_seasonality_change_freq': None}
                                               },
                                autoregression = {'autoreg_dict': None}
    )
)


# Silverkite_5 is meant to use a Gamma regression with an inverse link. The hypothesis is that since the Gamma distribution is right skewed, and sales data has the potential to be
 # right skewed (large spikes in sales), the Gamma regression has the potential to perform well.
# Since this regression method is not reguralized, we keep the number of regressors low by keeping a normal amount of seasonality, and no changepoints.
# Seasonality: Normal | Changepoints: None | Growth: Linear | AR: None | Algo: GLM (Gamma, Inv Link) | Events: Easter,Thanksgiving +-1
Silverkite_5 = ForecastConfig(
    forecast_horizon = forecast_horizon,  
    coverage = coverage,        
    metadata_param = metadata,
    computation_param = computation,
    evaluation_period_param = evaluation_period,
    model_components_param = ModelComponentsParam(
                                events = events,
                                custom = {'min_admissible_value': 0,
                                         'fit_algorithm_dict': {'fit_algorithm': 'statsmodels_glm'}
                                         },
                                growth = {'growth_term': 'linear'},
                                seasonality = {'yearly_seasonality': 15,
                                               'quarterly_seasonality': False,
                                               'monthly_seasonality': False},
                                changepoints = {'changepoints_dict': None},
                                autoregression = {'autoreg_dict': None}
    )
)


# Silverkite_6 is the same as Silverkite_5 except we increase the change point amount from none to a light amount of changepoints.
# Seasonality: Normal | Changepoints: Light | Growth: Linear | AR: None | Algo: GLM (Gamma, Inv Link) | Events: Easter,Thanksgiving +-1
Silverkite_6 = ForecastConfig(
    forecast_horizon = forecast_horizon,  
    coverage = coverage,        
    metadata_param = metadata,
    computation_param = computation,
    evaluation_period_param = evaluation_period,
    model_components_param = ModelComponentsParam(
                                events = events,
                                custom = {'min_admissible_value': 0,
                                         'fit_algorithm_dict': {'fit_algorithm': 'statsmodels_glm'}
                                         },
                                growth = {'growth_term': 'linear'},
                                seasonality = {'yearly_seasonality': 15,
                                               'quarterly_seasonality': False,
                                               'monthly_seasonality': False},
                                changepoints = {'changepoints_dict':
                                                {'method': 'auto',
                                                 'resample_freq': '7D',
                                                 'regularization_strength': 0.6,
                                                 'potential_changepoint_distance': '14D',
                                                 'no_changepoint_distance_from_end': '180D', 
                                                 'yearly_seasonality_order': 15, 
                                                 'yearly_seasonality_change_freq': None}
                                               },
                                autoregression = {'autoreg_dict': None}
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### CV Run

# COMMAND ----------

# all models to test
configs_run = {
    "SK0": Silverkite_0,
#     "SK1": Silverkite_1,
#     "SK2": Silverkite_2,
#     "SK3": Silverkite_3,
#     "SK4": Silverkite_4,
#     "SK5": Silverkite_5,
#     "SK6": Silverkite_6,
 }


# running the cv
bm = BenchmarkForecastConfig(df = df, configs = configs_run, tscv = tscv)
bm.run()

# COMMAND ----------

# averaging errors over all folds, for each model
# the model with the least MAPE is considered the best
   # Note this metric may have to change as some series may have values close to 0
evaluation_metrics = bm.get_evaluation_metrics(metric_dict = metric_dict)
errors = evaluation_metrics.drop(columns=["split_num"]).groupby("config_name").mean()

bm.extract_forecasts()

# COMMAND ----------

errors

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Reporting Metrics

# COMMAND ----------

best_model = errors[errors.test_MAPE == errors.test_MAPE.min()].index[0]
rolling_forecasts = bm.result[best_model]["rolling_forecast_df"]
rolling_forecasts['e'] = rolling_forecasts['actual'] - rolling_forecasts['forecast']
rolling_forecasts['pe'] = rolling_forecasts['e']/rolling_forecasts['actual']
rolling_forecasts.pivot(index = ['train_end_date', 'split_num'], columns = ['forecast_step'], values = ['e', 'pe']).reset_index().display()

# COMMAND ----------

rolling_forecasts_period = rolling_forecasts.copy()
rolling_forecasts_period['period'] = (rolling_forecasts_period['forecast_step'] - 1) // 4
rolling_forecasts_period = rolling_forecasts_period.groupby(by = ['train_end_date', 'split_num', 'period'], as_index = False).sum()
rolling_forecasts_period['e'] = rolling_forecasts_period['actual'] - rolling_forecasts_period['forecast']
rolling_forecasts_period['pe'] = rolling_forecasts_period['e']/rolling_forecasts_period['actual']
rolling_forecasts_period.pivot(index = ['train_end_date', 'split_num'], columns = ['forecast_step'], values = ['e', 'pe']).reset_index().display()

# COMMAND ----------

rolling_forecasts_quarter = rolling_forecasts.groupby(by = ['train_end_date', 'split_num'], as_index = False).sum()
rolling_forecasts_quarter['e'] = rolling_forecasts_quarter['actual'] - rolling_forecasts_quarter['forecast']
rolling_forecasts_quarter['pe'] = rolling_forecasts_quarter['e']/rolling_forecasts_quarter['actual']
rolling_forecasts_quarter.pivot(index = ['train_end_date', 'split_num'], columns = ['forecast_step'], values = ['e', 'pe']).reset_index().display()
