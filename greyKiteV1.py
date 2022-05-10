# -*- coding: utf-8 -*-
"""
Created on: 

@author: Grace Jin

Team: Finance Analytics

Project Name: 
    
Project Desc: 
    
Updated On:
    
Packges: tika 
    
Config: 
"""

import pandas as pd
import plotly
from collections import defaultdict
import warnings
from greykite.framework.templates.autogen.forecast_config import ForecastConfig,MetadataParam
from greykite.framework.templates.forecaster import Forecaster 
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.utils.result_summary import summarize_grid_search_results
warnings.filterwarnings("ignore")

# read in file
df =pd.read_csv('C:/Users/grjin/Desktop/Study/greyKite/Electric_Production.csv')

# change string to datetime
df['DATE'] = df['DATE'].astype('datetime64[ns]')

# col renmae
df.rename(columns = {'DATE' : 'ts', 'Value' :'y'}, inplace = True)

df.head(100)
