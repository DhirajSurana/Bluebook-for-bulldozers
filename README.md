<div class="cell code" data-_cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" data-_uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:00:34.989872Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:34.9554Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:34.955594Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:00:34.99067Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:34.954144Z&quot;}" data-trusted="true">

``` python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

</div>

<div class="cell markdown">

# Import libraries

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:00:36.352388Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:34.993479Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:34.993639Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:00:36.353204Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:34.993147Z&quot;}" data-trusted="true">

``` python
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_log_error, mean_absolute_error
```

</div>

<div class="cell markdown">

# Helper function to import the dataset

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:00:36.359452Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:36.355352Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:36.355391Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:00:36.360145Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:36.355021Z&quot;}" data-trusted="true">

``` python
def import_dataset(path):
    df = pd.read_csv(path, parse_dates = ['saledate'])
    return df
```

</div>

<div class="cell markdown">

# Helper function to preprocess dataframe

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:00:36.373186Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:36.361855Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:36.361902Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:00:36.373933Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:36.361455Z&quot;}" data-trusted="true">

``` python
def preprocess_dataframe_for_model(df):
    # change all srting type to categorical type
    for label, content in df.items():
        if pd.api.types.is_string_dtype(content):
            df[label]=df[label].astype("category").cat.as_ordered()
            
    # enrich the dataframe 
    enrich_df(df)
    df.drop("saledate",axis=1,inplace=True)
    
    # fill the numerical missing values with median and non-numerical values with their (category no. + 1)      
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                df[label]=content.fillna(content.median())
        else:
            df[label]=pd.Categorical(content).codes+1
    return df
```

</div>

<div class="cell markdown">

# Helper function to enrich the dataframe

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:00:36.386938Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:36.377474Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:36.377535Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:00:36.387727Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:36.376464Z&quot;}" data-trusted="true">

``` python
def enrich_df(df):
    """
    Adds following columns to dataframe saleYear, saleMonth, saleDay, saledayOfWeek, saleDayOfYear
    """
    temp_dict={
    "saleYear":"year",
    "saleMonth":"month",
    "saleDay":"day",
    "saleDayOfWeek":"dayofweek",
    "saleDayOfYear":"dayofyear"
    }
    
    for column, attribute in temp_dict.items():
        df[column] = df["saledate"].dt.__getattribute__(attribute)
    return df
```

</div>

<div class="cell markdown">

# Helper function to evaluate model

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:00:36.404568Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:36.389448Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:36.389484Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:00:36.405503Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:36.388854Z&quot;}" data-trusted="true">

``` python
# Create evaluation function (the competition uses Root Mean Square Log Error)

def rmsle(y_test, y_preds):
    return np.sqrt(mean_squared_log_error(y_test, y_preds))

# Create function to evaluate our model
def show_scores(model):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_valid)
    scores = {"Training MAE": mean_absolute_error(y_train, train_preds),
              "Valid MAE": mean_absolute_error(y_valid, val_preds),
              "Training RMSLE": rmsle(y_train, train_preds),
              "Valid RMSLE": rmsle(y_valid, val_preds),
              "Training R^2": model.score(X_train, y_train),
              "Valid R^2": model.score(X_valid, y_valid)}
    return scores
```

</div>

<div class="cell markdown">

# Import train and valid dataset

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:00:40.608879Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:36.407072Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:36.407103Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:00:40.609716Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:36.406834Z&quot;}" data-trusted="true">

``` python
# import train and valid dataset
df_test_and_valid = import_dataset("../input/bluebook-for-bulldozers/TrainAndValid.csv")
```

</div>

<div class="cell markdown">

# Different attributes of train and valid dataframe

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:00:41.859956Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:40.611456Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:40.611487Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:00:41.860957Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:40.611225Z&quot;}" data-trusted="true">

``` python
df_test_and_valid.info()
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:00:43.047504Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:41.862959Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:41.863012Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:00:43.048415Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:41.862603Z&quot;}" data-trusted="true">

``` python
df_test_and_valid.isna().sum()
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:00:43.177265Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:43.049879Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:43.049911Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:00:43.178287Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:43.049611Z&quot;}" data-trusted="true">

``` python
df_test_and_valid.describe()
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:00:43.195017Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:43.181218Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:43.181247Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:00:43.195768Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:43.180992Z&quot;}" data-trusted="true">

``` python
df_test_and_valid["saledate"].value_counts()
```

</div>

<div class="cell markdown">

# Visual plots to understand data in a better way

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:00:43.561931Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:43.197743Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:43.197784Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:00:43.56288Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:43.19699Z&quot;}" data-trusted="true">

``` python
fig, ax = plt.subplots(figsize=(15,5))
ax.scatter(df_test_and_valid["saledate"][:1000],df_test_and_valid["SalePrice"][:1000])
ax.set_xlabel("Sale Date",fontsize=14)
ax.set_ylabel("Sale Price",fontsize=14);
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:00:43.829228Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:43.564986Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:43.565037Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:00:43.830087Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:43.564532Z&quot;}" data-trusted="true">

``` python
fig, ax = plt.subplots(figsize=(15,5))
ax.hist(df_test_and_valid["SalePrice"])
ax.set_xlabel('Price',fontsize=14)
ax.set_ylabel('Sales',fontsize=14)
ax.set_title("Distribution of sales",fontsize=16);
```

</div>

<div class="cell markdown">

# Preprocess train and valid dataframe

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:00:51.712736Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:43.831644Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:43.831676Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:00:51.713603Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:43.831424Z&quot;}" data-trusted="true">

``` python
df_test_and_valid_modified=preprocess_dataframe_for_model(df_test_and_valid)
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:00:51.740215Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:51.715573Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:51.715615Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:00:51.740954Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:51.714723Z&quot;}" data-trusted="true">

``` python
df_test_and_valid_modified.head()
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:00:51.804593Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:51.742674Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:51.742711Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:00:51.805488Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:51.742059Z&quot;}" data-trusted="true">

``` python
df_test_and_valid_modified.info()
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:00:51.853103Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:51.806987Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:51.807018Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:00:51.853739Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:51.806779Z&quot;}" data-trusted="true">

``` python
df_test_and_valid_modified.isna().sum()
```

</div>

<div class="cell markdown">

# Modelling

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:00:51.8585Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:51.855614Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:51.855648Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:00:51.859244Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:51.855215Z&quot;}" data-trusted="true">

``` python
model = RandomForestRegressor(n_jobs=-1,random_state=42)
```

</div>

<div class="cell markdown">

## Slpitting train and valid data

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:00:51.946914Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:51.860919Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:51.861098Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:00:51.947658Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:51.860613Z&quot;}" data-trusted="true">

``` python
df_train=df_test_and_valid_modified[df_test_and_valid_modified.saleYear!=2012]
df_valid=df_test_and_valid_modified[df_test_and_valid_modified.saleYear==2012]
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:00:51.995533Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:51.951655Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:51.951689Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:00:51.996637Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:51.951402Z&quot;}" data-trusted="true">

``` python
X_train, y_train= df_train.drop(["SalePrice"],axis=1), df_train.SalePrice
X_valid, y_valid= df_valid.drop(["SalePrice"],axis=1), df_valid.SalePrice
```

</div>

<div class="cell markdown">

## Preparing for hyperparameter tuning

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:00:52.003993Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:51.998662Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:51.998714Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:00:52.004783Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:51.998158Z&quot;}" data-trusted="true">

``` python
search_grid={
    "n_estimators": np.arange(10, 30, 5),
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": np.arange(2, 10, 4),
    "min_samples_leaf": np.arange(1, 10, 4),
    "max_features": [0.5, 1, "sqrt", "auto"],
    "max_samples": [10000]
}
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:00:52.017034Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:52.006601Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:52.006635Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:00:52.017962Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:52.006149Z&quot;}" data-trusted="true">

``` python
lis=search_grid.values()
pro=1
for index,li in enumerate(lis):
    pro=len(li)*pro
print(f'Now we will fit {pro*2} models')
```

</div>

<div class="cell markdown">

# GridSreachCV for hyperparameter tuning

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:06:24.110959Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:00:52.019795Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:00:52.019827Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:06:24.111983Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:00:52.019577Z&quot;}" data-trusted="true">

``` python
%%time
ideal_model=GridSearchCV(
    RandomForestRegressor(),
    param_grid=search_grid,
    n_jobs=-1,
    cv=2
)
ideal_model.fit(X_train,y_train)
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:06:24.117425Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:06:24.114121Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:06:24.114158Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:06:24.118272Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:06:24.113649Z&quot;}" data-trusted="true">

``` python
# ideal_model=pickle.load(open("./bulldozer-sale-price-predictor.pkl","rb"))
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:06:24.130222Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:06:24.119998Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:06:24.120037Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:06:24.130999Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:06:24.119756Z&quot;}" data-trusted="true">

``` python
ideal_model.best_params_
```

</div>

<div class="cell markdown">

## How our model performs

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:06:27.225655Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:06:24.132864Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:06:24.132914Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:06:27.226661Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:06:24.132312Z&quot;}" data-trusted="true">

``` python
show_scores(ideal_model)
```

</div>

<div class="cell markdown">

# Import test data

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:06:27.437445Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:06:27.228654Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:06:27.228694Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:06:27.438334Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:06:27.228428Z&quot;}" data-trusted="true">

``` python
# import test data
df_test=import_dataset('../input/bluebook-for-bulldozers/Test.csv')
df_test.head()
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:06:27.646585Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:06:27.439773Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:06:27.439801Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:06:27.647407Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:06:27.439573Z&quot;}" data-trusted="true">

``` python
df_test_modified=preprocess_dataframe_for_model(df_test)
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:06:27.67486Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:06:27.653111Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:06:27.653152Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:06:27.675605Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:06:27.652511Z&quot;}" data-trusted="true">

``` python
df_test_modified.head()
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:06:27.695827Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:06:27.677299Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:06:27.677342Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:06:27.696496Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:06:27.676777Z&quot;}" data-trusted="true">

``` python
df_test_modified.isna().sum()
```

</div>

<div class="cell markdown">

# Collecting predictions of our model

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:06:27.753315Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:06:27.698066Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:06:27.698098Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:06:27.754132Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:06:27.697808Z&quot;}" data-trusted="true">

``` python
test_preds=ideal_model.predict(df_test_modified)
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:06:27.760144Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:06:27.755794Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:06:27.75583Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:06:27.761592Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:06:27.755375Z&quot;}" data-trusted="true">

``` python
df_preds=pd.DataFrame()
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:06:27.772841Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:06:27.762891Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:06:27.762917Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:06:27.773721Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:06:27.762701Z&quot;}" data-trusted="true">

``` python
df_preds["SalesID"]=df_test_modified.SalesID
df_preds["SalePrice"]=test_preds
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:06:27.794229Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:06:27.775313Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:06:27.775371Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:06:27.795089Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:06:27.774994Z&quot;}" data-trusted="true">

``` python
df_preds
```

</div>

<div class="cell markdown">

# Saving csv file for submission

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:06:27.860549Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:06:27.797379Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:06:27.79743Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:06:27.861418Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:06:27.796709Z&quot;}" data-trusted="true">

``` python
df_preds.to_csv("SalePrice-Submission.csv",index=False)
```

</div>

<div class="cell markdown">

# Saving our model

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:06:27.909202Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:06:27.863133Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:06:27.863167Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:06:27.909993Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:06:27.862921Z&quot;}" data-trusted="true">

``` python
pickle.dump(ideal_model,open('bulldozer-sale-price-predictor.pkl',"wb"))
```

</div>

<div class="cell code" data-execution="{&quot;shell.execute_reply&quot;:&quot;2022-04-09T09:06:27.914859Z&quot;,&quot;shell.execute_reply.started&quot;:&quot;2022-04-09T09:06:27.911764Z&quot;,&quot;iopub.execute_input&quot;:&quot;2022-04-09T09:06:27.911801Z&quot;,&quot;iopub.status.idle&quot;:&quot;2022-04-09T09:06:27.91552Z&quot;,&quot;iopub.status.busy&quot;:&quot;2022-04-09T09:06:27.911392Z&quot;}" data-trusted="true">

``` python
# df_preds=pd.read_csv("./SalePrice-Submission.csv")
# df_preds
```

</div>
