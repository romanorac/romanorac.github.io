---
layout: post
title:  "Churn prediction"
use_math: true
categories: machine learning
---

<div style="font-size:80%; text-align:center;">
<div align="middle">
<img src="{{site.url}}/assets/images/2019-01-25-churn-prediction/mantas-hesthaven-135478-unsplash.jpg
" alt="Photo by Mantas Hesthaven on Unsplash">
</div>
Photo by Mantas Hesthaven on Unsplash
</div>

Customer churn, also known as customer attrition, occurs when customers stop doing business with a company. The companies are interested in identifying segments of these customers because the price for acquiring a new customer is usually higher than retaining the old one. For example, if Netflix knew a segment of customers who were at risk of churning they could proactively engage them with special offers instead of simply losing them.

In this blog post, we will create a simple customer churn prediction model using [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn).  We chose a decision tree to model churned customers, pandas for data crunching and matplotlib for visualizations. We will do all of that above in Python.
The code can be used with another dataset with a few minor adjustments to train the baseline model. We also provide few references and give ideas for new features and improvements. 

You can run this code by downloading this [Jupyter notebook]({{site.url}}/assets/notebooks/2019-01-25-churn-prediction.ipynb).
    
Follow me on [twitter](https://twitter.com/romanorac) to get latest updates.

Let's get started.

## Requirements


```python
import platform
import pandas as pd
import sklearn
import numpy as np
import graphviz
import matplotlib
import matplotlib.pyplot as plt

%matplotlib inline
```


```python
print('python version', platform.python_version())
print('pandas version', pd.__version__)
print('sklearn version', sklearn.__version__)
print('numpy version', np.__version__)
print('graphviz version', graphviz.__version__)
print('matplotlib version', matplotlib.__version__)
```

    python version 3.7.0
    pandas version 0.23.4
    sklearn version 0.19.2
    numpy version 1.15.1
    graphviz version 0.10.1
    matplotlib version 2.2.3


## Data Preprocessing

We use pandas to read the dataset and preprocess it. Telco dataset has one customer per line with many columns (features).
There aren't any rows with all missing values or duplicates (this rarely happens with real-world datasets). 
There are 11 samples that have TotalCharges set to " ", which seems like a mistake in the data. We remove those samples and set the type to numeric (float).


```python
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.shape
```




    (7043, 21)




```python
df.head()
```




<div style="overflow-x:scroll;">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>...</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>1</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-GNVDE</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>34</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>1889.5</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-QPYBK</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-CFOCW</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>45</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>
</div>




```python
df = df.dropna(how="all")  # remove samples with all missing values
df.shape
```




    (7043, 21)




```python
df = df[~df.duplicated()] # remove duplicates
df.shape
```




    (7043, 21)




```python
total_charges_filter = df.TotalCharges == " "
df = df[~total_charges_filter]
df.shape
```




    (7032, 21)




```python
df.TotalCharges = pd.to_numeric(df.TotalCharges)
```

## Exploratory Data Analysis

We have 2 types of features in the dataset: categorical (two or more values and without any order) and numerical. Most of the feature names are self-explanatory, except for:
 - Partner: whether the customer has a partner or not (Yes, No),
 - Dependents: whether the customer has dependents or not (Yes, No),
 - OnlineBackup: whether the customer has online backup or not (Yes, No, No internet service),
 - tenure: number of months the customer has stayed with the company,
 - MonthlyCharges: the amount charged to the customer monthly,
 - TotalCharges: the total amount charged to the customer.
 
There are 7032 customers in the dataset and 19 features without customerID (non-informative) and Churn column (target variable). Most of the categorical features have 4 or less unique values.


```python
df.describe(include='all')
```




<div style="overflow-x:scroll;">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>...</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7032</td>
      <td>7032</td>
      <td>7032.000000</td>
      <td>7032</td>
      <td>7032</td>
      <td>7032.000000</td>
      <td>7032</td>
      <td>7032</td>
      <td>7032</td>
      <td>7032</td>
      <td>...</td>
      <td>7032</td>
      <td>7032</td>
      <td>7032</td>
      <td>7032</td>
      <td>7032</td>
      <td>7032</td>
      <td>7032</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>7032</td>
      <td>2</td>
      <td>NaN</td>
      <td>2</td>
      <td>2</td>
      <td>NaN</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>7989-CHGTL</td>
      <td>Male</td>
      <td>NaN</td>
      <td>No</td>
      <td>No</td>
      <td>NaN</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>No</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>3549</td>
      <td>NaN</td>
      <td>3639</td>
      <td>4933</td>
      <td>NaN</td>
      <td>6352</td>
      <td>3385</td>
      <td>3096</td>
      <td>3497</td>
      <td>...</td>
      <td>3094</td>
      <td>3472</td>
      <td>2809</td>
      <td>2781</td>
      <td>3875</td>
      <td>4168</td>
      <td>2365</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5163</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.162400</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32.421786</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>64.798208</td>
      <td>2283.300441</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.368844</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24.545260</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.085974</td>
      <td>2266.771362</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.250000</td>
      <td>18.800000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>35.587500</td>
      <td>401.450000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>70.350000</td>
      <td>1397.475000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>55.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>89.862500</td>
      <td>3794.737500</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>72.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>118.750000</td>
      <td>8684.800000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 21 columns</p>
</div>
</div>


We combine features into two lists so that we can analyze them jointly. 

```python
categorical_features = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
target = "Churn"
```

### Feature distribution

We plot distributions for numerical and categorical features to check for outliers and compare feature distributions with target variable.

#### Numerical features distribution

Numeric summarizing techniques (mean, standard deviation, etc.) don't show us spikes, shapes of distributions and it is hard to observe outliers with it. That is the reason we use histograms.


```python
df[numerical_features].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tenure</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7032.000000</td>
      <td>7032.000000</td>
      <td>7032.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>32.421786</td>
      <td>64.798208</td>
      <td>2283.300441</td>
    </tr>
    <tr>
      <th>std</th>
      <td>24.545260</td>
      <td>30.085974</td>
      <td>2266.771362</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>18.250000</td>
      <td>18.800000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.000000</td>
      <td>35.587500</td>
      <td>401.450000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>29.000000</td>
      <td>70.350000</td>
      <td>1397.475000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>55.000000</td>
      <td>89.862500</td>
      <td>3794.737500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>72.000000</td>
      <td>118.750000</td>
      <td>8684.800000</td>
    </tr>
  </tbody>
</table>
</div>



At first glance, there aren't any outliers in the data. No data point is disconnected from distribution or too far from the mean value. To confirm that we would need to calculate [interquartile range (IQR)](https://www.purplemath.com/modules/boxwhisk3.htm) and show that values of each numerical feature are within the 1.5 IQR from first and third quartile. 

We could convert numerical features to ordinal intervals. For example, tenure is numerical, but often we don't care about small numeric differences and instead group tenure to customers with short, medium and long term tenure. One reason to convert it would be to reduce the noise, often small fluctuates are just noise.


```python
df[numerical_features].hist(bins=30, figsize=(10, 7))
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x10adb71d0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x10ed897f0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x10edb2b00>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x10eddce10>]],
          dtype=object)



<div align="middle">
<figure>
<img src="{{site.url}}/assets/images/2019-01-25-churn-prediction/output_22_1.png" alt="Histograms of numerical features">
<figcaption>Histograms of numerical features</figcaption>
</figure>
</div>


We look at distributions of numerical features in relation to the target variable. We can observe that the greater TotalCharges and tenure are the less is the probability of churn.


```python
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
df[df.Churn == "No"][numerical_features].hist(bins=30, color="blue", alpha=0.5, ax=ax)
df[df.Churn == "Yes"][numerical_features].hist(bins=30, color="red", alpha=0.5, ax=ax)
```




    array([<matplotlib.axes._subplots.AxesSubplot object at 0x10eed6a90>,
           <matplotlib.axes._subplots.AxesSubplot object at 0x11125b8d0>,
           <matplotlib.axes._subplots.AxesSubplot object at 0x11127ef60>],
          dtype=object)




<div align="middle">
<figure>
<img src="{{site.url}}/assets/images/2019-01-25-churn-prediction/output_24_1.png" alt="Numerical features in relation to the target variable">
<figcaption>Numerical features in relation to the target variable</figcaption>
</figure>
</div>


#### Categorical feature distribution

To analyze categorical features, we use bar charts. We observe that Senior citizens and customers without phone service are less represented in the data.


```python
ROWS, COLS = 4, 4
fig, ax = plt.subplots(ROWS, COLS, figsize=(18, 18))
row, col = 0, 0
for i, categorical_feature in enumerate(categorical_features):
    if col == COLS - 1:
        row += 1
    col = i % COLS
    df[categorical_feature].value_counts().plot('bar', ax=ax[row, col]).set_title(categorical_feature)
```


<div align="middle">
<figure>
<img src="{{site.url}}/assets/images/2019-01-25-churn-prediction/output_26_0.png" alt="Distribution of categorical features">
<figcaption>Distribution of categorical features</figcaption>
</figure>
</div>


The next step is to look at categorical features in relation to the target variable. We do this only for contract feature. Users who have a month-to-month contract are more likely to churn than users with long term contracts.


```python
feature = 'Contract'
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
df[df.Churn == "No"][feature].value_counts().plot('bar', ax=ax[0]).set_title('not churned')
df[df.Churn == "Yes"][feature].value_counts().plot('bar', ax=ax[1]).set_title('churned')
```




    Text(0.5,1,'churned')




<div align="middle">
<figure>
<img src="{{site.url}}/assets/images/2019-01-25-churn-prediction/output_28_1.png" alt="Contract feature in relation to the target variable">
<figcaption>Contract feature in relation to the target variable</figcaption>
</figure>
</div>


#### Target variable distribution

Target variable distribution shows that we are dealing with an imbalanced problem as there are many more non-churned as churned users. The model would achieve high accuracy as it would mostly predict majority class - users who didn't churn in our example.

Few things we can do to minimize the influence of imbalanced dataset:
- resample data (https://imbalanced-learn.readthedocs.io/en/stable/),
- collect more samples,
- use precision and recall as accuracy metrics.


```python
df[target].value_counts().plot('bar').set_title('churned')
```




    Text(0.5,1,'churned')




<div align="middle">
<figure>
<img src="{{site.url}}/assets/images/2019-01-25-churn-prediction/output_31_1.png" alt="Target variable distribution">
<figcaption>Target variable distribution</figcaption>
</figure>
</div>


## Features

Telco dataset is already grouped by customerID so it is difficult to add new features. When working on the churn prediction we usually get a dataset that has one entry per customer session (customer activity in a certain time). Then we could add features like: 
 - number of sessions before buying something,
 - average time per session,
 - time difference between sessions (frequent or less frequent customer),
 - is a customer only in one country.

Sometimes we even have customer event data, which enables us to find patterns of customer behavior in relation to the outcome (churn).

### Encoding features

To prepare the dataset for modeling churn, we need to encode categorical features to numbers. This means encoding "Yes", "No" to 0 and 1 so that algorithm can work with the data. This process is called [onehot encoding](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/).


```python
from sklearn.preprocessing import LabelEncoder

categorical_feature_names = []
label_encoders = {}
for categorical in categorical_features + [target]:
    label_encoders[categorical] = LabelEncoder()
    df[categorical] = label_encoders[categorical].fit_transform(df[categorical])
    names = label_encoders[categorical].classes_.tolist()
    print('Label encoder %s - values: %s' % (categorical, names))
    if categorical == target:
        continue
    categorical_feature_names.extend([categorical + '_' + str(name) for name in names])
```

    Label encoder gender - values: ['Female', 'Male']
    Label encoder SeniorCitizen - values: [0, 1]
    Label encoder Partner - values: ['No', 'Yes']
    Label encoder Dependents - values: ['No', 'Yes']
    Label encoder PhoneService - values: ['No', 'Yes']
    Label encoder MultipleLines - values: ['No', 'No phone service', 'Yes']
    Label encoder InternetService - values: ['DSL', 'Fiber optic', 'No']
    Label encoder OnlineSecurity - values: ['No', 'No internet service', 'Yes']
    Label encoder OnlineBackup - values: ['No', 'No internet service', 'Yes']
    Label encoder DeviceProtection - values: ['No', 'No internet service', 'Yes']
    Label encoder TechSupport - values: ['No', 'No internet service', 'Yes']
    Label encoder StreamingTV - values: ['No', 'No internet service', 'Yes']
    Label encoder StreamingMovies - values: ['No', 'No internet service', 'Yes']
    Label encoder Contract - values: ['Month-to-month', 'One year', 'Two year']
    Label encoder PaperlessBilling - values: ['No', 'Yes']
    Label encoder PaymentMethod - values: ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check']
    Label encoder Churn - values: ['No', 'Yes']



```python
df.head()
```




<div style="overflow-x:scroll;">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>...</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-GNVDE</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>56.95</td>
      <td>1889.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-QPYBK</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-CFOCW</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>
</div>



## Classifier

We use sklearn, a Machine Learning library in Python, to create a classifier.
The sklearn way is to use pipelines that define feature processing and the classifier. In our example, the pipeline takes a dataset in the input, it preprocesses features and trains the classifier.
When trained, it takes the same input and returns predictions in the output. 

In the pipeline, we separately process categorical and numerical features. We onehot encode categorical features and scale numerical features by removing the mean and scaling them to unit variance.
We chose a decision tree model because of its interpretability and set max depth to 3 (arbitrarily).


```python
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        return df[self.key]
```


```python
pipeline = Pipeline(
    [
        (
            "union",
            FeatureUnion(
                transformer_list=[
                    (
                        "categorical_features",
                        Pipeline(
                            [
                                ("selector", ItemSelector(key=categorical_features)),
                                ("onehot", OneHotEncoder()),
                            ]
                        ),
                    )
                ]
                + [
                    (
                        "numerical_features",
                        Pipeline(
                            [
                                ("selector", ItemSelector(key=numerical_features)),
                                ("scalar", StandardScaler()),
                            ]
                        ),
                    )
                ]
            ),
        ),
        ("classifier", tree.DecisionTreeClassifier(max_depth=3, random_state=42)),
    ]
)
```

### Training the model

We split the dataset to train (75% samples) and test (25% samples). 
We train (fit) the pipeline and make predictions. 


```python
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)

pipeline.fit(df_train, df_train[target])
pred = pipeline.predict(df_test)
```

### Testing the model

With classification_report we calculate precision and recall with actual and predicted values.
For class 1 (churned users) model achieves 0.67 precision and 0.37 recall.
Precision tells us how many churned users did our classifier predicted correctly. On the other side, recall tell us how many churned users it missed. In layman terms, the classifier is not very accurate for churned users.


```python
from sklearn.metrics import classification_report

print(classification_report(df_test[target], pred))
```

                 precision    recall  f1-score   support
    
              0       0.81      0.94      0.87      1300
              1       0.67      0.37      0.48       458
    
    avg / total       0.77      0.79      0.77      1758
    


## Model interpretability

Decision Tree model uses Contract, MonthlyCharges, InternetService, TotalCharges, and tenure features to make a decision if a customer will churn or not. These features separate churned customers from others well based on the split criteria in the decision tree.

Each customer sample traverses the tree and final node gives the prediction. 
For example, if Contract_Month-to-month is:
 - equal to 0, continue traversing the tree with True branch, 
 - equal to 1, continue traversing the tree with False branch, 
 - not defined, it outputs the class 0.
 
This is a great approach to see how the model is making a decision or if any features sneaked in our model that shouldn't be there.


```python
dot_data = tree.export_graphviz(pipeline.named_steps['classifier'], out_file=None, 
                         feature_names = categorical_feature_names + numerical_features,
                         class_names=[str(el) for el in pipeline.named_steps.classifier.classes_],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data) 
graph
```



<div align="middle">
<figure>
<img src="{{site.url}}/assets/images/2019-01-25-churn-prediction/output_47_0.svg" alt="Interpretation of the Decision tree model">
<figcaption>Interpretation of the Decision tree model</figcaption>
</figure>
</div>




## Further reading

- [Handling class imbalance in customer churn prediction](https://www.sciencedirect.com/science/article/pii/S0957417408002121) - how can we better handle class imbalance in churn prediction.
- [A Survey on Customer Churn Prediction using Machine Learning Techniques](https://www.researchgate.net/publication/310757545_A_Survey_on_Customer_Churn_Prediction_using_Machine_Learning_Techniques) - This paper reviews the most popular machine learning algorithms used by researchers for churn  predicting.
- [Telco customer churn on kaggle](https://www.kaggle.com/blastchar/telco-customer-churn) - churn analysis on kaggle.
- [WTTE-RNN-Hackless-churn-modeling](https://ragulpr.github.io/2016/12/22/WTTE-RNN-Hackless-churn-modeling) - event based churn prediction.
