---
layout: post
title:  "Exploratory Data Analysis with pandas"
use_math: true
categories: machine learning
---

<div style="font-size:80%; text-align:center;">
<div align="middle">
<img src="{{site.url}}/assets/images/2019-11-13-5-lesser-known-pandas-tricks-part-2/pandas.jpeg
" alt="Pandas">
</div>
Pandas provides high-performance, easy-to-use data structures and data analysis tools for the Python
</div>

As a Data Scientist, I use pandas daily and I am always amazed by how many functionalities it has. 
These 5 pandas tricks will make you better with Exploratory Data Analysis, which is an approach to analyzing data sets to summarize their main characteristics, often with visual methods. Many complex visualizations can be achieved with pandas and usually, there is no need to import other libraries. 
To run the examples download this [Jupyter notebook]({{site.url}}/assets/notebooks/2019-11-13-5-lesser-known-pandas-tricks-part-2.ipynb).

In case you've missed <a target="_blank" href="https://towardsdatascience.com/5-lesser-known-pandas-tricks-e8ab1dd21431">5 lesser-known pandas tricks</a>.


## Setup


```python
%matplotlib inline
```


```python
from platform import python_version

import matplotlib as mpl
import pandas as pd
```


```python
print("python version==%s" % python_version())
print("pandas==%s" % pd.__version__)
print("matplotlib==%s" % mpl.__version__)
```

    python version==3.7.3
    pandas==0.25.0
    matplotlib==3.0.3



```python
pd.np.random.seed(42)
```

Let's create a pandas DataFrame with 5 columns and 1000 rows:
- a1 and a2 have random samples drawn from a normal (Gaussian) distribution,
- a3 has randomly distributed integers from a set of (0, 1, 2, 3, 4),
- y1 has numbers spaced evenly on a log scale from 0 to 1,
- y2 has randomly distributed integers from a set of (0, 1).


```python
mu1, sigma1 = 0, 0.1
mu2, sigma2 = 0.2, 0.2
n = 1000

df = pd.DataFrame(
    {
        "a1": pd.np.random.normal(mu1, sigma1, n),
        "a2": pd.np.random.normal(mu2, sigma2, n),
        "a3": pd.np.random.randint(0, 5, n),
        "y1": pd.np.logspace(0, 1, num=n),
        "y2": pd.np.random.randint(0, 2, n),
    }
)
```

Readers with Machine Learning background will recognize the notation where a1, a2 and a3 represent attributes and y1 and y2 represent target variables.
In short, Machine Learning algorithms try to find patterns in the attributes and use them to predict the unseen target variable - but this is not the main focus of this blog post.
The reason that we have two target variables (y1 and y2) in the DataFrame (one binary and one continuous) is to make examples easier to follow.

We reset the index, which adds the index column to the DataFrame to enumerates the rows.


```python
df.reset_index(inplace=True)
```


```python
df.head()
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
      <th>index</th>
      <th>a1</th>
      <th>a2</th>
      <th>a3</th>
      <th>y1</th>
      <th>y2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.049671</td>
      <td>0.479871</td>
      <td>2</td>
      <td>1.000000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-0.013826</td>
      <td>0.384927</td>
      <td>2</td>
      <td>1.002308</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.064769</td>
      <td>0.211926</td>
      <td>2</td>
      <td>1.004620</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.152303</td>
      <td>0.070613</td>
      <td>3</td>
      <td>1.006939</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-0.023415</td>
      <td>0.339645</td>
      <td>4</td>
      <td>1.009262</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## 1. Plot Customizations

When I first started working with pandas, the plotting functionality seemed clunky.
I was so wrong on this one because pandas exposes full matplotlib functionality.

### 1.1 Customize axes on the output

Pandas plot function returns matplotlib.axes.Axes or numpy.ndarray of them so we can additionally customize our plots.
In the example below, we add a horizontal and a vertical red line to pandas line plot.
This is useful if we need to: 
- add the average line to a histogram,
- mark an important point on the plot, etc.


```python
ax = df.y1.plot()
ax.axhline(6, color="red", linestyle="--")
ax.axvline(775, color="red", linestyle="--")
```






<div align="middle">
<img src="{{site.url}}/assets/images/2019-11-13-5-lesser-known-pandas-tricks-part-2/output_16_1.png">
</div>



### 1.2 Customize axes on the input

Pandas plot function also takes Axes argument on the input. 
This enables us to customize plots to our liking.
In the example below, we create a two-by-two grid with different types of plots.


```python
fig, ax = mpl.pyplot.subplots(2, 2, figsize=(14,7))
df.plot(x="index", y="y1", ax=ax[0, 0])
df.plot.scatter(x="index", y="y2", ax=ax[0, 1])
df.plot.scatter(x="index", y="a3", ax=ax[1, 0])
df.plot(x="index", y="a1", ax=ax[1, 1])
```








<div align="middle">
<img src="{{site.url}}/assets/images/2019-11-13-5-lesser-known-pandas-tricks-part-2/output_19_1.png">
</div>


## 2. Histograms

A histogram is an accurate representation of the distribution of numerical data. 
It is an estimate of the probability distribution of a continuous variable and was first introduced by Karl Pearson[[1]](https://en.wikipedia.org/wiki/Histogram).

### 2.1 Stacked Histograms

Pandas enables us to compare distributions of multiple variables on a single histogram with a single function call.


```python
df[["a1", "a2"]].plot(bins=30, kind="hist")
```








<div align="middle">
<img src="{{site.url}}/assets/images/2019-11-13-5-lesser-known-pandas-tricks-part-2/output_23_1.png">
</div>


To create two separate plots, we set `subplots=True`.


```python
df[["a1", "a2"]].plot(bins=30, kind="hist", subplots=True)
```







<div align="middle">
<img src="{{site.url}}/assets/images/2019-11-13-5-lesser-known-pandas-tricks-part-2/output_25_1.png">
</div>




### 2.2 Probability Density Function

A Probability density function (PDF) is a function whose value at any given sample in the set of possible values can be interpreted as a relative likelihood that the value of the random variable would equal that sample [[2]](https://en.wikipedia.org/wiki/Probability_density_function).
In other words, the value of the PDF at two different samples can be used to infer, in any particular draw of the random variable, how much more likely it is that the random variable would equal one sample compared to the other sample.

Note that in pandas, there is a `density=1` argument that we can pass to `hist` function, but with it, we don't get a PDF, because the y-axis is not on the scale from 0 to 1 as can be seen on the plot below.
The reason for this is explained in [numpy documentation](https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html):  "Note that the sum of the histogram values will not be equal to 1 unless bins of unity width are chosen; it is not a probability mass function.".


```python
df.a1.hist(bins=30, density=1)
```


<div align="middle">
<img src="{{site.url}}/assets/images/2019-11-13-5-lesser-known-pandas-tricks-part-2/output_29_1.png">
</div>



To calculate a PDF for a variable, we use the `weights` argument of a `hist` function.
We can observe on the plot below, that the maximum value of the y-axis is less than 1.


```python
weights = pd.np.ones_like(df.a1.values) / len(df.a1.values)
df.a1.hist(bins=30, weights=weights)
```




<div align="middle">
<img src="{{site.url}}/assets/images/2019-11-13-5-lesser-known-pandas-tricks-part-2/output_31_1.png">
</div>




### 2.3 Cumulative Distribution Function

A cumulative histogram is a mapping that counts the cumulative number of observations in all of the bins up to the specified bin.

Let's make a cumulative histogram for a1 column.
We can observe on the plot below that there are approximately 500 data points where the x is smaller or equal to 0.0.


```python
df.a1.hist(bins=30, cumulative=True)
```






<div align="middle">
<img src="{{site.url}}/assets/images/2019-11-13-5-lesser-known-pandas-tricks-part-2/output_34_1.png">
</div>



A normalized cumulative histogram is what we call Cumulative distribution function (CDF) in statistics.
The CDF is the probability that the variable takes a value less than or equal to x.
In the example below, the probability that x <= 0.0 is 0.5 and x <= 0.2 is cca. 0.98.
Note that `densitiy=1` argument works as expected with cumulative histograms.


```python
df.a1.hist(bins=30, cumulative=True, density=1)
```






<div align="middle">
<img src="{{site.url}}/assets/images/2019-11-13-5-lesser-known-pandas-tricks-part-2/output_36_1.png">
</div>




## 3. Plots for separate groups

Pandas enables us to visualize data separated by the value of the specified column.
Separating data by certain columns and observing differences in distributions is a common step in Exploratory Data Analysis.
Let's separate distributions of a1 and a2 columns by the y2 column and plot histograms.


```python
df[['a1', 'a2']].hist(by=df.y2)
```







<div align="middle">
<img src="{{site.url}}/assets/images/2019-11-13-5-lesser-known-pandas-tricks-part-2/output_39_1.png">
</div>



There is not much difference between separated distributions as the data was randomly generated. 

We can do the same for the line plot.


```python
df[['a1', 'a2']].plot(by=df.y2, subplots=True)
```



<div align="middle">
<img src="{{site.url}}/assets/images/2019-11-13-5-lesser-known-pandas-tricks-part-2/output_42_1.png">
</div>







## 4. Dummy variables

Some Machine Learning algorithms don't work with multivariate attributes, like a3 column in our example.
a3 column has 5 distinct values (0, 1, 2, 3, 4 and 5). 
To transform a multivariate attribute to multiple binary attributes, we can binarize the column, so that we get 5 attributes with 0 and 1 values.

Let's look at the example below. 
The first three rows of a3 column have value 2. 
So a3_2 attribute has the first three rows marked with 1 and all other attributes are 0.
The fourth row in a3 has a value 3, so a3_3 is 1 and all others are 0, etc.


```python
df.a3.head()
```




    0    2
    1    2
    2    2
    3    3
    4    4
    Name: a3, dtype: int64




```python
df_a4_dummy = pd.get_dummies(df.a3, prefix='a3_')
df_a4_dummy.head()
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
      <th>a3__0</th>
      <th>a3__1</th>
      <th>a3__2</th>
      <th>a3__3</th>
      <th>a3__4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



`get_dummies` function also enables us to drop the first column, so that we don't store redundant information.
Eg. when a3_1, a3_2, a3_3, a3_4 are all 0 we can assume that a3_0 should be 1 and we don't need to store it.


```python
pd.get_dummies(df.a3, prefix='a3_', drop_first=True).head()
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
      <th>a3__1</th>
      <th>a3__2</th>
      <th>a3__3</th>
      <th>a3__4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Now that we have binarized the a3 column, let's remove it from the DataFrame and add binarized attributes to it.


```python
df = df.drop('a3', axis=1)
```


```python
df = pd.concat([df, df_a4_dummy], axis=1)
df.shape
```




    (1000, 10)




```python
df.head()
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
      <th>index</th>
      <th>a1</th>
      <th>a2</th>
      <th>y1</th>
      <th>y2</th>
      <th>a3__0</th>
      <th>a3__1</th>
      <th>a3__2</th>
      <th>a3__3</th>
      <th>a3__4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.049671</td>
      <td>0.479871</td>
      <td>1.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-0.013826</td>
      <td>0.384927</td>
      <td>1.002308</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.064769</td>
      <td>0.211926</td>
      <td>1.004620</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.152303</td>
      <td>0.070613</td>
      <td>1.006939</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-0.023415</td>
      <td>0.339645</td>
      <td>1.009262</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## 5. Fitting lines

Sometimes we would like to compare a certain distribution with a linear line.
Eg. To determine if monthly sales growth is higher than linear.
When we observe that our data is linear, we can predict future values.

Pandas (with the help of numpy) enables us to fit a linear line to our data.
This is a Linear Regression algorithm in Machine Learning, which tries to make the vertical distance between the line and the data points as small as possible. 
This is called “fitting the line to the data.” 

The plot below shows the y1 column. 
Let's draw a linear line that closely matches data points of the y1 column.


```python
df.plot.scatter(x='index', y='y1', s=1)
```










<div align="middle">
<img src="{{site.url}}/assets/images/2019-11-13-5-lesser-known-pandas-tricks-part-2/output_56_1.png">
</div>

The code below calculates the least-squares solution to a linear equation. 
The output of the function that we are interested in is the least-squares solution.


```python
df['ones'] = pd.np.ones(len(df))
m, c = pd.np.linalg.lstsq(df[['index', 'ones']], df['y1'], rcond=None)[0]
```

Equation for a line is `y = m * x + c`. 
Let's use the equation and calculate the values for the line `y` that closely fits the `y1` line.


```python
df['y'] = df['index'].apply(lambda x: x * m + c)
```


```python
df[['y', 'y1']].plot()
```









<div align="middle">
<img src="{{site.url}}/assets/images/2019-11-13-5-lesser-known-pandas-tricks-part-2/output_61_1.png">
</div>

## Conclusion
Hopefully, you learned something new that will make you more productive when working on Exploratory Data Analysis. When I learned how powerful pandas is it motivated me to look for ways how can I improve my workflow. 

Which is your favorite pandas trick? Let me know in the comments below.
