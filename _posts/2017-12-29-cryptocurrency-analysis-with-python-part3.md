---
layout: post
title:  "Cryptocurrency Analysis with Python - Log Returns"
use_math: true
categories: cryptocurrency analysis
---

In [previous post]({{site.url}}/cryptocurrency/analysis/2017/12/25/cryptocurrency-analysis-with-python-part2.html), we analyzed raw price changes of cryptocurrencies. The problem with that approach is that prices of different cryptocurrencies are not normalized and we cannot use comparable metrics. 

In this post, we describe benefits of using log returns for analysis of price changes. You can download this [Jupyter Notebook]({{site.url}}/assets/notebooks/2017-12-29-cryptocurrency-analysis-with-python-part3.ipynb) and the [data](https://github.com/romanorac/romanorac.github.io/tree/master/assets/data). 

Follow me on [twitter](https://twitter.com/romanorac) to get latest updates.

<div align="middle">
<figure>
<img src="{{site.url}}/assets/images/2017-12-29-cryptocurrency-analysis-with-python-part3/log_returns.png" alt="Bitcoin, Ethereum, and Litecoin Log Returns">
<figcaption>Bitcoin, Ethereum, and Litecoin Log Returns</figcaption>
</figure>
</div>

## Disclaimer
<span style="color:red">
**I am not a trader and this blog post is not a financial advice. This is purely introductory knowledge.
The conclusion here can be misleading as we analyze the time period with immense growth.**
</span>

## Requirements

- [SciPy - scientific and numerical tools for Python](https://www.scipy.org/)

For other requirements, see my
[first blog post]({{site.url}}/cryptocurrency/analysis/2017/12/17/cryptocurrency-analysis-with-python-part1.html)
of this series.

## Load the data


```python
import pandas as pd

df_btc = pd.read_csv('BTC_USD_Coinbase_hour_2017-12-24.csv', index_col='datetime')
df_eth = pd.read_csv('ETH_USD_Coinbase_hour_2017-12-24.csv', index_col='datetime')
df_ltc = pd.read_csv('LTC_USD_Coinbase_hour_2017-12-24.csv', index_col='datetime')
```


```python
df = pd.DataFrame({'BTC': df_btc.close,
                   'ETH': df_eth.close,
                   'LTC': df_ltc.close})
df.index = df.index.map(pd.to_datetime)
df = df.sort_index()
```


```python
df.head()
```




<div>
<table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BTC</th>
      <th>ETH</th>
      <th>LTC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-10-02 08:00:00</th>
      <td>4448.85</td>
      <td>301.37</td>
      <td>54.72</td>
    </tr>
    <tr>
      <th>2017-10-02 09:00:00</th>
      <td>4464.49</td>
      <td>301.84</td>
      <td>54.79</td>
    </tr>
    <tr>
      <th>2017-10-02 10:00:00</th>
      <td>4461.63</td>
      <td>301.95</td>
      <td>54.63</td>
    </tr>
    <tr>
      <th>2017-10-02 11:00:00</th>
      <td>4399.51</td>
      <td>300.02</td>
      <td>54.01</td>
    </tr>
    <tr>
      <th>2017-10-02 12:00:00</th>
      <td>4383.00</td>
      <td>297.51</td>
      <td>53.71</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```




<div>
<table border="1" >
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BTC</th>
      <th>ETH</th>
      <th>LTC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2001.000000</td>
      <td>2001.000000</td>
      <td>2001.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9060.256122</td>
      <td>407.263793</td>
      <td>106.790100</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4404.269591</td>
      <td>149.480416</td>
      <td>89.142241</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4150.020000</td>
      <td>277.810000</td>
      <td>48.610000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5751.020000</td>
      <td>301.510000</td>
      <td>55.580000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7319.950000</td>
      <td>330.800000</td>
      <td>63.550000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>11305.000000</td>
      <td>464.390000</td>
      <td>100.050000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>19847.110000</td>
      <td>858.900000</td>
      <td>378.660000</td>
    </tr>
  </tbody>
</table>
</div>



## Why Log Returns?

Benefit of using returns, versus prices, is normalization: measuring all variables in a comparable metric, thus enabling evaluation of analytic relationships amongst two or more variables despite originating from price series of unequal values (for details, see [Why Log Returns](https://quantivity.wordpress.com/2011/02/21/why-log-returns/)).


Let's define return as:

$$
r_{i} = \frac{p_i - p_j}{p_j},
$$

where $r_i$ is return at time $i$, $p_i$ is the price at time $i$ and $j = i-1$.


### Calculate Log Returns

Author of [Why Log Returns](https://quantivity.wordpress.com/2011/02/21/why-log-returns/)
outlines several benefits of using log returns instead of returns so we transform **returns** equation to **log returns** equation:

$$
r_{i} = \frac{p_i - p_j}{p_j}
$$

$$
r_i = \frac{p_i}{p_j} - \frac{p_j}{p_j}
$$

$$
1 + r_i = \frac{p_i}{p_j}
$$


$$
log(1+r_i) = log(\frac{p_i}{p_j})
$$

$$
log(1+r_i) = log(p_i) - log(p_j)
$$

Now, we apply the log returns equation to closing prices of cryptocurrencies:


```python
import numpy as np

# shift moves dates back by 1
df_change = df.apply(lambda x: np.log(x) - np.log(x.shift(1))) 
```


```python
df_change.head()
```

<div>
<table border="1" >
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BTC</th>
      <th>ETH</th>
      <th>LTC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-10-02 08:00:00</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2017-10-02 09:00:00</th>
      <td>0.003509</td>
      <td>0.001558</td>
      <td>0.001278</td>
    </tr>
    <tr>
      <th>2017-10-02 10:00:00</th>
      <td>-0.000641</td>
      <td>0.000364</td>
      <td>-0.002925</td>
    </tr>
    <tr>
      <th>2017-10-02 11:00:00</th>
      <td>-0.014021</td>
      <td>-0.006412</td>
      <td>-0.011414</td>
    </tr>
    <tr>
      <th>2017-10-02 12:00:00</th>
      <td>-0.003760</td>
      <td>-0.008401</td>
      <td>-0.005570</td>
    </tr>
  </tbody>
</table>
</div>



### Visualize Log Returns

We plot normalized changes of closing prices for last 50 hours. Log differences can be interpreted as the percentage change.


```python
df_change[:50].plot(figsize=(15, 10)).axhline(color='black', linewidth=2)
```

<div align="middle">
<figure>
<img src="{{site.url}}/assets/images/2017-12-29-cryptocurrency-analysis-with-python-part3/output_13_1.png" alt="Bitcoin, Ethereum, and Litecoin Log Returns for last 50 hours">
<figcaption>Bitcoin, Ethereum, and Litecoin Log Returns for last 50 hours</figcaption>
</figure>
</div>

### Are LTC prices distributed log-normally?

If we assume that prices are distributed log-normally, 
then $log(1 + r_i)$ is conveniently normally distributed 
(for details, see [Why Log Returns](https://quantivity.wordpress.com/2011/02/21/why-log-returns/))

On the chart below, we plot the distribution of LTC hourly closing prices.
We also estimate parameters for log-normal distribution and plot estimated log-normal distribution with a red line.


```python
from scipy.stats import lognorm
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

values = df['LTC']

shape, loc, scale = stats.lognorm.fit(values) 
x = np.linspace(values.min(), values.max(), len(values))
pdf = stats.lognorm.pdf(x, shape, loc=loc, scale=scale) 
label = 'mean=%.4f, std=%.4f, shape=%.4f' % (loc, scale, shape)

ax.hist(values, bins=30, normed=True)
ax.plot(x, pdf, 'r-', lw=2, label=label)
ax.legend(loc='best')
```

<div align="middle">
<figure>
<img src="{{site.url}}/assets/images/2017-12-29-cryptocurrency-analysis-with-python-part3/output_15_1.png" alt="Distribution of LTC prices">
<figcaption>Distribution of LTC prices</figcaption>
</figure>
</div>


### Are LTC log returns normally distributed?

On the chart below, we plot the distribution of LTC log returns.
We also estimate parameters for normal distribution and plot estimated normal distribution with a red line.


```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

values = df_change['LTC'][1:]  # skip first NA value
x = np.linspace(values.min(), values.max(), len(values))

loc, scale = stats.norm.fit(values)
param_density = stats.norm.pdf(x, loc=loc, scale=scale)
label = 'mean=%.4f, std=%.4f' % (loc, scale)

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(values, bins=30, normed=True)
ax.plot(x, param_density, 'r-', label=label)
ax.legend(loc='best')
```
<div align="middle">
<figure>
<img src="{{site.url}}/assets/images/2017-12-29-cryptocurrency-analysis-with-python-part3/output_17_1.png" alt="Distribution of LTC Log Returns">
<figcaption>Distribution of LTC Log Returns</figcaption>
</figure>
</div>


### Pearson Correlation with log returns

We calculate Pearson Correlation from log returns.
The correlation matrix below has similar values as the one at [Sifr Data](https://www.sifrdata.com/cryptocurrency-correlation-matrix/). 
There are differences because:
 - we don't calculate [volume-weighted average daily prices](https://www.investopedia.com/terms/v/vwap.asp)
 - different time period (hourly and daily),
 - different data source (Coinbase and Poloniex).
 
**Observations**
- BTC and ETH have moderate positive relationship,
- LTC and ETH have strong positive relationship.


```python
import seaborn as sns
import matplotlib.pyplot as plt

# Compute the correlation matrix
corr = df_change.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(10, 10))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, annot=True, fmt = '.4f', mask=mask, center=0, square=True, linewidths=.5)
```

<div align="middle">
<figure>
<img src="{{site.url}}/assets/images/2017-12-29-cryptocurrency-analysis-with-python-part3/output_19_1.png" alt="Correlation matrix with BTC, ETH and LTC">
<figcaption>Correlation matrix with BTC, ETH and LTC </figcaption>
</figure>
</div>

## Conclusion

We showed how to calculate log returns from raw prices with a practical example. 
This way we normalized prices, which simplifies further analysis. 
We also showed how to estimate parameters for normal and log-normal distributions.
