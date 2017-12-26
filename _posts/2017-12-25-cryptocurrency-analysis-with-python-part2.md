---
layout: post
title:  "Cryptocurrency Analysis with Python - Buy and Hold"
use_math: true
categories: cryptocurrency analysis
---

In this part, I am going to analyze which coin (**Bitcoin**, **Ethereum** or **Litecoin**) was the most profitable in last two months using buy and hold strategy. 
We'll go through the analysis of these 3 cryptocurrencies and try to give an objective answer.

You can run this code by downloading this [Jupyter notebook]({{site.url}}/assets/notebooks/2017-12-25-cryptocurrency-analysis-with-python-part2.ipynb).
    
Follow me on [twitter](https://twitter.com/romanorac) to get latest updates.

<div align="middle">
<figure>
<img src="{{site.url}}/assets/images/2017-12-25-cryptocurrency-analysis-with-python-part2/bitcoin-ethereum-litecoin.png" alt="Bitcoin, Ethereum, and Litecoin">
<figcaption>Bitcoin, Ethereum, and Litecoin </figcaption>
</figure>
</div>

## Disclaimer
<span style="color:red">
**I am not a trader and this blog post is not a financial advice. This is purely introductory knowledge.
The conclusion here can be misleading as I analyze the time period with immense growth. 
Experts agree that cryptocurrencies are a [bubble](https://www.forbes.com/sites/investor/2017/09/18/cryptocurrency-is-a-bubble-whats-next/#3953479a2cb8).**
</span>

## Requirements

- [seaborn: statistical data visualization](https://seaborn.pydata.org/)

For other requirements, see my
[previous blog post](https://romanorac.github.io/cryptocurrency/analysis/2017/12/17/cryptocurrency-analysis-with-python-part1.html)
in this series.

## Getting the data

To get the latest data, go to [previous blog post]({{site.url}}/cryptocurrency/analysis/2017/12/17/cryptocurrency-analysis-with-python-part1.html),
where I described how to download it using Cryptocompare API.
You can also use the [data]({{site.url}}/assets/data/) I work with in this example.

First, we download hourly data for BTC, ETH and LTC from Coinbase exchange.
This time we work with hourly time interval as it has higher granularity.
Cryptocompare API limits response to 2000 samples, which is 2.7 months of data for each coin.


```python
import pandas as pd

def get_filename(from_symbol, to_symbol, exchange, datetime_interval, download_date):
    return '%s_%s_%s_%s_%s.csv' % (from_symbol, to_symbol, exchange, datetime_interval, download_date)

def read_dataset(filename):
    print('Reading data from %s' % filename)
    df = pd.read_csv(filename)
    df.datetime = pd.to_datetime(df.datetime) # change type from object to datetime
    df = df.set_index('datetime') 
    df = df.sort_index() # sort by datetime
    print(df.shape)
    return df
```

## Load the data


```python
df_btc = read_dataset(get_filename('BTC', 'USD', 'Coinbase', 'hour', '2017-12-24'))
df_eth = read_dataset(get_filename('ETH', 'USD', 'Coinbase', 'hour', '2017-12-24'))
df_ltc = read_dataset(get_filename('LTC', 'USD', 'Coinbase', 'hour', '2017-12-24'))
```

    Reading data from BTC_USD_Coinbase_hour_2017-12-24.csv
    (2001, 6)
    Reading data from ETH_USD_Coinbase_hour_2017-12-24.csv
    (2001, 6)
    Reading data from LTC_USD_Coinbase_hour_2017-12-24.csv
    (2001, 6)



```python
df_btc.head()
```




<div>
<table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>low</th>
      <th>high</th>
      <th>open</th>
      <th>close</th>
      <th>volumefrom</th>
      <th>volumeto</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-10-02 08:00:00</th>
      <td>4435.00</td>
      <td>4448.98</td>
      <td>4435.01</td>
      <td>4448.85</td>
      <td>85.51</td>
      <td>379813.67</td>
    </tr>
    <tr>
      <th>2017-10-02 09:00:00</th>
      <td>4448.84</td>
      <td>4470.00</td>
      <td>4448.85</td>
      <td>4464.49</td>
      <td>165.17</td>
      <td>736269.53</td>
    </tr>
    <tr>
      <th>2017-10-02 10:00:00</th>
      <td>4450.27</td>
      <td>4469.00</td>
      <td>4464.49</td>
      <td>4461.63</td>
      <td>194.95</td>
      <td>870013.62</td>
    </tr>
    <tr>
      <th>2017-10-02 11:00:00</th>
      <td>4399.00</td>
      <td>4461.63</td>
      <td>4461.63</td>
      <td>4399.51</td>
      <td>326.71</td>
      <td>1445572.02</td>
    </tr>
    <tr>
      <th>2017-10-02 12:00:00</th>
      <td>4378.22</td>
      <td>4417.91</td>
      <td>4399.51</td>
      <td>4383.00</td>
      <td>549.29</td>
      <td>2412712.73</td>
    </tr>
  </tbody>
</table>
</div>



## Extract closing prices

We are going to analyze closing prices, which are prices at which the hourly period closed. 
We merge BTC, ETH and LTC closing prices to a Dataframe to make analysis easier.


```python
df = pd.DataFrame({'BTC': df_btc.close,
                   'ETH': df_eth.close,
                   'LTC': df_ltc.close})
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
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
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



## Analysis

### Basic statistics

In 2.7 months, all three cryptocurrencies fluctuated a lot as you can observe in the table below. 

For each coin, we count the number of events and calculate mean, standard deviation, minimum, quartiles and maximum closing price. 

**Observations**
- The difference between the highest and the lowest BTC price was more than \$15000 in 2.7 months.
- The LTC surged from \\$48.61 to \$378.66 at a certain point, which is an increase of 678.98%.


```python
df.describe()
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



### Lets dive deeper into LTC

We visualize the data in the table above with a box plot. 
A box plot shows the quartiles of the dataset with points that are determined to be outliers using a method of the [inter-quartile range](https://en.wikipedia.org/wiki/Interquartile_range) (IQR). 
In other words, the IQR is the first quartile (25%) subtracted from the third quartile (75%).

On the box plot below, 
we see that LTC closing hourly price was most of the time between \\$50 and \\$100 in the last 2.7 months. 
All values over \\$150 are outliers (using IQR). Note that outliers are specific to this data sample.


```python
import seaborn as sns

ax = sns.boxplot(data=df['LTC'], orient="h")
```


<div align='middle'>
<figure>
<img src="{{site.url}}/assets/images/2017-12-25-cryptocurrency-analysis-with-python-part2/output_16_0.png" alt="Boxplot LTC">
<figcaption>LTC Boxplot with closing hourly prices </figcaption>
</figure>
</div>

#### Histogram of LTC closing price

Let's estimate the frequency distribution of LTC closing prices. 
The histogram shows the number of hours LTC had a certain value.

**Observations**
- LTC closing price was not over \$100 for many hours.
- it has right-skewed distribution because a natural limit prevents outcomes on one side.
- blue dashed line (median) shows that half of the time closing prices were under \$63.50.


```python
df['LTC'].hist(bins=30, figsize=(15,10)).axvline(df['LTC'].median(), color='b', linestyle='dashed', linewidth=2)
```

<div align='middle'>
<figure>
<img src="{{site.url}}/assets/images/2017-12-25-cryptocurrency-analysis-with-python-part2/output_19_1.png" alt="Histogram LTC">
<figcaption>Histogram for LTC with median</figcaption>
</figure>
</div>


### Visualize absolute closing prices

The chart below shows absolute closing prices. 
It is not of much use as BTC closing prices are much higher 
then prices of ETH and LTC.


```python
df.plot(grid=True, figsize=(15, 10))
```

<div align='middle'>
<figure>
<img src="{{site.url}}/assets/images/2017-12-25-cryptocurrency-analysis-with-python-part2/output_22_1.png" alt="Absolute closing price changes of BTC, ETH and LTC"> 
<figcaption>Absolute closing price changes of BTC, ETH and LTC</figcaption>
</figure>
</div>


### Visualize relative changes of closing prices

We are interested in a relative change of the price 
rather than in absolute price, so we use three different y-axis scales.

We see that closing prices move in tandem. When one coin closing price increases so do the other.


```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax1 = plt.subplots(figsize=(20, 10))
ax2 = ax1.twinx()
rspine = ax2.spines['right']
rspine.set_position(('axes', 1.15))
ax2.set_frame_on(True)
ax2.patch.set_visible(False)
fig.subplots_adjust(right=0.7)

df['BTC'].plot(ax=ax1, style='b-')
df['ETH'].plot(ax=ax1, style='r-', secondary_y=True)
df['LTC'].plot(ax=ax2, style='g-')

# legend
ax2.legend([ax1.get_lines()[0],
            ax1.right_ax.get_lines()[0],
            ax2.get_lines()[0]],
           ['BTC', 'ETH', 'LTC'])
```

<div align='middle'>
<figure>
<img src="{{site.url}}/assets/images/2017-12-25-cryptocurrency-analysis-with-python-part2/output_24_1.png" alt="Relative closing price changes of BTC, ETH and LTC"> 
<figcaption>Relative closing price changes of BTC, ETH and LTC</figcaption>
</figure>
</div>


### Measure correlation of closing prices 

We calculate [Pearson correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
between closing prices of BTC, ETH and LTC. 
Pearson correlation is a measure of the linear correlation between two variables X and Y. 
It has a value between +1 and −1, where 1 is total positive linear correlation, 0 is no linear correlation, and −1 is total negative linear correlation.
Correlation matrix is symmetric so we only show the lower half. 

[Sifr Data](https://www.sifrdata.com/cryptocurrency-correlation-matrix/) daily updates Pearson correlations for many cryptocurrencies.

**Observations**
- BTC, ETH and LTC were highly correlated in past 2 months. This means, when BTC closing price increased, ETH and LTC followed. 
- ETH and LTC were even more correlated with 0.9565 Pearson correlation coefficient.


```python
import seaborn as sns
import matplotlib.pyplot as plt

# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(10, 10))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, annot=True, fmt = '.4f', mask=mask, center=0, square=True, linewidths=.5)
```

<div align='middle'>
<figure>
<img src="{{site.url}}/assets/images/2017-12-25-cryptocurrency-analysis-with-python-part2/output_26_1.png" alt="Pearson Corelation BTC, ETH and LTC"> 
<figcaption>Pearson Correlation for BTC, ETH and LTC</figcaption>
</figure>
</div>

## Buy and hold strategy

[Buy and hold](https://www.investopedia.com/terms/b/buyandhold.asp) is a passive investment strategy in which an investor buys a cryptocurrency and holds it for a long period of time, regardless of fluctuations in the market. 

Let's analyze returns using buy and hold strategy for past 2.7 months. 
We calculate the return percentage, where $t$ represents a certain time period and $price_0$ is initial closing price:

$$
return_{t, 0} = \frac{price_t}{price_0}
$$


```python
df_return = df.apply(lambda x: x / x[0])
df_return.head()
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
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-10-02 08:00:00</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2017-10-02 09:00:00</th>
      <td>1.003516</td>
      <td>1.001560</td>
      <td>1.001279</td>
    </tr>
    <tr>
      <th>2017-10-02 10:00:00</th>
      <td>1.002873</td>
      <td>1.001925</td>
      <td>0.998355</td>
    </tr>
    <tr>
      <th>2017-10-02 11:00:00</th>
      <td>0.988909</td>
      <td>0.995520</td>
      <td>0.987025</td>
    </tr>
    <tr>
      <th>2017-10-02 12:00:00</th>
      <td>0.985198</td>
      <td>0.987192</td>
      <td>0.981542</td>
    </tr>
  </tbody>
</table>
</div>



### Visualize returns

We show that LTC was the most profitable for time period between October 2, 2017 and December 24, 2017.


```python
df_return.plot(grid=True, figsize=(15, 10)).axhline(y = 1, color = "black", lw = 2)
```

<div align='middle'>
<figure>
<img src="{{site.url}}/assets/images/2017-12-25-cryptocurrency-analysis-with-python-part2/output_30_1.png" alt="Visualize returns"> 
<figcaption>Factor of returns for BTC, ETH and LTC in 2.7 months</figcaption>
</figure>
</div>


## Conclusion

The cryptocurrencies we analyzed fluctuated a lot but all gained in a given 2.7 months period.

### What is the percentage increase?


```python
df_perc = df_return.tail(1) * 100
ax = sns.barplot(data=df_perc)
df_perc
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
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-12-24 16:00:00</th>
      <td>314.688065</td>
      <td>226.900488</td>
      <td>501.407164</td>
    </tr>
  </tbody>
</table>
</div>

<div align='middle'>
<figure>
<img src="{{site.url}}/assets/images/2017-12-25-cryptocurrency-analysis-with-python-part2/output_33_1.png" alt="What is the percentage increase?"> 
<figcaption>The percentage increase for BTC, ETH and LTC in 2.7 months</figcaption>
</figure>
</div>


### How many coins could we bought for \$1000?


```python
budget = 1000 # USD
df_coins = budget/df.head(1)

ax = sns.barplot(data=df_coins)
df_coins
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
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-10-02 08:00:00</th>
      <td>0.224777</td>
      <td>3.31818</td>
      <td>18.274854</td>
    </tr>
  </tbody>
</table>
</div>




<div align='middle'>
<figure>
<img src="{{site.url}}/assets/images/2017-12-25-cryptocurrency-analysis-with-python-part2/output_35_1.png" alt="How many coins could we bought for \$1000?"> 
<figcaption>The amount of coins we could buy with \$1000 a while ago</figcaption>
</figure>
</div>

### How much money would we make?


```python
df_profit = df_return.tail(1) * budget

ax = sns.barplot(data=df_profit)
df_profit
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
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-12-24 16:00:00</th>
      <td>3146.880655</td>
      <td>2269.004878</td>
      <td>5014.071637</td>
    </tr>
  </tbody>
</table>
</div>


<div align='middle'>
<figure>
<img src="{{site.url}}/assets/images/2017-12-25-cryptocurrency-analysis-with-python-part2/output_37_1.png" alt="How much money would we make?">
<figcaption>The amount of money we would make if we invested \$1000 a while ago</figcaption>
</figure>
</div>

