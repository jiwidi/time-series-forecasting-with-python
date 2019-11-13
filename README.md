# time-series-forecasting-wiki



* 1  Time Series
* 2  Sample datasets
* 4  Importing data
* 3  Time series analysis
  * 5.1  Stationality
  * 5.1  Seasonality 
  * 5.1  Autocorrelation
* 5  Data Visualization and Preprocessing methods
  * 5.1  Decomposing time series
  * 5.1  Seasonality, "moving average"  
  * 5.1  Stationarity  
    * 5.1.1  ACF and PACF plots
    * 5.1.2  Plotting Rolling Statistics
	* 5.1.3  Augmented Dickey-Fuller Test
  * 5.2  Making Time Series Stationary
	* 5.2.1  Transformations
	  * 5.2.1.1  Log Scale Transformation
	  * 5.2.1.2  Other possible transformations:
	* 5.2.2  Techniques to remove Trend - Smoothing
	  * 5.2.2.1  Moving Average
	  * 5.2.2.2  Exponentially weighted moving average:
	* 5.2.3  Further Techniques to remove Seasonality and Trend
	  * 5.2.3.1  Differencing
	  * 5.2.3.2  Decomposition
  * 5.3  Filters and noise removal
  * 5.4  Evaluation metrics
* 6  Univariate Time Series forecasting
  * 6.1  Autoregression (AR)
  * 6.1.1  Reversing the transformations
  * 6.1.2  Forecast quality scoring metrics
  * 6.2  Moving Average (MA)
  * 6.3  Autoregressive Moving Average (ARMA)
  * 6.4  Autoregressive Integrated Moving Average (ARIMA)
  * 6.5  Interpreting ACF plots
	* 6.5.1  Auto ARIMA
  * 6.6  Seasonal Autoregressive Integrated Moving-Average (SARIMA)
	* 6.6.1  Auto - SARIMA
	* 6.6.2  Tuned SARIMA
  * 6.7  SARIMAX
  * 6.8  Prophet
  * 6.9  Improving Time Series Forecast models
  * 6.10  Solve a problem!
  
* 7  Multivariate Time Series Forecasting

# Sample Datasets
## Univariate datasets
*Univariate Datasets obtained [here](https://machinelearningmastery.com/time-series-datasets-for-machine-learning/)*

* Bike Sharing Dataset Data Set [link](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)

## Multivariate datasets

* Beijing air quaility [link](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data)
* Metro Interstate Traffic Volume Data Set [link](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data)

* London bike sharing [link](https://www.kaggle.com/hmavrodiev/london-bike-sharing-dataset/data#_=_)

We will use the bike sharing dataset at London. This dataset contains the number of bikes shared each day and a set of features on the weather conditions. This is a multivariate dataset but we will also use it for the univariate analysis. For each case we will use a selection of features:

* Univariate
  * "timestamp" - timestamp field for grouping the data
  * "cnt" - the count of a new bike shares
  
* Multivariate 
  * "timestamp" - timestamp field for grouping the data
  * "cnt" - the count of a new bike shares
  * "t1" - real temperature in C
  * "t2" - temperature in C "feels like"
  * "hum" - humidity in percentage
  * "wind_speed" - wind speed in km/h
  * "weather_code" - category of the weather
  * "is_holiday" - boolean field - 1 holiday / 0 non holiday
  * "is_weekend" - boolean field - 1 if the day is weekend
  * "season" - category field meteorological seasons: 0-spring ; 1-summer; 2-fall; 3-winter.


