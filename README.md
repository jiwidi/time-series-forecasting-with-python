# time-series-forecasting-wiki


Table of contents
=================
* 1  Time Series
* [2 Sample datasets](#sample-datasets)
* 3  Importing data
* 4  Time series analysis
  * 4.1  Seasonality
  * 4.2  Stationarity
* 5  Data Visualization and Preprocessing methods
  * 5.1  Decomposing time series
  * 5.2  Seasonality, "moving average"  
  * 5.3  Stationarity  
    * 5.3.1  ACF and PACF plots
    * 5.3.2  Plotting Rolling Statistics
  	* 5.3.3  Augmented Dickey-Fuller Test
  * 5.4  Making Time Series Stationary
	* 5.4.1  Transformations
	  * 5.4.1.1  Log Scale Transformation
	  * 5.4.1.2  Other possible transformations:
	* 5.4.2  Techniques to remove Trend - Smoothing
	  * 5.4.2.1  Moving Average
	  * 5.4.2.2  Exponentially weighted moving average:
	* 5.4.3  Further Techniques to remove Seasonality and Trend
	  * 5.4.3.1  Differencing
	  * 5.4.3.2  Decomposition
  * 5.5  Filters and noise removal
  * 5.6  Model selection
* [6  Evaluation metrics](#evaluation-metrics)
* 7  Univariate Time Series forecasting
  * 7.1  Autoregression (AR)
  * 7.1.1  Reversing the transformations
  * 7.1.2  Forecast quality scoring metrics
  * 7.2  Moving Average (MA)
  * 7.3  Autoregressive Moving Average (ARMA)
  * 7.4  Autoregressive Integrated Moving Average (ARIMA)
  * 7.5  Autoregressive Fractionally Integrated Moving Average (ARFIMA)
  * 7.6  Autoregressive conditional heteroskedasticity (ARCH)
  * 7.7  Interpreting ACF plots
	* 7.7.1  Auto ARIMA
  * 7.8  Seasonal Autoregressive Integrated Moving-Average (SARIMA)
	* 7.8.1  Auto - SARIMA
	* 7.8.2  Tuned SARIMA
  * 7.9  SARIMAX
  * 7.10  Prophet
  * 7.11  Improving Time Series Forecast models
  * 7.12  Solve a problem!
  
* 8  Multivariate Time Series Forecasting

* [9 Additional resources & literature](#additional-resources-and-literature)
  

# Sample Datasets
## Univariate datasets
*Univariate Datasets obtained [here](https://machinelearningmastery.com/time-series-datasets-for-machine-learning/)*

* [Bike Sharing Dataset Data Set](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)

## Multivariate datasets

* [Beijing air quality](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data)
* [Metro Interstate Traffic Volume Data Set](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data)

* [London bike sharing](https://www.kaggle.com/hmavrodiev/london-bike-sharing-dataset/data#_=_)

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
# Evaluation Metrics

There are many measures that can be used to analyze the performance of a prediction:

* Geometric Mean Absolute Error (GMAE)
* Geometric Mean Relative Absolute Error (GMRAE)
* Integral Normalized Root Squared Error (INRSE)
* Mean Absolute Error (MAE) / Mean Absolute Deviation (MAD)
* Mean Absolute Percentage Error (MAPE)
* Mean Absolute Scaled Error (MASE)
* Mean Arctangent Absolute Percentage Error (MAAPE)
* Mean Bounded Relative Absolute Error (MBRAE)
* Mean Directional Accuracy (MDA)
* Mean Error (ME)
* Mean Percentage Error (MPE)
* Mean Relative Absolute Error (MRAE)
* Mean Relative Error (MRE)
* Mean Squared Error (MSE)
* Median Absolute Error (MDAE)
* Median Absolute Percentage Error (MDAPE)
* Median Relative Absolute Error (MDRAE)
* Normalized Absolute Error
* Normalized Absolute Percentage Error
* Normalized Root Mean Squared Error (NRMSE)
* Relative Absolute Error (RAE)
* Root Mean Squared Error (RMSE)
* Root Mean Squared Percentage Error (RMSPE)
* Root Mean Squared Scaled Error (RMSSE)
* Root Median Squared Percentage Error (RMSPE)
* Root Relative Squared Error (RRSE)
* Symmetric Mean Absolute Percentage Error (MAPE)
* Symmetric Median Absolute Percentage Error (SMDAPE)
* Unscaled Mean Bounded Relative Absolute Error (UMBRAE)

# Additional resources and literature
## Papers

|| |
| - | - |
| Adhikari, R., & Agrawal, R. K. (2013). An introductory study on time series modeling and forecasting.|[[1]](https://arxiv.org/ftp/arxiv/papers/1302/1302.6613.pdf)|