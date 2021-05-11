# :hourglass_flowing_sand: time-series-forecasting-wiki
This repository contains a series of analysis, transforms and forecasting models frequently used when dealing with time series. The aim of this repository is to showcase how to model time series from the scratch, for this we are using a real usecase dataset ([Beijing air polution dataset](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data) to avoid perfect use cases far from reality that are often present in this types of tutorials. If you want to rerun the notebooks make sure you install al neccesary dependencies, [Guide](docs/setup.md)

<img src="results/beijing.jpg">


You can find the more detailed toc on the main [notebook](time-series-forecasting-tutorial.ipynb) 



# :open_file_folder: Dataset

The dataset used is the [Beijing air quality](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data) public dataset. This dataset contains polution data from 2014 to 2019 sampled every 10 minutes along with extra weather features such as preassure, temperature etc. We decided to resample the dataset with daily frequency for both easier data handling and proximity to a real use case scenario (no one would build a model to predict polution 10 minutes ahead, 1 day ahead looks more realistic). In this case the series is already stationary with some small seasonalities which change every year #MORE ONTHIS

In order to obtain a exact copy of the dataset used in this tutorial please run the [script](https://github.com/jiwidi/time-series-forecasting-wiki/blob/master/datasets/download_datasets.py) under `datasets/download_datasets.py` which will automatically download the dataset and preprocess it for you.

#  📚 Analysis and transforms

* Time series decomposition
  * Level
  * Trend
  * Seasonality 
  * Noise
  
* Stationarity
  * AC and PAC plots
  * Rolling mean and std
  * Dickey-Fuller test
  
* Making our time series stationary
  * Difference transform
  * Log scale
  * Smoothing
  * Moving average

# :triangular_ruler: Models tested

* Autoregression ([AR](https://www.statsmodels.org/stable/generated/statsmodels.tsa.ar_model.AR.html))
* Moving Average (MA)
* Autoregressive Moving Average (ARMA)
* Autoregressive integraded moving average (ARIMA)
* Seasonal autoregressive integrated moving average (SARIMA)
* Bayesian regression [Link](https://scikit-learn.org/stable/auto_examples/linear_model/plot_bayesian_ridge.html)
* Lasso [Link](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
* SVM [Link](https://scikit-learn.org/stable/modules/classes.html?highlight=svm#module-sklearn.svm)
* Randomforest [Link](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=randomforest#sklearn.ensemble.RandomForestRegressor)
* Nearest neighbors [Link](https://scikit-learn.org/stable/modules/neighbors.html)
* XGBoost [Link](https://xgboost.readthedocs.io/en/latest/)
* Lightgbm [Link](https://github.com/microsoft/LightGBM)
* Prophet [Link](https://facebook.github.io/prophet/docs/quick_start.html)
* Long short-term memory with tensorflow (LSTM)[Link](https://www.tensorflow.org/)

* DeepAR


# :mag: Forecasting results
We will devide our results wether the extra features columns such as temperature or preassure were used by the model as this is a huge step in metrics and represents two different scenarios. Metrics used were:

## Evaluation Metrics
* Mean Absolute Error (MAE) 
* Mean Absolute Percentage Error (MAPE)
* Root Mean Squared Error (RMSE)
* Coefficient of determination (R2)


<table class="table table-bordered table-hover table-condensed">
<thead><tr><th title="Field #1">Model</th>
<th title="Field #2">mae</th>
<th title="Field #3">rmse</th>
<th title="Field #4">mape</th>
<th title="Field #5">r2</th>
</tr></thead>
<tbody><tr>
<td>EnsembleXG+TF</td>
<td>27.636715371194235</td>
<td>40.22713163966418</td>
<td>0.417235373186213</td>
<td>0.7560414156259503</td>
</tr>
<tr>
<td>EnsembleLIGHT+TF</td>
<td>27.343883702452075</td>
<td>39.2701143554976</td>
<td>0.4165273453614097</td>
<td>0.7675110569347134</td>
</tr>
<tr>
<td>EnsembleXG+LIGHT+TF</td>
<td>27.634858862129317</td>
<td>39.685112028877015</td>
<td>0.4398299604512803</td>
<td>0.7625713120947151</td>
</tr>
<tr>
<td>EnsembleXG+LIGHT</td>
<td>29.95312427473199</td>
<td>42.70452481595056</td>
<td>0.5232424810249069</td>
<td>0.7250677041717724</td>
</tr>
<tr>
<td>Randomforest tunned</td>
<td>40.79113861034052</td>
<td>53.19804047844643</td>
<td>0.9032153211034835</td>
<td>0.5733524634338175</td>
</tr>
<tr>
<td>SVM RBF GRID SEARCH</td>
<td>38.56562478061001</td>
<td>50.34006040681473</td>
<td>0.7767794489140278</td>
<td>0.6179629913034281</td>
</tr>
<tr>
<td>DeepAR</td>
<td>71.37152301795753</td>
<td>103.96899244382487</td>
<td>0.9640163628258902</td>
<td>-0.6296173573772541</td>
</tr>
<tr>
<td>Tensorflow simple LSTM</td>
<td>30.131065860622638</td>
<td>43.075867339114566</td>
<td>0.4159828417964767</td>
<td>0.7202654993160147</td>
</tr>
<tr>
<td>Prophet multivariate</td>
<td>38.24990881198136</td>
<td>50.445918809437885</td>
<td>0.7416946500170641</td>
<td>0.6163545566281877</td>
</tr>
<tr>
<td>Kneighbors</td>
<td>57.04884684382101</td>
<td>80.38733591079851</td>
<td>1.0829364988041466</td>
<td>0.025788518944025896</td>
</tr>
<tr>
<td>SVM RBF</td>
<td>40.808894441967176</td>
<td>56.03280030028863</td>
<td>0.7942235072980369</td>
<td>0.5266715316287722</td>
</tr>
<tr>
<td>Lightgbm</td>
<td>30.208044846890726</td>
<td>42.75762737887313</td>
<td>0.5228233367915872</td>
<td>0.7243835290488294</td>
</tr>
<tr>
<td>XGBoost</td>
<td>32.132726550323085</td>
<td>45.587756858787024</td>
<td>0.5588847740800054</td>
<td>0.6866898836748925</td>
</tr>
<tr>
<td>Randomforest</td>
<td>45.83794230098986</td>
<td>59.448943203671895</td>
<td>1.0292758595380396</td>
<td>0.46719750636098334</td>
</tr>
<tr>
<td>Lasso</td>
<td>39.23696633914533</td>
<td>54.583997908701555</td>
<td>0.7090311778288818</td>
<td>0.5508321591036289</td>
</tr>
<tr>
<td>BayesianRidge</td>
<td>39.24300096256203</td>
<td>54.63447702907304</td>
<td>0.7078739844522808</td>
<td>0.5500009967624746</td>
</tr>
<tr>
<td>Prophet univariate</td>
<td>61.32766979896922</td>
<td>83.63611560092615</td>
<td>1.255915591523897</td>
<td>-0.05454636657481493</td>
</tr>
<tr>
<td>AutoSARIMAX (1, 0, 1),(0, 0, 0, 6)</td>
<td>51.2919825059347</td>
<td>71.48683847305887</td>
<td>0.9125628930440329</td>
<td>0.2295753804112044</td>
</tr>
<tr>
<td>SARIMAX</td>
<td>51.25048186072873</td>
<td>71.32864345356596</td>
<td>0.9052778110315475</td>
<td>0.2329813915764456</td>
</tr>
<tr>
<td>AutoARIMA (0, 0, 3)</td>
<td>47.01056723044805</td>
<td>64.7122976566249</td>
<td>1.0022566314516306</td>
<td>0.3686770004286427</td>
</tr>
<tr>
<td>ARIMA</td>
<td>48.24923372419579</td>
<td>66.38753923351497</td>
<td>1.0616715242063703</td>
<td>0.3355671235683002</td>
</tr>
<tr>
<td>ARMA</td>
<td>47.09683864535579</td>
<td>64.86169188736405</td>
<td>1.0056439887364004</td>
<td>0.3657587024902095</td>
</tr>
<tr>
<td>MA</td>
<td>49.0438818653784</td>
<td>66.20166785625156</td>
<td>1.0528694833926353</td>
<td>0.3392824644097636</td>
</tr>
<tr>
<td>AR</td>
<td>47.2380490738903</td>
<td>65.3217182921559</td>
<td>1.015593381744262</td>
<td>0.3567301864236727</td>
</tr>
<tr>
<td>HWES</td>
<td>52.96026256576073</td>
<td>74.67173007275758</td>
<td>1.1126266238341962</td>
<td>0.1593980396294502</td>
</tr>
<tr>
<td>SES</td>
<td>52.96026256576073</td>
<td>74.67173007275758</td>
<td>1.1126266238341962</td>
<td>0.1593980396294502</td>
</tr>
<tr>
<td>Yesterdays value</td>
<td>52.67495069033531</td>
<td>74.52276372084813</td>
<td>1.044049705124814</td>
<td>0.1627486115825163</td>
</tr>
<tr>
<td>Naive mean</td>
<td>59.32093968874532</td>
<td>81.44435990224613</td>
<td>1.3213573049116003</td>
<td>0.0</td>
</tr>
</tbody></table>

 

# :shipit: Additional resources and literature

## Models not tested but that are gaining popularity 
There are several models we have not tried in this tutorials as they come from the academic world and their implementation is not 100% reliable, but is worth mentioning them:

* Neural basis expansion analysis for interpretable time series forecasting (N-BEATS) | [link](https://arxiv.org/abs/1905.10437) [Code](https://github.com/philipperemy/n-beats)
* ESRRN [link](https://eng.uber.com/m4-forecasting-competition/)  [Code](https://github.com/damitkwr/ESRNN-GPU)


#
| | |
| - | - |
| Adhikari, R., & Agrawal, R. K. (2013). An introductory study on time series modeling and forecasting | [[1]](https://arxiv.org/ftp/arxiv/papers/1302/1302.6613.pdf)|
| Introduction to Time Series Forecasting With Python | [[2]](https://machinelearningmastery.com/introduction-to-time-series-forecasting-with-python/)|
| Deep Learning for Time Series Forecasting | [[3]](https://machinelearningmastery.com/deep-learning-for-time-series-forecasting/ )
| The Complete Guide to Time Series Analysis and Forecasting| [[4]](https://towardsdatascience.com/the-complete-guide-to-time-series-analysis-and-forecasting-70d476bfe775)| 
| How to Decompose Time Series Data into Trend and Seasonality| [[5]](https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/)


# Contributing
Want to see another model tested? Do you have anything to add or fix? I'll be happy to talk about it! Open an issue/PR :) 

