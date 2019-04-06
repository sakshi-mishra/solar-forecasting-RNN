# Multi-time Horizon Solar Forecasting using Recurrent Neural Networks

This repository contains code to reproduce the results published in the ["Multi-time-horizon Solar Forecasting Using Recurrent Neural Network"](https://arxiv.org/abs/1807.05459) paper. In addition, a LSTM implementation for multi-time horizon solar forecasting is available at this repository: ["PyTorch implementation of LSTM Model for Multi-time-horizon Solar Forecasting"](https://github.com/sakshi-mishra/LSTM_Solar_Forecasting).

## Conda environment for running the code

A conda environment file is provided for convenience. Assuming you have Anaconda python distribution available on your computer, you can create a new conda environment with the necessary packages using the following command:

`conda env create -f multi-tscale-slim.yaml -n "multi_time_horizon"`

## Predictions with fixed time horizon 
The Jupyter Notebooks in [fixed-time-horizon-prediction](fixed-time-horizon-prediction) explain the experiments on forecasting solar on fixed time interval basis as described in Section V.A of the [paper](https://arxiv.org/abs/1807.05459). 

  There are seven different sites for which the prediction is being done, thus seven ipython notebooks for each site (1-hour ahead forecast).
  * [fixed-time-horizon-prediction/Exp_1_RNN_Bondville.ipynb](fixed-time-horizon-prediction/Exp_1_RNN_Bondville.ipynb) : Code to train and predict for Bondville location, 1-hour ahead forecast
  * [fixed-time-horizon-prediction/Exp_1_RNN_Boulder.ipynb](fixed-time-horizon-prediction/Exp_1_RNN_Boulder.ipynb): Code to train and predict for Boulder location, 1- hour ahead forecast
  * [fixed-time-horizon-prediction/Exp_1_RNN_Desert_Rock.ipynb](fixed-time-horizon-prediction/Exp_1_RNN_Desert_Rock.ipynb): Code to train and predict for Desert Rock location, 1- hour ahead forecast
  * [fixed-time-horizon-prediction/Exp_1_RNN_Fort_Peck.ipynb](fixed-time-horizon-prediction/Exp_1_RNN_Fort_Peck.ipynb): Code to train and predict for Fore Peck location, 1- hour ahead forecast
  * [fixed-time-horizon-prediction/Exp_1_RNN_Goodwin_Creek.ipynb](fixed-time-horizon-prediction/Exp_1_RNN_Goodwin_Creek.ipynb): Code to train and predict for Goodwin Creek location, 1- hour ahead forecast
  * [fixed-time-horizon-prediction/Exp_1_RNN_Penn_State.ipynb](fixed-time-horizon-prediction/Exp_1_RNN_Penn_State.ipynb): code to train and predict for Penn State location 1-hour ahead forecast
  * [fixed-time-horizon-prediction/Exp_1_RNN_Sioux_Falls.ipynb](fixed-time-horizon-prediction/Exp_1_RNN_Sioux_Falls.ipynb): code to train and predict for Sioux Falls location 1-hour ahead forecast

2-hour ahead forecast Jupyter Notebooks for all seven locations:
  * [fixed-time-horizon-prediction/Exp_1.2_Bondville_2hour.ipynb](fixed-time-horizon-prediction/Exp_1.2_Bondville_2hour.ipynb): code to train and predict for Bondville location 2-hour ahead forecast
  * [fixed-time-horizon-prediction/Exp_1.2_Boulder_2hour.ipynb](fixed-time-horizon-prediction/Exp_1.2_Boulder_2hour.ipynb): code to train and predict for Boulder location 2-hour ahead forecast
  *  [fixed-time-horizon-prediction/Exp_1.2_Desert_Rock_2hour.ipynb](fixed-time-horizon-prediction/Exp_1.2_Desert_Rock_2hour.ipynb) : code to train and predict for Desert Rock location 2-hour ahead forecast
  *  [fixed-time-horizon-prediction/Exp_1.2_FortPeck_2hour.ipynb](fixed-time-horizon-prediction/Exp_1.2_FortPeck_2hour.ipynb): code to train and predict for Fort Peck location 2-hour ahead forecast
  *  [fixed-time-horizon-prediction/Exp_1.2_GoodwinCreek_2hour.ipynb](fixed-time-horizon-prediction/Exp_1.2_GoodwinCreek_2hour.ipynb): code to train and predict for Goodwin Creek location 2-hour ahead forecast
  *  [fixed-time-horizon-prediction/Exp_1.2_PenState_2hour.ipynb](fixed-time-horizon-prediction/Exp_1.2_PenState_2hour.ipynb): code to train and predict for Penn State location 2-hour ahead forecast
  *  [fixed-time-horizon-prediction/Exp_1.2_SiouxFalls_2hour.ipynb](fixed-time-horizon-prediction/Exp_1.2_SiouxFalls_2hour.ipynb): code to train and predict for Sioux Falls location 2-hour ahead forecast

3-hour ahead forecast Jupyter Notebook for Bondville location:
  *  [fixed-time-horizon-prediction/Exp_1.3_Bondville_3hour.ipynb](fixed-time-horizon-prediction/Exp_1.3_Bondville_3hour.ipynb): code to train and predict for Bondville location 3-hour ahead forecast

#### [fixed-time-horizon-prediction/Exp_1.2](fixed-time-horizon-prediction/Exp_1.2) folder contains the .py version of the Jupyter Notebooks listed above, along with additional .py files predicting 3-hour ahead and 4-ahead forecasts for all the seven locations.

## Predictions with multi-time horizon

The python scripts in [multi-time-horizon-prediction](multi-time-horizon-prediction) explain the experiments on forecasting solar on multi-time-horizon basis as described in Section V.B of the [paper](https://arxiv.org/abs/1807.05459).

The models are trained on year 2010 and 2011. The predictions are done for year 2009, 2015, 2016 and 2017 for all seven locations. There are four test years and seven locations, so there are total 4*7 .py files for training and prediction purpose.

The Jupyter Notebook [multi-time-horizon-prediction/Exp_2.1_multi-time-scale_All_Locations.ipynb](multi-time-horizon-prediction/Exp_2.1_multi-time-scale_All_Locations.ipynb) contains the code for predicting the solar irradiance for all the locations as well as all the test years (2009, 2015, 2016 and 2017).

### Training/Testing Data

The training and testing data needs to be downloaded from the [NOAA FTP server](ftp://aftp.cmdl.noaa.gov/data/radiation/surfrad/) for the locations/sites. You can use GNU wget to automate the download process. The scripts assume that the data is in the *data* folder as per the structure outlined in the [data_dir_struct.txt](data_dir_struct.txt) file.

If you face any issues running the code or reproducing the results, create an issue on this repo. Contributions are welcome too :)

## Citing
If you find this work useful for your research, please cite the paper:

```bibtex
@misc{1807.05459,
Author = {Sakshi Mishra and Praveen Palanisamy},
Title = {Multi-time-horizon Solar Forecasting Using Recurrent Neural Network},
Year = {2018},
Eprint = {arXiv:1807.05459},
}
```
