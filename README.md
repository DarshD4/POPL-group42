# ARIMA Forecasting and Pyro Models

## Problem Statement

The project focuses on time series forecasting for gasoline prices, employing a comprehensive approach through ARIMA (AutoRegressive Integrated Moving Average) modeling, standard linear regression, and Pyro-based probabilistic forecasting. The primary aim is to capture seasonality patterns inherent in the data, offering insights into the temporal fluctuations of gasoline prices. The code uniquely addresses the intricate challenge of time series forecasting for gasoline prices, standing out by incorporating a probabilistic programming approach through Pyro. Beyond seasonality patterns, the goal is to provide a nuanced understanding of uncertainty in predictions. Probabilistic programming, a key aspect of POPL, enhances the forecasting models, making them more adaptable to real-world uncertainties compared to traditional methods. Additionally, backtesting methodologies are employed to assess model performance. Challenges involve fine-tuning model parameters, mastering probabilistic programming concepts, and addressing the complexities of backtesting in the context of time series forecasting. The project serves as a practical exploration of diverse forecasting methodologies, laying the groundwork for further enhancements and applications in the realm of time series analysis. This has been done by comparing time-series forecasting results obtained from Pyro, against conventional Python models like ARIMA and focusing on the edge that Pyro provides for such problem statements. We explicitly compare various models written in Pyro against simple linear regression and even more well-suited ARIMA models to tackle the problem and display how Pyro fares against them in various parameters leveraging the probabilistic programming principles.

## Software Architecture

The code is written in Python and utilizes Jupyter notebooks. The main components include:
- **ARIMA Forecasting:** Utilizes the `statsmodels` library to implement an ARIMA model for time series forecasting.
- **Standard Linear Regression:** Uses the `scikit-learn` library to implement a linear regression model as a baseline.
- **Pyro Forecasting Models:** Implements forecasting models using Pyro, a probabilistic programming library.

### Software Architecture Details

The integration of Pyro is highlighted in the architecture, emphasizing how probabilistic programming augments the forecasting models. The flexibility of Pyro is showcased, aligning with the principle of modularity. Python's data science libraries, coupled with Pyro, foster a comprehensive approach to capturing uncertainty in the gasoline price forecasting process. We leverage the collaborative environment provided by various IPython Notebook handlers like Jupyter Notebooks and Google Collab to integrate our contributions. Key components include data preprocessing, exploratory data analysis, and the implementation of time series forecasting models. The code is modular, with dedicated sections for data visualization, linear regression, ARIMA modeling, and Pyro-based forecasting. The input dataset, "gasoline.csv," promotes simplicity and ease of reproducibility. The architecture, balancing accessibility and functionality, is well-suited for educational purposes and exploratory analysis.

### Graphs for Enhanced Understanding

- **Dataset Overview:** A graph emphasizing the probabilistic nature of the dataset, aligning with the use of Pyro for uncertainty
![dataset 1](https://github.com/DarshD4/POPL-group42/assets/142094108/c4e4e610-e78a-4132-a78c-22a4880bf62e)

...
![dataset 2](https://github.com/DarshD4/POPL-group42/assets/142094108/c0702bae-5f7c-4699-964c-b84a0e7266fb)

Results and Tests

The code includes visualization of the data, predictions, and evaluation metrics for different models. The ARIMA model, linear regression, and Pyro-based models are compared using CRPS, MAE, and RMSE metrics. Visualizations include time series plots and forecast comparisons.
result of linear reg

![linear eg1](https://github.com/DarshD4/POPL-group42/assets/142094108/f7335348-20a8-4643-a5ac-ca735ec07674)

zooming in on the predict vs test section highlighted in orange

![linear reg2](https://github.com/DarshD4/POPL-group42/assets/142094108/995141f8-b10d-4798-8573-a400fdfc5580)

line plot of residuals in ARIMA
![rsidual 1](https://github.com/DarshD4/POPL-group42/assets/142094108/8ec07d85-1204-4d88-b12c-5d5eb3132034)

density plot of residuals

![residual2](https://github.com/DarshD4/POPL-group42/assets/142094108/02982ce6-7253-4b4f-ac61-bfe682bf244e)

autocorrelation plot of residuals
![residual 3](https://github.com/DarshD4/POPL-group42/assets/142094108/c8d3346d-ebfe-4384-842c-3a7468750bfc)

using rolling forecasting ARIMA model
![arima 1](https://github.com/DarshD4/POPL-group42/assets/142094108/c3612618-3845-432a-8784-3e05b980ed88)

plotting ARIMA forecast vs actual values
![arima 2](https://github.com/DarshD4/POPL-group42/assets/142094108/6d963389-8fa1-40dc-99b4-5e29ae368611)

Pyro model 1
![pyro model1](https://github.com/DarshD4/POPL-group42/assets/142094108/470291a9-643f-4ba0-905b-da1c56b6bf4f)

Pyro model 1 close up![pyro model1 2](https://github.com/DarshD4/POPL-group42/assets/142094108/5d69af8b-8c43-4f5a-9610-f08d8a1cceb0)

pyro model2
![pyro model2 1](https://github.com/DarshD4/POPL-group42/assets/142094108/158f5071-2832-4330-a867-e8ddc345274a)

zoom in on the predict vs test portion
![pyro model2 2](https://github.com/DarshD4/POPL-group42/assets/142094108/995b4a69-cdcf-49b5-bff7-9046f6d405cf)

pyro model 3
![pyro model 3 1](https://github.com/DarshD4/POPL-group42/assets/142094108/9c841ac8-3d61-46a7-b732-185191e24dd3)

pyro model 3 close-up
![pyro model3 2](https://github.com/DarshD4/POPL-group42/assets/142094108/cfd90d75-499d-4c86-bb18-c24b9bf53539)

evaluating model performance using backtesting
![backtest](https://github.com/DarshD4/POPL-group42/assets/142094108/5fc29ef6-d9ca-456b-ba3d-7e90dd938234)

## Potential for Future Work

Given more time, potential areas for improvement and expansion include:

- **Hyperparameter Tuning:** Fine-tune hyperparameters for ARIMA and Pyro models to enhance performance.
- **Ensemble Methods:** Explore ensemble methods to combine predictions from multiple models for better forecasting accuracy.
- **Feature Engineering:** Experiment with additional features or transformations to improve model understanding and predictive power.
- **Dynamic Model Updating:** Implement dynamic model updating to adapt models to changing patterns in the data.

## Experience and Difficulties

- **ARIMA Complexity:** ARIMA models require careful selection of parameters, and tuning them for optimal performance can be challenging.
- **Probabilistic Programming:** Working with Pyro involves understanding complex probabilistic concepts, which may pose a learning curve.
- **Backtesting Challenges:** Designing effective backtesting strategies and interpreting results can be complex in time series forecasting.
