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

## POPL Aspects


Probabilistic Programming (Pyro): The integration of Pyro introduces a probabilistic approach to forecasting, capturing uncertainty in predictions through Bayesian modeling.

Time Series Analysis (ARIMA): The ARIMA model is employed to analyze and forecast time-dependent patterns, considering autocorrelation and seasonality in the data.

Data Visualization: Matplotlib is used for data visualization, aligning with the importance of conveying insights through clear and informative plots.

Backtesting Methodologies: The inclusion of backtesting techniques addresses the need for robust model evaluation, ensuring predictive capabilities extend beyond training data.

Modular Code Design: The code is structured into modular sections, promoting code readability, reusability, and ease of maintenance.



Probabilistic Programming (Pyro): The use of Pyro for probabilistic programming aligns with principles related to uncertainty modeling and Bayesian methods.

Modularity: The modular design of the code, as seen in the distinct sections for linear regression, ARIMA modeling, and Pyro-based forecasting, reflects a principle of modularity, promoting code organization and reusability.

Data Abstraction: The project involves abstracting and manipulating data, adhering to principles of data abstraction common in programming languages.

Error Handling: Principles related to error handling and robustness may be evident in how the code deals with exceptions, especially in the backtesting and evaluation stages.

Readability: The use of clear variable names, comments, and structured code contributes to readability, a fundamental principle in programming languages design.

Testing Methodologies: Backtesting and model evaluation methodologies showcase a commitment to testing and validating the robustness of the implemented models.

Interactive Development Environment: The use of Jupyter Notebooks and Google Colaboratory aligns with principles of interactive programming environments, facilitating exploratory analysis and collaborative development.

Pyro Forecasting Models
Probabilistic Programming (Pyro):

Lines 39-58: The entire block involves defining and fitting Pyro models. The use of Pyro is explicit here, adhering to the probabilistic programming principles.
Backtesting and Evaluation:

Line 70: print("CRPS: ", crps(y_test, y_pred_dists)) - This line calculates and prints the Continuous Ranked Probability Score (CRPS), a metric that emphasizes the probabilistic nature of predictions from Pyro models.
Data Visualization
Data Visualization:
Lines 33-44: These lines involve plotting time series, residuals, and other visualizations. While not explicitly probabilistic, visualizations play a crucial role in understanding uncertainty and model performance.
Overall Project Structure
Modular Code Design:

The code is structured into sections dedicated to different aspects of the project, adhering to the principle of modularity. Each section focuses on a specific modeling approach (linear regression, ARIMA, Pyro), making the code more readable and maintainable.
Probabilistic Programming (Pyro):

The sections related to Pyro models (lines 39-58 and beyond) demonstrate the modular integration of probabilistic programming, aligning with the modularity principle.
These lines collectively demonstrate how the code incorporates various POPL aspects, emphasizing both linear regression and probabilistic programming principles.


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
