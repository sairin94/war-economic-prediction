# Capstone Project- Impact of America's Involvement in Global Wars on United States GDP

##  Table of Contents
- [ Table of Contents](#-table-of-contents)
- [ Overview](#-overview)
- [ Features](#-features)
- [ repository Structure](#-repository-structure)
- [ Modules](#modules)
- [ Getting Started](#-getting-started)
    - [ Installation](#-installation)
    - [ Running ](#-running-)
- [ Roadmap](#-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

---


##  Overview

This repository contains code and files for a project focused on model training, evaluation, and data management. It includes notebooks for exploratory data analysis and cleaning, as well as code files for model training and evaluation using LSTM, GRU, and CNN architectures. The code utilizes libraries such as TensorFlow, Keras, and scikit-learn for model training and evaluation. The repository also includes a requirements.txt file that specifies the dependencies needed to recreate the project environment.

---

##  Report

This report of the project is [here.](docs/Impact_of_Americas_Involvement_in_Global_Wars_on_United_States_GDP.pdf)

---

##  Features

|    | Feature           | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|----|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ⚙️ | **Architecture**  | The codebase follows a modular design pattern, with separate files for data management, model evaluation, and model training. The data management file provides functions for loading, preprocessing, and splitting data. The model evaluation file contains functions for model evaluation, hyperparameter tuning, and plotting. The model training file defines and trains models using LSTM, GRU, and CNN architectures. This design allows for separation of concerns and easy maintenance. The codebase also utilizes TensorFlow and Keras libraries for deep learning tasks.                                                                                                                                             |
| 📄 | **Documentation** | Project documentation is provided under capstone/docs/Capstone Project- Impact of America's Involvement in Global Wars on United States GDP                                                                                                                                         |
| 🔗 | **Dependencies**  | The codebase has a wide range of external dependencies. These include TensorFlow, Keras, pandas, numpy, scikit-learn, and various other libraries. The requirements.txt file lists the dependencies with their corresponding versions.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 🧩 | **Modularity**    | The codebase demonstrates a good level of modularity. It is organized into separate files, each focusing on a specific aspect of the project, such as data management, model evaluation, and model training. The functions within each file handle specific tasks, promoting reusability and maintainability. This modular design allows for easy modification, debugging, and extension of individual components without impacting the entire system. It also enhances code readability, reduces code duplication, and facilitates collaboration among team members. Having clearly defined responsibilities for each file and function simplifies debugging and testing efforts.                                                                                                                                      |
                                                                                                                                                                                                                                                         

---


##  Repository Structure

```sh
├── /
│   ├── data/
│   │   ├── consumer_price.xls
│   │   ├── Fedfunds.xls
│   │   ├── Gross_Domestic_Product.xls
│   │   ├── Gross_Savings_And_Investment.xls
│   │   ├── Inflation.xls
│   │   ├── military_spending.xls
│   │   ├── personal_consumption_expenditure.xls
│   │   ├── UNRATE.xls
│   │   └── war_final.csv
│   ├── docs/
│   │   └── Capstone_Project-ImpactOfAmericasInvolvementInGlobalWarsOnUnitedStatesGDP.pdf
│   ├── notebooks/
│   │   ├── Best_Performing_DL_Model_Analysis.ipynb
│   │   ├── DL_Model_Finding.ipynb
│   │   ├── Exploratory_Data_Analysis.ipynb
│   │   └── Cleaning_And_PreProcessing_Dataset.ipynb
│   ├── requirements.txt
│   └── src/
│       ├── data_management.py
│       ├── model_evaluation.py
│       └── model_training.py

```

##  Modules

<details closed><summary>Root</summary>

| File                            | Summary                                                                                                                                                                                                                                                                                                                                                                   |
| ---                             | ---                                                                                                                                                                                                                                                                                                                                                                       |
| [requirements.txt]({file_path}) | This code snippet represents a directory tree structure and a file named requirements.txt. The directory tree contains various folders and files related to a project. The requirements.txt file is used to define the dependencies and their versions required for the project. It specifies the packages and their versions needed to recreate the project environment. |

</details>

<details closed><summary>Notebooks</summary>

| File                                                        | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ---                                                         | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| [Best_Performing_DL_Model_Analysis.ipynb]({notebooks/Best_Performing_DL_Model_Analysis.ipynb}) | The code in the notebook Best_Performing_Model_Analysis_modular.ipynb is primarily focused on evaluation of best performing.                                                                                                                                                                                                                                                                                                                                                                 |
| [Cleaning_And_PreProcessing_Dataset.ipynb]({notebooks/Cleaning_And_PreProcessing_Dataset.ipynb})                       | The code is a part of a directory tree structure for a project. It is specifically located in the notebooks directory under the file name cleaning_dataset.ipynb. The code file contains list of data sources regarding macroeconomic factors such as Consumer Price Index (CPI), Unemployment Rate (UNRATE), Personal Savings, Inflation Rate, Gross Domestic Product (GDP), Personal Consumption Expenditures (PCE), Gross Savings, Gross Saving as a Percentage of Gross National Income, Gross Domestic Investment, Net Saving as a Percentage of Gross National Income, military spending and Federal Funds Rate (FED FUNDS) pulled from Fred Economic Data, Bureau of Labor Statistics, and Bureau of Economic Analysis. The code appears to be documenting and providing references to different data sources for the project. |
| [DL_Model_Finding.ipynb]({notebooks/DL_Model_Finding.ipynb})              | This code, located in the Deep_Learning_modularized.ipynb notebook, imports necessary libraries for deep learning tasks. It specifically designed for model selection using hyperparamter tuning and performing analysis on target variable.                                                                                                                                                                                          |
| [Exploratory_Data_Analysis.ipynb]({notebooks/Exploratory_Data_Analysis})      | The code in the Exploratory_data_analysis_modular.ipynb notebook is used to import the necessary libraries for performing exploratory data analysis. This file contains exploratory data analysis about macroeconomic factors used in project.                                                                                                                                                                                                   |

</details>

<details closed><summary>Src</summary>

| File                               | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ---                                | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| [model_evaluation.py]({file_path}) | The code in the `model_evaluation.py` file performs model evaluation and prediction tasks for various types of models such as LSTM, CNN, and GRU. It includes functions for grid search of hyperparameters, scoring the model, predicting test data, and plotting the actual vs predicted values. Additionally, it provides functions for plotting the predicted values during war periods and comparing the predictions with and without the war variable. The code utilizes TensorFlow, Keras, and scikit-learn libraries for model training and evaluation.                                                                                                                                                                                 |
| [model_training.py]({file_path})   | The code in src/model_training.py defines functions for model training using LSTM, GRU, and CNN architectures. The define_model_lstm, define_model_gru, and define_model_cnn functions define and compile the respective models with the specified parameters. The train_model function trains the model using the given data and returns the trained model.                                                                                                                                                                                                                                                                                                                                                                                   |
| [data_management.py]({file_path})  | The code in the `src/data_management.py` file provides functions for loading and preprocessing data, as well as splitting the data into training and testing sets. The `load_data` function reads a CSV file and returns a DataFrame, removing a specified number of rows at the end. The `preprocess_data` function preprocesses the data by setting the index, converting object columns to numeric, applying standard scaling, and imputing missing values. The `train_data_split` function splits the preprocessed data into input features (X) and target variables (Y) for training and testing. It creates sliding windows of past and future data points and returns the training and testing sets, as well as their respective sizes. |

</details>

---

##  Getting Started

***Dependencies***

All dependicies used in project are present in requirements.txt

###  Installation

1. Clone the  repository:
```sh
git clone https://github.com/sairin94/war-economic-prediction
```

2. Change to the project directory:
```sh
cd war-economic-prediction
```

3. Install the dependencies:
```sh
pip install -r requirements.txt
```

###  Running 

```sh
execute notebook
```



## License

This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file.

---

##  Acknowledgments
This project would not have been possible without the support of:


- Dr Gabrielle O'Brien, 
Lecturer III & Research Investigator(University Of Michigan)


## References
Brownlee, J. (n.d.). How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras. Machine Learning Mastery. Retrieved from https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
 
Brownlee, J. (n.d.). How to Grid Search Deep Learning Models for Time Series Forecasting. Machine Learning Mastery. Retrieved from https://machinelearningmastery.com/how-to-grid-search-deep-learning-models-for-time-series-forecasting/
 
Brownlee, J. (n.d.). How to Develop Convolutional Neural Network Models for Time Series Forecasting. Machine Learning Mastery. Retrieved from https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/
 
Muthun, A. K. (n.d.). Time-Series-Neural-Network-Grid-Search. GitHub. Retrieved from https://github.com/akmuthun/Time-Series-Neural-Network-Grid-Search
 
Jiwidi. (n.d.). Time Series Forecasting with Python. GitHub. Retrieved from https://github.com/jiwidi/time-series-forecasting-with-python/blob/master/time-series-forecasting-tutorial.ipynb



