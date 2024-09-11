# Cab Booking Demand Prediction

This project aims to forecast the demand for cab bookings in a city by leveraging historical booking data and weather information. The goal is to predict the total number of cab bookings for each hour, based on various features such as temperature, weather conditions, holidays, and time of day.

## Project Overview

Cab booking systems have transformed the way people travel within cities by enabling users to book rides via apps. Efficiently predicting the demand for cabs can help reduce waiting times, optimize cab availability, and improve overall customer satisfaction. This project aims to combine historical cab booking data with external factors like weather to develop a machine learning model that can accurately predict hourly booking demand.

### Problem Statement

Given hourly cab booking data spanning two years, you are tasked with predicting the total count of cabs booked for each hour in the test set. The dataset contains information on weather conditions, whether the day is a holiday or a working day, and the timestamp of the booking. The goal is to train machine learning models that can forecast the demand based on these features.

## Dataset

The dataset contains the following columns:

- `datetime`: Hourly timestamp of cab booking.
- `season`: The season of the year (Spring, Summer, Autumn, Winter).
- `holiday`: Whether the day is a holiday.
- `workingday`: Whether the day is a working day (not a weekend or holiday).
- `weather`: Weather conditions (Clear, Cloudy, Light Rain, Heavy Rain).
- `temp`: Temperature in Celsius.
- `atemp`: Feels like temperature in Celsius.
- `humidity`: Relative humidity.
- `windspeed`: Wind speed.
- `Total_booking`: The total number of cabs booked (only in training data).

## Workflow

### 1. Data Import and Preprocessing
- Import necessary libraries and load the training and test datasets.
- Check the data types, shape, and presence of any missing values.

### 2. Feature Engineering
- Create new columns (`date`, `hour`, `weekDay`, `month`) from the `datetime` column.
- Convert categorical columns (`season`, `holiday`, `workingday`, `weather`) into appropriate categories.
- Drop the `datetime` column after extracting useful features.

### 3. Outlier Detection and Removal
- Identify outliers using box plots for various features like `season`, `hour`, and `workingday`.
- Remove outliers from the dataset.

### 4. Correlation Analysis
- Analyze the correlation between `Total_booking` and other continuous features (`temp`, `atemp`, `humidity`, `windspeed`).

### 5. Data Visualization
- Visualize the distribution of the `Total_booking` column.
- Explore how the number of bookings varies by different features such as `Month`, `Season`, `Hour`, and `Weekday`.
- Plot histograms of continuous variables to understand their distributions.

### 6. Data Transformation
- Convert categorical variables into one-hot encoded vectors to make them suitable for machine learning algorithms.

### 7. Model Building
- Split the dataset into training and testing sets.
- Fit several machine learning models, including:
  - Random Forest Regressor
  - AdaBoost Regressor
  - Bagging Regressor
  - Support Vector Regressor (SVR)
  - K-Neighbors Regressor
- Evaluate each model based on **Root Mean Squared Error (RMSE)**.

### 8. Hyperparameter Tuning
- Use **GridSearchCV** to find the best hyperparameters for the top-performing model.

### 9. Prediction and Evaluation
- Make predictions on the test dataset.
- Compute the **Mean Squared Log Error (MSLE)** for model evaluation.

## Models Used

The following machine learning models are used in this project:
- Random Forest Regressor
- AdaBoost Regressor
- Bagging Regressor
- Support Vector Regressor (SVR)
- K-Neighbors Regressor

## Performance Metric

- **Root Mean Squared Error (RMSE)** is used for evaluating the performance of the models.
- **Mean Squared Log Error (MSLE)** is used for final evaluation on the test set.

## Setup and Installation

You can run the project in **Edureka’s CloudLab**, a cloud-based Jupyter Notebook environment pre-installed with Python and other necessary packages. Alternatively, if you wish to run the project locally, ensure that the following libraries are installed:

```bash
pip install numpy pandas scikit-learn seaborn matplotlib
```

## Project Structure

- **data/**: Contains the training and test datasets.
- **notebooks/**: Contains Jupyter Notebooks for data analysis, feature engineering, model building, and evaluation.
- **models/**: Saved models for future predictions.
- **results/**: Plots and visualizations generated during the analysis.
- **README.md**: The project documentation.

## Results

- Each machine learning model is evaluated based on **RMSE**. The results are visualized using a factor plot to compare the models' performance.
- After selecting the best-performing model, hyperparameter tuning is performed to further optimize the model.
- The final predictions on the test set are evaluated using **Mean Squared Log Error (MSLE)**.

## Conclusion

This project demonstrates how historical data, along with external factors like weather and holidays, can be used to predict cab booking demand. By optimizing cab availability based on demand forecasts, ride-hailing services can reduce user wait times and improve service efficiency.

## Future Work

- Additional features such as traffic data or special events could be incorporated to further improve the model’s accuracy.
- Experiment with more advanced models like Gradient Boosting or deep learning models for potential performance improvements.

