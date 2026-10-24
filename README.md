# SC5002 Lab Assignment 1: Linear vs Ridge Regression

This repo contains the Google Colab notebook for Part 2 of the SC5002 AI Fundamentals & Applications lab assignment. I trained Linear Regression and Ridge Regression models on the California Housing dataset to see how changing Ridge's penalty (alpha) affects how well the model predicts house prices.

## What's in this repo

- `SC5002_Lab_Assignment_1.ipynb`: the full notebook, with all outputs saved (dataset preview, scores and chart)

## What the notebook does

1. **Loads the data.** It takes a random sample of 1,000 rows from the California Housing dataset.
2. **Sets up the problem.** The target is `MedHouseVal`, the median house value of an area (in units of $100,000). The model predicts it using 8 features: median income, house age, average rooms, average bedrooms, population, average occupancy, latitude and longitude.
3. **Scales the features.** The columns are on very different scales (population is in the thousands, average bedrooms is around 1), so `StandardScaler` puts them all on the same scale before training.
4. **Trains and scores the models.** It compares Linear Regression with Ridge Regression at alpha = 0.1, 10 and 1000, using 5-fold cross-validation and R² as the score.
5. **Plots the results** as a bar chart.

## Results

| Model | Mean 5-Fold CV R² |
|---|---|
| Linear Regression | 0.6457 |
| Ridge (alpha = 0.1) | 0.6457 |
| Ridge (alpha = 10) | 0.6443 |
| Ridge (alpha = 1000) | 0.3574 |

## What I found

- **Small alpha barely changed anything.** Ridge with alpha = 0.1 and 10 scored almost the same as plain Linear Regression. This indicates that the model wasn't really overfitting, so Ridge's penalty didn't have much to fix.
- **A huge alpha made the model worse.** At alpha = 1000, R² dropped to 0.3574. The penalty shrank the feature weights so much that the model could barely use the data, which is underfitting. It still learned something (R² is well above 0), just a lot less.
- **Alpha is a balance.** It should be strong enough to stop overfitting, but not so strong that the model underfits.
- **Why cross-validation.** Instead of testing on one train/test split, 5-fold cross-validation tests on five different splits and averages them, so one lucky or unlucky split can't skew the score.

