# FYS-STK3155/4155 Project 1

## Group members: 

Viktor Bilstad, Frederik Callin Østern, Heine Elias Husdal

## Project description

In this project, we have investigated the performance of Ordinary Least Squares (OLS), Ridge and Lasso regression techniques for polynomial fits to the Runge function. This includes an analysis of analytical solutions, numerical solutions with stochastic and non-stochastic gradient descent algorithms, as well as resampling techniques (bias-variance tradeoff and k-fold cross validation). 

## Description of GitHub folders and files

### code

Contains all of the code for the project, including: 

#### project_1ab.ipynb: 

Contains all of the results for the subsection "Analytical results for OLS and Ridge regression" in the report. 

#### project_1c.ipynb: 

Contains the results in figure 9 in the report. 

#### project_d.ipynb: 

Contains the top plots (on non-stochastic gradient descent) in figures 6 and 7 in the report, as well as the results for the first two columns in table 1. 

#### project_1ef.ipynb: 

Contains the results for the bottom plots (on stochastic gradient descent) in figures 6 and 7, the results for figure 8, as well as the results for the last column in table 1. 

#### project_1g.ipynb

Contains the results for figure 1, as well as the bias-variance results in table 2. 

#### project_1h.ipynb

Contains the results in figure 11 on k-fold cross validation. 

##### utils.py

A module with some simple functions used throughout the project. 

#### methods

Folder with files implementing the methods used. This includes: 

- **regression_methods.py**: Implements the gradients of the cost functions. 

- **resampling.py**: Implements resampling techniques, including bias-variance tradeoff and k-fold cross validation. 
- **step_methods.py**: Implements classes for the update of the parameters in one iteration, for each of the various gradient descent algorithms. 
- **training_methods.py**: Implements classes to train a model using gradient descent. 


## How to run code

-Clone repository: git clone https://github.com/HeineEH/[RepositoryName].git

-Navigate to project:
cd [RepositoryName]

-pip install -r requirements






