#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 17:44:24 2020

@author: lupotto
"""
#data handling
import numpy as np
#dataset
from sklearn.datasets import load_boston
#libraries for plotting
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
#system library
import sys

class LinearRegression():
    
    def __init__(self, n_features):
        
        #thetas is the parameters of the model we add + 1 for the bias term
        #the initialitzation is random everytime the model is called
        self.thetas = np.random.rand(n_features + 1)
        
    #TODO: Add num_iterations & update thetas
    #TODO: RMSE error 
    def fit(self, X, y):
        #learning rate
        alpha = 1
        #num of samples
        m = X.shape[0]
        
        #hypotesis definition 
        h_x = self.thetas[0] + self.thetas[1] * X
        
        #show how it fits to the data
        show_data_matplot(X ,y, h_x)
        
        
        #gradient descent to optimize params.
        self.thetas[0] = self.thetas[0] - (alpha/m) * np.sum(h_x - y)
        
        self.thetas[1] = self.thetas[1] - (alpha/m) * np.sum(
                                        np.dot(((h_x) - y), self.thetas[1]))
        
       

       

#TODO: Check why does not work with Plotly library
def show_data_plotly(X, y):
        
    #X and Y axis inputs for Plotly graph.
    plot_data = [
            go.Scatter(
                x=X,
                y=y
                )   
            ]
    plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Boston Home Prices'
        )
    #fig = go.Figure(data=plot_data, layout=plot_layout)
    fig = go.Figure(data=go.Scatter(x=X, y=y, mode='markers'))
    fig.show()
    
#Provisional way to plot the data with one feature (CRIM)
def show_data_matplot(X, y, h_x):
    
     plt.figure(figsize=(5, 4))
     plt.plot(X, h_x, '-r')
     plt.scatter(X, y)
     plt.ylabel('Price')
     plt.xlabel("CRIM Attribute")
     plt.show()



if __name__ == '__main__':
    
    #load dataset
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    
  
    #fiting the model with one feature
    model = LinearRegression(n_features = 1)
    model.fit(X[:, 0], y)
    
    
    
    
    