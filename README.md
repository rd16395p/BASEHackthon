![Mapbox][https://github.com/rd16395p/BASEHackthon/blob/master/Screen%20Shot%202019-12-14%20at%201.15.01%20PM.png]

# BASEHackthon

Rebecca D’Agostino, Naiyu Tian, Sabryna Davis, Lena Pinard  


## Introduction

Poverty and job vacancies is a large problem in NYC. In order to combat this issue, we are proposing a web app that is powered by real data in order to match job seekers to job centers, where they can transition to open jobs. 

## How to use the code

For the python section, a download of the packages is required. Anacaonda is recommended to download and create the python environment.  
For the javascript section, no setup is needed as CDNs were used.   

checkdata.py - Data pipeline, takes in the raw data and returns a .csv of the data group by job titles  
plotdata.ipynb - Juypter Notebook that visualizes prepared data  
predictionmodel.py - Python file that creates a linear regression for prediction, using tensorflow  
index.html - Map of training locations where people can learn new skills for the careers found in the dataset  

## Future Work

The app could be expanded to be a full web-app, with a Javascript Front-end, and a mongo backend. 
