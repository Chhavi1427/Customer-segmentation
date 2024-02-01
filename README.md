
# Customer Segmentation 

This Streamlit app performs customer segmentation using K-Means clustering. It allows users to upload their dataset in CSV or Excel format and visualize the segmentation results in a 2D scatter plot.

## Table of Contents
- [Prerequisites](#Prerequisites)
- [Usage](#usage)
  - [Getting-Started](#getting-started)
  - [Running-the-Application](#running-the-application)
- [Features](#features)
- [Files](#files)
- [Technologies](#technologies)
- [How-to-Use](#how-to-use)
## Prerequisites

The code you provided is a Streamlit app for customer segmentation, and to run it successfully, you'll need a few prerequisites. Here are the prerequisites and steps to set up the environment:

**1. Python:**
- Ensure you have Python installed on your system. You can download it from [Python's official website](https://www.python.org/).

**2.Dependencies:** 

- Install the required dependencies by running the following command in the terminal:
```bash
pip install streamlit pandas numpy scikit-learn plotly

```

**3. Streamlit:**

Streamlit should be installed using the following command:
```bash
pip install streamlit

```
## Usage

**Getting-Started**

1 Clone the repository to your local machine:

git clone https://github.com/Chhavi1427/Customer-segmentation

cd Email-spam-detection

2 Install the required dependencies. Make sure you have Python installed: 
```bash  
pip install -r requirements.txt
```

**Running-the-Application**


3.Run the Streamlit app:

```bash
streamlit run app.py
```
4 Open your web browser and go to https://customer-segmentation-fvo8ysvgvpd4dsfnirc6a5.streamlit.app/ to interact with the Spam Detection.


## Features

- **File Upload:** Easily upload your dataset by dragging and dropping or selecting files.
- **Cluster Selection:** Choose the number of clusters for the K-Means algorithm using a slider.
- **Interactive Visualization:** Visualize customer segmentation in a dynamic scatter plot, with each point representing a customer and colored according to their cluster.


## Files

- `app.py`: The Streamlit application code.
- `requirements.txt`: List of required Python packages.
## Technologies

- [Streamlit](https://www.streamlit.io/ "Streamlit Official Website")

- [NumPy](https://numpy.org/ "NumPy official Website")

- [Scikit-learn](https://scikit-learn.org/stable/)

- [Plotly-Express](https://plotly.com/python/)



## Contact Information and Project Details

**Project Name:** Customer Segmentation

**Author:** [Chhavi Modi ](https://github.com/Chhavi1427)

**Project Link:** https://github.com/Chhavi1427/Customer-segmentation

**Email:** modichavi1427@gmail.com

## How to Use

 **1. Upload Data:**
Drag and drop or select your dataset in CSV or Excel format.

**2.Select Number of Clusters:** Use the slider to choose the desired number of clusters for customer segmentation.

**3.View Results:** Explore the interactive scatter plot to understand the segmentation of customers based on selected features.