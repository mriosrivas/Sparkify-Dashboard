# Sparkify Dashboard

The Sparkify Dashboard is a web application developed on Streamlit that allows the user to explore insights from the Sparkify music streaming service. It features several interactive charts that provide valuable information about the service, including user engagement, time spent on the platform, interactions with the service, and the overall churn rate. Additionally, the app provides a churn prediction model developed using PySpark's Random Forest Classifier.

## Installation

To run the Sparkify Dashboard, you will need to install the following dependencies:

- `numpy==1.23.5`
- `pandas==1.2.4`
- `streamlit==1.19.0`
- `pyspark==3.3.1`
- `altair==4.2.2`

You can install them using pip:

```bash
pip install numpy==1.23.5 pandas==1.2.4 streamlit==1.19.0 pyspark==3.3.1 altair==4.2.2
```

Or with the provided `requirements.txt` file

```bash
pip install -r requirements.txt
```

## Running the App

To run the Sparkify Dashboard, navigate to the directory where the `sparkify.py` file is located and execute the following command on the terminal:

```bash
streamlit run sparkify.py
```

This command will start the Streamlit server and open the application in your default web browser.

You can also see a live version of this app on Streamlit Cloud here:

https://mriosrivas-sparkify-dashboard-sparkify-crrui4.streamlit.app/



## Features

The Sparkify Dashboard allows you to explore the following interactive charts:

- **Sparkify's Overall Churn:** A pie chart that displays the percentage of users who churned from the service and those who remained.
- **User's Engagement:** A scatter plot that shows the relationship between the number likes and dislikes on the platform. The chart also allows filtering users regions of interest.
- **User's Time on Sparkify:** A histogram that displays the distribution of user's time spent on the platform. The chart also allows the user to choose the number of bins.
- **User's Interactions with Sparkify:** A pie chart that displays the percentage of interactions (i.e. likes, dislikes, advertisements seen, etc.) of all users.
- **Churn Prediction Model:** A form that allows the user to input features related to a specific user and obtain the predicted probability of that user churning. The model considers features such as time spent on the app, likes and dislikes, songs played, advertisements seen, among others.

## License

The Sparkify Dashboard is licensed under the MIT License.
