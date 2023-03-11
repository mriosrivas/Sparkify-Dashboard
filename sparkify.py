import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel

st.set_page_config(layout="wide")


@st.cache_data
def get_histogram(df, col, bins):
    hist_values, bin_edges = np.histogram(df[col], bins=bins, range=(0, df[col].max()))
    return hist_values, bin_edges


@st.cache_data
def get_bins_pos(bins):
    pos = (bins[1:] + bins[:-1]) / 2
    delta = bins[1] - bins[0]
    return pos, delta


@st.cache_data
def get_churn(df):
    df['result'] = df['label'].apply(lambda x: 'churn' if x == 1 else 'no churn')
    return df


@st.cache_data
def get_hours(df):
    df['time'] = df['total_length'].apply(lambda x: int(x / 3600))
    return df


@st.cache_data
def get_interactions(df, selection):
    cols = ['home',
            'roll_advert',
            'about',
            'add_playlist',
            'nextsong',
            'error']

    vals = []
    for c in cols:
        if selection == 'all':
            vals.append(round(100 * (df[c] / df['n_pages']).mean(), 1))
        elif selection == 'churn':
            vals.append(round(100 * (df[df['label'] == 1][c] / df[df['label'] == 1]['n_pages']).mean(), 1))
        elif selection == 'no_churn':
            vals.append(round(100 * (df[df['label'] == 0][c] / df[df['label'] == 0]['n_pages']).mean(), 1))

    cols.append('other')
    vals.append(round(100 - sum(vals), 1))

    return cols, vals


def get_churn_plot(df):
    churn_percentage = (df['label'].mean() * 100).round(1)

    source = pd.DataFrame({"Retention": ['churn', 'no churn'],
                           "value": [churn_percentage, 100 - churn_percentage]})

    pie = alt.Chart(source).mark_arc(radius=100, innerRadius=50).encode(
        theta=alt.Theta(field="value", type="quantitative"),
        color=alt.Color(field="Retention", type="nominal"),
    )

    labels = pie.mark_text(radius=125, size=15).encode(text="value:Q")

    return pie, labels


def get_interactions_plot(df, percentage):
    df = get_churn(df)
    brush = alt.selection_interval()

    max_val = df.shape[0]

    points = alt.Chart(df.head(int(percentage * max_val / 100))).mark_point().encode(
        x='thumbs_down:Q',
        y='thumbs_up:Q',
        color=alt.condition(brush, 'result:N', alt.value('lightgray'))
    ).add_selection(
        brush
    )

    bars = alt.Chart(df).mark_bar().encode(
        y='result:N',
        color='result:N',
        x='count(result):Q'
    ).transform_filter(
        brush
    )
    return points, bars


def get_users_time_plot(df, bins):
    df = get_hours(df)

    hist_time, bin_time = get_histogram(df, 'time', bins * 100)
    pos, delta = get_bins_pos(bin_time)

    hist_df = pd.DataFrame(np.array([pos.astype('int'), hist_time]).T,
                           columns=['hours', 'users'])

    return hist_df  # st.line_chart(hist_df, x='hours', y=['users'])


def get_users_interactions_plot(df, selection):
    cols, vals = get_interactions(df, selection)

    source = pd.DataFrame({"Interactions": cols, "value": vals})

    pie_inter = alt.Chart(source).mark_arc(radius=75, innerRadius=25).encode(
        theta=alt.Theta(field="value", type="quantitative"),
        color=alt.Color(field="Interactions", type="nominal"),
    )

    legend = pie_inter.mark_text(radius=100, size=10).encode(text="value:Q")
    return pie_inter, legend


# Create spark session
@st.cache_resource
def get_spark_session(name):
    spark = SparkSession \
        .builder \
        .appName(name) \
        .getOrCreate()
    return spark


@st.cache_resource
def load_model(file):
    model = PipelineModel.load(file)
    return model


@st.cache_data
def get_churn_pred(user):
    user_df = spark.createDataFrame(user)
    prob_churn = round(rfModel.transform(user_df).collect()[0][-2][1], 2)
    return prob_churn


# Show Plots


if __name__ == '__main__':
    spark = get_spark_session('Sparkify')
    rfModel = load_model('model/')

    file_path = 'data/clean_data.csv'
    df = pd.read_csv(file_path)

    st.title('Sparkify Data Analysis')

    if st.checkbox('Show statistics'):
        st.write(df.describe().loc[:, df.columns != 'userId'])

    with st.sidebar:
        st.subheader('Churn Prediction Model')
        st.write('Fill in the values for our model to make a churn prediction on Sparkify.')
        n_pages = st.number_input('n pages', min_value=0, max_value=14000, value=433)
        thumbs_down = st.number_input('thumbs down', min_value=0, max_value=200, value=6)
        home = st.number_input('home', min_value=0, max_value=500, value=17)
        downgrade = st.number_input('down grade', min_value=0, max_value=150, value=5)
        roll_advert = st.number_input('roll advert', min_value=0, max_value=300, value=0)
        about = st.number_input('about', min_value=0, max_value=50, value=0)
        add_playlist = st.number_input('add playlist', min_value=0, max_value=350, value=12)
        nextsong = st.number_input('nextsong', min_value=0, max_value=12000, value=357)
        thumbs_up = st.number_input('thumbs_up', min_value=0, max_value=1000, value=16)
        error = st.number_input('error', min_value=0, max_value=30, value=0)
        submit_upgrade = st.number_input('submit upgrade', min_value=0, max_value=10, value=1)
        total_length = st.number_input('total length in hours', min_value=0, max_value=850, value=24)

    user = [{'n_pages': n_pages,
             'thumbs_down': thumbs_down,
             'home': home,
             'downgrade': downgrade,
             'roll_advert': roll_advert,
             'about': about,
             'add_playlist': add_playlist,
             'nextsong': nextsong,
             'thumbs_up': thumbs_up,
             'error': error,
             'submit_upgrade': submit_upgrade,
             'total_length': total_length/3600}]

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    pred_churn = get_churn_pred(user)
    col1.subheader("Predicted Churn Model:")
    col1.write("Use the left panel to set the values for our churn model predictor.")
    col1.write("User's churn probability:")
    col1.title(pred_churn)

    pie, labels = get_churn_plot(df)
    col1.subheader("Sparkify's Overall Churn")
    col1.write(pie + labels)

    percentage = col2.slider('Percentage of Users on Sparkify', 1, 100, 20)
    points, bars = get_interactions_plot(df, percentage)
    col2.subheader("User's Engagement")
    col2.write(points & bars)

    col3.subheader("User's Time on Sparkify")
    bins = col3.slider('Granularity', 1, 10, 5)
    hist_df = get_users_time_plot(df, bins)
    col3.line_chart = col3.line_chart(hist_df, x='hours', y=['users'])

    col4.subheader("User's Interactions with Sparkify")
    selection = col4.selectbox('Select the type of user', ['all', 'churn', 'no_churn'])
    pie_inter, legend = get_users_interactions_plot(df, selection)
    col4.write(pie_inter + legend)



