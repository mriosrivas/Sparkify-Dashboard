import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel

st.set_page_config(layout="wide")


@st.cache_data
def get_histogram(df, col, bins):
    """
    Compute a histogram of the values in a column of a pandas DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to histogram.
    col : str
        The name of the column in `df` to use for the histogram.
    bins : int or sequence of scalars or str, optional
        Specification of histogram bins. If an int, it defines the number of equal-width bins in the given range (0 to the maximum value of `col` in `df`). If a sequence, it defines the bin edges, including the rightmost edge. If a string, it must be one of the methods described in `numpy.histogram()` documentation.

    Returns:
    --------
    hist_values : numpy.ndarray
        The values of the histogram bins. This array has shape `(bins,)`.
    bin_edges : numpy.ndarray
        The edges of the histogram bins. This array has shape `(bins+1,)`.

    Raises:
    -------
    TypeError
        If `df` is not a pandas DataFrame, `col` is not a string, or `bins` is not an int or sequence of scalars or str.
    ValueError
        If `col` is not a column name in `df`.
        If `bins` is not a valid specification for histogram bins.

    Examples:
    ---------
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 10, 30, 20]})
    >>> hist_values, bin_edges = get_histogram(df, 'B', bins=3)
    >>> hist_values
    array([2, 1, 2])
    >>> bin_edges
    array([ 0.        , 10.        , 20.        , 30.        ])

    Note:
    -----
    This function relies on the `numpy.histogram()` function to compute the histogram. Refer to the `numpy.histogram()` documentation for more information about the `bins` parameter and other details of the computation.
    """

    hist_values, bin_edges = np.histogram(df[col], bins=bins, range=(0, df[col].max()))
    return hist_values, bin_edges


@st.cache_data
def get_bins_pos(bins):
    """
        Compute the position and width of the bins in a histogram.

        Parameters:
        -----------
        bins : numpy.ndarray
            The edges of the bins in the histogram. This array has shape `(n_bins+1,)`.

        Returns:
        --------
        pos : numpy.ndarray
            The positions of the bins in the histogram. This array has shape `(n_bins,)` and is computed as the midpoint between adjacent bin edges.
        delta : float
            The width of the bins in the histogram. This value is computed as the difference between the second and first elements of the `bins` array.

        Raises:
        -------
        TypeError
            If `bins` is not a numpy array or if its elements are not numeric.

        Examples:
        ---------
        >>> bins = np.array([0, 1, 2, 3])
        >>> pos, delta = get_bins_pos(bins)
        >>> pos
        array([0.5, 1.5, 2.5])
        >>> delta
        1

        Note:
        -----
        This function assumes that the bins in the histogram are of equal width. If the bins are not of equal width, the `delta` value returned by this function will not be accurate.
    """
    pos = (bins[1:] + bins[:-1]) / 2
    delta = bins[1] - bins[0]
    return pos, delta


@st.cache_data
def get_churn(df):
    """
        Add a 'result' column to a pandas DataFrame indicating whether each row corresponds to a customer who churned or not.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing the customer data. This DataFrame should have a column named 'label' containing binary labels (0 for no churn, 1 for churn).

        Returns:
        --------
        result_df : pandas.DataFrame
            The input DataFrame with an additional column named 'result'. This column contains strings indicating whether each row corresponds to a customer who churned or not. The values of this column are 'churn' for label==1 and 'no churn' for label==0.

        Raises:
        -------
        TypeError
            If `df` is not a pandas DataFrame or if it does not contain a column named 'label'.
        ValueError
            If the 'label' column contains values other than 0 and 1.

        Examples:
        ---------
        >>> df = pd.DataFrame({'label': [0, 1, 0, 1, 1]})
        >>> result_df = get_churn(df)
        >>> result_df
           label    result
        0      0  no churn
        1      1     churn
        2      0  no churn
        3      1     churn
        4      1     churn

        Note:
        -----
        This function assumes that the 'label' column of the input DataFrame contains binary labels (0 for no churn, 1 for churn). If the column contains other values, the function will raise a ValueError.
    """
    df['result'] = df['label'].apply(lambda x: 'churn' if x == 1 else 'no churn')
    return df


@st.cache_data
def get_hours(df):
    """
        Compute the duration of each row in a pandas DataFrame in hours and add a 'time' column.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing the duration data. This DataFrame should have a column named 'total_length' containing durations in seconds.

        Returns:
        --------
        time_df : pandas.DataFrame
            The input DataFrame with an additional column named 'time'. This column contains the duration of each row in hours, computed by dividing the 'total_length' column by 3600 and rounding down to the nearest integer.

        Raises:
        -------
        TypeError
            If `df` is not a pandas DataFrame or if it does not contain a column named 'total_length'.
        ValueError
            If the 'total_length' column contains non-numeric values.

        Examples:
        ---------
        >>> df = pd.DataFrame({'total_length': [3600, 7200, 5400, 9000]})
        >>> time_df = get_hours(df)
        >>> time_df
           total_length  time
        0          3600     1
        1          7200     2
        2          5400     1
        3          9000     2

        Note:
        -----
        This function assumes that the 'total_length' column of the input DataFrame contains numeric durations in seconds. If the column contains non-numeric values, the function will raise a ValueError.
        """
    df['time'] = df['total_length'].apply(lambda x: int(x / 3600))
    return df


@st.cache_data
def get_interactions(df, selection):
    """
            Compute the average percentage of user interactions for each page category in a pandas DataFrame.

            Parameters:
            -----------
            df : pandas.DataFrame
                The DataFrame containing user interaction data. This DataFrame should have columns named 'home', 'roll_advert', 'about', 'add_playlist', 'nextsong', 'error', 'n_pages', and 'label'.
            selection : str
                A string indicating the selection criterion for computing interaction percentages. Valid values are 'all' (to compute percentages for all users), 'churn' (to compute percentages for churned users only), or 'no_churn' (to compute percentages for non-churned users only).

            Returns:
            --------
            cols : list
                A list of strings containing the page category names.
            vals : list
                A list of floats containing the average percentage of user interactions for each page category, as computed based on the selection criterion.

            Raises:
            -------
            ValueError
                If `selection` is not one of the valid values ('all', 'churn', or 'no_churn'), or if the input DataFrame does not contain the required columns.

            Examples:
            ---------
            >>> df = pd.DataFrame({
            ...     'home': [200, 150, 250, 180],
            ...     'roll_advert': [10, 5, 20, 8],
            ...     'about': [15, 12, 18, 13],
            ...     'add_playlist': [20, 18, 22, 19],
            ...     'nextsong': [500, 400, 600, 450],
            ...     'error': [5, 3, 8, 4],
            ...     'n_pages': [1000, 800, 1200, 900],
            ...     'label': [0, 1, 1, 0]
            ... })
            >>> cols, vals = get_interactions(df, 'all')
            >>> cols
            ['home', 'roll_advert', 'about', 'add_playlist', 'nextsong', 'error', 'other']
            >>> vals
            [19.0, 1.5, 2.0, 2.0, 48.3, 0.5, 26.7]

            Note:
            -----
            This function computes the average percentage of user interactions for each page category based on the selection criterion specified by the `selection` parameter. If the `selection` parameter is not one of the valid values ('all', 'churn', or 'no_churn'), the function will raise a ValueError. If the input DataFrame does not contain the required columns, the function may raise a KeyError or a TypeError. The 'other' category contains the percentage of interactions that do not fall into any of the other categories.
    """

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
    """
        Generate an Altair visualization of the churn percentage in a given dataframe.

        Parameters:
        df (pandas.DataFrame): A pandas DataFrame containing the data.

        Returns:
        chart (altair.Chart): An Altair chart representing the churn percentage.
        labels (altair.Chart): An Altair chart representing the labels on the chart.

        Example:
        >>> chart, labels = get_churn_plot(df)
    """
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
    """
        Creates an interactive scatter plot and a bar chart of the interactions of users with the music app.

        Args:
        - df: A pandas DataFrame containing the interactions data.
        - percentage: A float representing the percentage of data to be plotted.

        Returns:
        - points: An Altair chart object representing the scatter plot.
        - bars: An Altair chart object representing the bar chart.
    """

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
    """
        Creates a histogram of the number of users by the amount of time they spent on the music app.

        Args:
        - df: A pandas DataFrame containing the interactions data.
        - bins: An integer representing the number of bins to be used in the histogram.

        Returns:
        - hist_df: A pandas DataFrame with the columns 'hours' and 'users' containing the histogram data.
    """

    df = get_hours(df)

    hist_time, bin_time = get_histogram(df, 'time', bins * 100)
    pos, delta = get_bins_pos(bin_time)

    hist_df = pd.DataFrame(np.array([pos.astype('int'), hist_time]).T,
                           columns=['hours', 'users'])

    return hist_df


def get_users_interactions_plot(df, selection):
    """
        Creates a pie chart showing the percentage of interactions in the music app by category for a given selection.

        Args:
        - df: A pandas DataFrame containing the interactions data.
        - selection: A string representing the selection to be used in the chart. It can be 'all', 'churn', or 'no_churn'.

        Returns:
        - pie_inter: An altair Chart object representing the pie chart.
        - legend: An altair Chart object representing the legend of the pie chart.
    """

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
    """
       Creates a SparkSession with a given name if it doesn't exist yet, otherwise returns an existing one.

       Args:
       - name: A string representing the name of the SparkSession.

       Returns:
       - spark: A SparkSession object with the given name.
    """

    spark = SparkSession.builder.appName(name).getOrCreate()
    return spark


@st.cache_resource
def load_model(file):
    """
        Loads a previously trained pipeline model from disk.

        Args:
        - file: A string representing the path to the file containing the trained pipeline model.

        Returns:
        - model: A PipelineModel object representing the trained pipeline model loaded from disk.
    """

    model = PipelineModel.load(file)
    return model


@st.cache_data
def get_churn_pred(user):
    """
        Predicts the probability of a user churning using a trained random forest model.

        Args:
        - user: A dictionary containing user data as key-value pairs.

        Returns:
        - prob_churn: A float representing the predicted probability of the user churning.
    """

    user_df = spark.createDataFrame(user)
    prob_churn = round(rfModel.transform(user_df).collect()[0][-2][1], 2)
    return prob_churn


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
             'total_length': total_length / 3600}]

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
