import pandas as pd
import numpy as np
import os
import warnings
from scipy.stats import chisquare
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from scipy.stats import (
    ttest_1samp,
    shapiro,
    levene,
    ttest_ind,
    mannwhitneyu,
    pearsonr,
    spearmanr,
    kendalltau,
    f_oneway,
    kruskal,
)
from statsmodels.stats.proportion import proportions_ztest

df_user_day = pd.read_csv("<user_day_fileoath>", parse_dates=["ds"])
df_user = pd.read_csv("<user table filepath>", parse_dates=["install_date"])



# Data Cleaning and Exploration
def check_table(df):
    """
    Function to check for missing values and duplicate rows in a table.

    Parameters:
    - df: DataFrame containing the table data.

    Returns:
    - missing_values: DataFrame containing the count of missing values for each column.
    - duplicate_rows: DataFrame containing duplicate rows, if any.
    """
    # Check for missing values
    missing_values = df.isnull().sum()

    # Check for duplicate rows
    duplicate_rows = df[df.duplicated()]

    return missing_values, duplicate_rows

check_table(df_user)
check_table(df_user_day)



# aggregate total time spent,convert to min and merge with user table
df_user_all = df_user.merge(
    df_user_day.groupby("userid", as_index=False)["time_spent"].sum(),
    on="userid",
    how="inner",
).assign(time_spent_mins=lambda x: x["time_spent"] / 60)

# check summary statistics (mean, std, count for each variant)
df_user_all.groupby("branch")["time_spent_mins"].agg({"mean", "std", "count"})



# Define a function to calculate the proportion of users who returned after installation
def calculate_returned_proportion(df_user, df_user_day):
    """
    Function to calculate the retention rate of users who returned after their installation date
    Parameters:
    - df: DataFrame containing the table data.
    Returns:
    - proportion:  retention rate for test and control
    """
    # Merge user data with their corresponding activity data
    merged_df = pd.merge(df_user, df_user_day, on="userid")

    # Create a mask for users who have played on days after their install date
    returned_mask = merged_df["ds"] > merged_df["install_date"]
    returned_users_df = merged_df[returned_mask]
    unique_returned_users = returned_users_df[["userid", "branch"]].drop_duplicates()

    # Count the number of returned users by branch
    returned_users_count = unique_returned_users["branch"].value_counts()
    # Calculate the total number of users in each branch from the original user dataframe
    total_users_by_branch = df_user["branch"].value_counts()

    # Calculate the proportion of returned users by dividing the returned user count by the total user count
    returned_users_proportion = returned_users_count / total_users_by_branch

    # Return the proportion of returned users for each branch
    return returned_users_proportion

retention_proportions = calculate_returned_proportion(df_user, df_user_day)
print(retention_proportions)



def calculate_average_daily_retention(df_user, df_user_day):
    """
    Calculate the average daily retention rates for the Control and Test groups.

    Parameters:
    - df_user (DataFrame): User dataframe with 'userid', 'install_date', and 'branch'.
    - df_user_day (DataFrame): User activity dataframe with 'userid', 'ds' (day of activity).

    Returns:
    - DataFrame: A dataframe containing the average daily retention rates by branch.
    """
    # Merge the user and activity dataframes
    merged_df = pd.merge(df_user, df_user_day, on="userid")

    # Calculate the number of days since installation
    merged_df["days_since_install"] = (
        merged_df["ds"] - merged_df["install_date"]
    ).dt.days

    # Remove entries with negative days since install
    merged_df = merged_df[merged_df["days_since_install"] >= 0]

    # Count unique active users per day since installation
    daily_active_users = (
        merged_df.groupby(["install_date", "branch", "days_since_install"])["userid"]
        .nunique()
        .reset_index(name="active_users")
    )

    # Count the initial users per day
    initial_users_per_day = (
        df_user.groupby(["install_date", "branch"])["userid"]
        .nunique()
        .reset_index(name="initial_users")
    )

    # Merge the active user counts with the initial user counts
    retention_data_daily = pd.merge(
        daily_active_users, initial_users_per_day, on=["install_date", "branch"]
    )

    # Calculate the daily retention rates
    retention_data_daily["retention_rate"] = (
        retention_data_daily["active_users"] / retention_data_daily["initial_users"]
    )

    # Calculate the average daily retention rate for each branch and day since installation
    average_daily_retention = (
        retention_data_daily.groupby(["branch", "days_since_install"])["retention_rate"]
        .mean()
        .reset_index()
    )

    return average_daily_retention

average_daily_retention = calculate_average_daily_retention(df_user, df_user_day)



# Plot the average daily retention rate per variant
import seaborn as sns

sns.lineplot(
    data=average_daily_retention,
    x="days_since_install",
    y="retention_rate",
    hue="branch",
)
plt.title("Average Daily Retention Rate: Control vs. Test")
plt.xlabel("Days Since Installation")
plt.ylabel("Average Retention Rate")
plt.show()



# Sanity Checks and Internal Validity
def run_chi_square_ratio_mismatch_test(
    dataframe, branch_col="branch", user_col="userid"
):
    """
     Performs a chi-square test to detect sample ratio mismatch between two groups.

    Returns:
     - chi2_stat: The chi-square statistic.
     - p_value: The p-value associated with the chi-square statistic.
     - expected: The expected frequencies based on the observed data.
    """
    # Count unique users in each group
    control_users = dataframe[dataframe[branch_col] == "Control"][user_col].nunique()
    test_users = dataframe[dataframe[branch_col] == "Test"][user_col].nunique()

    # Total users
    total_users = control_users + test_users

    # Observed counts
    observed = [control_users, test_users]

    # Expected counts assuming equal distribution
    expected = [total_users / 2, total_users / 2]

    chi = chisquare(observed, f_exp=expected)

    if chi[1] < 0.01:
        return "SRM detected", chi
    else:
        return "No SRM detected", chi

    return chi

run_chi_square_ratio_mismatch_test(df_user_all)



# Sanitiy Checks and external Validity
def check_simpsons_paradox(df, group_cols, value_col):
    """
    Check for evidence of Simpson's Paradox in our data.

    Parameters:
    - df (DataFrame): The dataframe to analyze.
    - group_cols (list): A list of column names to group by. The first should be 'branch'.
    - value_col (str): The name of the column containing the values to aggregate.

    Returns:
    - DataFrame: A dataframe containing the mean of the value column for each group.
    """
    # Calculate the overall mean by the main grouping factor (e.g., 'branch')
    overall_mean = df.groupby(group_cols[0])[value_col].mean().reset_index(name="mean")

    # Calculate the mean by the additional segments within the main grouping factor
    segmented_mean = df.groupby(group_cols)[value_col].mean().reset_index(name="mean")

    # Return both overall and segmented means
    return overall_mean, segmented_mean

overall_mean, segmented_mean_channel = check_simpsons_paradox(
    df_user_all, ["branch", "channel"], "time_spent_mins"
)

overall_mean, segmented_mean_surface = check_simpsons_paradox(
    df_user_all, ["branch", "surface"], "time_spent_mins"
)

overall_mean, segmented_mean_device = check_simpsons_paradox(
    df_user_all, ["branch", "device_simple"], "time_spent_mins"
)




def plot_group_novelty_effect(user, user_day, days_threshold):
    """
    Plot the trend for the Test group to check for a novelty effect in terms of time spent.

    Parameters:
    - df (DataFrame): The dataframe containing the time spent data.
    - days_threshold (int): The number of days to consider after the first experience of the changes.
    """
    df = pd.merge(user, user_day, on="userid")

    df["install_date"] = pd.to_datetime(df["install_date"])
    df["ds"] = pd.to_datetime(df["ds"])

    # Calculate the difference in days since the install date
    df["days_since_install"] = (df["ds"] - df["install_date"]).dt.days

    # Filter the DataFrame for the Test group
    test_df = df[df["branch"] == "Test"]

    # Further filter the data for the initial period after the first experience
    df = df[df["days_since_install"] <= days_threshold]

    # Calculate the mean time spent for each day
    df_trend = (
        df.groupby(["branch", "days_since_install"])["time_spent"].mean().reset_index()
    )

    # Plot the trend
    # Plotting
    sns.lineplot(data=df_trend, x="days_since_install", y="time_spent", hue="branch")
    plt.title("Daily Average Time Spent by Branch")
    plt.xlabel("Days Since Install")
    plt.ylabel("Average Time Spent (seconds)")
    plt.legend(title="Branch")
    plt.xticks(rotation=45)
    plt.show()
    # Example usage of the function:

plot_group_novelty_effect(df_user, df_user_day, 35)



#Check TT Test Assumptions
def visualize_distribution(data):
    """
    Visualize the distribution of time spent and presence of outliers by group.

    Parameters:
    - data: DataFrame containing the data to visualize.
    """
    # Visual check for normality of distribution 
    sns.histplot(data=data, x='time_spent_mins', hue='branch', element='step', stat='density', common_norm=False)
    plt.title('Distribution of Time Spent by Group')
    plt.show()

    # Boxplot to visualize outliers 
    sns.boxplot(data=data, x='branch', y='time_spent_mins')
    plt.title('Time Spent Boxplot by Group')
    plt.show()

visualize_distribution(df_user_all)




def detect_outliers(df, column):
    """
    Detect the presence of outliers using the IQR method
    Parameters:
    - data: Data showing number of outliers in each variant
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# Detect outliers in Control and Test groups
control_outliers = detect_outliers(df_user_all[df_user_all['branch'] == 'Control'], 'time_spent_mins')
test_outliers = detect_outliers(df_user_all[df_user_all['branch'] == 'Test'], 'time_spent_mins')

print(f"Number of outliers in Control group: {len(control_outliers)}")
print(f"Number of outliers in Test group: {len(test_outliers)}")

# Remove outliers for statistical test
filtered_df_user_all = df_user_all[~df_user_all.index.isin(control_outliers.index) & ~df_user_all.index.isin(test_outliers.index)]




def detect_outliers(df, column):
    """
    Detect the presence of outliers using the IQR method
    Parameters:
    - data: Data showing number of outliers in each variant
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# Detect outliers in Control and Test groups
control_outliers = detect_outliers(df_user_all[df_user_all['branch'] == 'Control'], 'time_spent_mins')
test_outliers = detect_outliers(df_user_all[df_user_all['branch'] == 'Test'], 'time_spent_mins')

print(f"Number of outliers in Control group: {len(control_outliers)}")
print(f"Number of outliers in Test group: {len(test_outliers)}")

# Remove outliers for statistical test
filtered_df_user_all = df_user_all[~df_user_all.index.isin(control_outliers.index) & ~df_user_all.index.isin(test_outliers.index)]
 
control_time_spent = filtered_df_user_all[filtered_df_user_all['branch'] == 'Control']['time_spent']
test_time_spent = filtered_df_user_all[filtered_df_user_all['branch'] == 'Test']['time_spent']



def check_normality_and_equal_variance(control, test):
    """
     Statistically perrform Shapiro-Wilk test for normality and Levene's Test for equal variances
    on Control and Test groups.

    Parameters:
    - control: Data for the Control group.
    - test: Data for the Test group.
    """
    def shapiro_test(data, group_name):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p = shapiro(data)
        print('%s group - Shapiro-Wilk Test: Statistics=%.3f, p=%.3f' % (group_name, stat, p))

    shapiro_test(control, 'Control')
    shapiro_test(test, 'Test')

    # Levene's Test for equal variances
    stat, p = levene(control, test)
    print('Levene’s Test: Statistics=%.3f, p=%.3f' % (stat, p))

check_normality_and_equal_variance(control_time_spent, test_time_spent)



def subgroup_mann_whitney(data, subgroup_var, control_var, test_var):
    """
    Perform Mann-Whitney U Test for subgroups within the data.

    Parameters:
    - data: DataFrame containing the data.
    - subgroup_var: Name of the column representing the subgroup variable.
    - control_var: Name of the column representing the control group .
    - test_var: Name of the column representing the test group .

    Returns:
    - subgroup_results: Dictionary containing Mann-Whitney U test results for each subgroup.
    """
    # Get unique subgroups
    subgroups = data[subgroup_var].unique()

    # Initialize dictionary to store results
    subgroup_results = {}

    # Iterate over each subgroup
    for subgroup in subgroups:
        subgroup_data = data[data[subgroup_var] == subgroup]
        control_data = subgroup_data[subgroup_data[control_var] == 'Control'][test_var]
        test_data = subgroup_data[subgroup_data[control_var] == 'Test'][test_var]

        # Perform Mann-Whitney U test
        stat, p = mannwhitneyu(control_data, test_data, alternative='greater')
        subgroup_results[subgroup] = {'statistic': stat, 'p_value': p}

    return subgroup_results

results = subgroup_mann_whitney(filtered_df_user_all, 'device_simple', 'branch', 'time_spent_mins')
results = subgroup_mann_whitney(filtered_df_user_all, 'channel', 'branch', 'time_spent_mins')
results = subgroup_mann_whitney(filtered_df_user_all, 'surface', 'branch', 'time_spent_mins')
