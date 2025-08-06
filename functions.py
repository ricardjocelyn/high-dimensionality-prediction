import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


def count_groups(data, id_col="Subject ID", group_col="Group"):
    """
    Count how many Demented and Nondemented participants there are.

    Parameters:
    - data (pd.DataFrame): Your dataset.
    - id_col (str): Column representing participant ID.
    - group_col (str): Column representing group ('Demented', 'Nondemented').

    Returns:
    - pd.Series with counts for each group.
    """

    # Drop duplicates to count each participant once
    unique_participants = data[[id_col, group_col]].drop_duplicates()

    # Count by group
    group_counts = unique_participants[group_col].value_counts()

    return group_counts


def session_counts_by_group(
    df, subject_col="Subject ID", group_col="Group", session_col="Visit"
):
    """
    Count how many participants in each group have how many sessions.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - subject_col (str): Column with participant IDs.
    - group_col (str): Column with patient group labels (e.g., 'CN', 'MCI', 'AD').
    - session_col (str): Column with session or visit number.

    Returns:
    - pd.DataFrame with the count of participants per group by number of sessions.
    """

    # Count number of sessions per participant
    session_counts = (
        df.groupby(subject_col)[session_col].nunique().reset_index(name="n_sessions")
    )

    # Merge back the group info (assuming group is constant per participant)
    session_counts = session_counts.merge(
        df[[subject_col, group_col]].drop_duplicates(), on=subject_col, how="left"
    )

    # Count how many participants in each group have N sessions
    summary = (
        session_counts.groupby([group_col, "n_sessions"])
        .size()
        .reset_index(name="count")
    )

    return summary.sort_values([group_col, "n_sessions"])


def box_plot(data, value_col, group_col, title, xlabel, ylabel):
    """
    Create a boxplot for two distinct groups.

    Parameters:
    - data (pd.DataFrame): The dataset containing the values and group labels.
    - value_col (str): Name of the column containing the values to plot.
    - group_col (str): Name of the column containing the group labels.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    """
    # Ensure only two groups are present
    unique_groups = data[group_col].unique()
    if len(unique_groups) != 3:
        raise ValueError(
            f"Expected 3 groups, but found {len(unique_groups)}: {unique_groups}"
        )

    sns.boxplot(x=group_col, y=value_col, data=data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def line_plot(df, value_col, group_col, title, xlabel="Visit Number"):

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Visit", y=value_col, hue=group_col, marker="o", alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(f"{group_col} Score")
    plt.legend(
        title="Subject ID", bbox_to_anchor=(1.05, 1), loc="upper left"
    )  # move legend outside
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def one_way_anova(df, value_col, group_col="Group"):
    """
    Perform a one-way ANOVA test to compare means across multiple groups.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - value_col (str): Name of the column with numerical values.
    - group_col (str): Name of the column with group labels.

    Returns:
    - f_stat (float): The computed F-statistic.
    - p_value (float): The p-value for the test.
    """

    # Validate input
    if value_col not in df.columns or group_col not in df.columns:
        raise ValueError(
            f"Columns '{value_col}' and/or '{group_col}' not found in DataFrame."
        )

    # Extract unique groups
    groups = df[group_col].unique()

    # Collect values for each group
    group_data = [df[df[group_col] == group][value_col].dropna() for group in groups]

    # Perform ANOVA
    f_stat, p_value = f_oneway(*group_data)

    return f_stat, p_value


def tukey_posthoc(df, value_col, group_col="Group", alpha=0.05):
    """
    Perform Tukey's HSD post-hoc test after ANOVA to compare all group pairs.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - value_col (str): Column name with numeric values (e.g., MMSE scores).
    - group_col (str): Column name with group labels (e.g., 'Group').
    - alpha (float): Significance level (default = 0.05).

    Returns:
    - Summary table of pairwise group comparisons.
    """

    # Drop missing values
    data = df[[value_col, group_col]].dropna()

    # Run Tukey HSD test
    tukey = pairwise_tukeyhsd(
        endog=data[value_col], groups=data[group_col], alpha=alpha
    )

    return tukey.summary()


def predict_visit2_mmse(
    df,
    features_visit1,
    target_visit2,
    subject_col="Subject ID",
    visit_col="Visit",
    kernel="rbf",
    C=1.0,
    epsilon=0.1,
    test_size=0.2,
    random_state=42,
    plot=True,
):
    """
    Predict Visit 2 MMSE using features from Visit 1 with SVR and plot results.

    Splits data into train/test before scaling and training.

    Returns:
    - svr (SVR): Trained model
    - scaler (StandardScaler): Scaler used
    - merged (pd.DataFrame): Merged input dataset
    - metrics (dict): Dictionary with train/test MSE and R²
    """
    # Split data into visit 1 and 2
    df1 = df[df[visit_col] == 1]
    df2 = df[df[visit_col] == 2]

    # Merge visit 1 features with visit 2 target
    merged = pd.merge(
        df1[[subject_col] + features_visit1],
        df2[[subject_col, target_visit2]],
        on=subject_col,
        how="inner",
        suffixes=("_visit1", "_visit2"),
    ).dropna()

    # Rename features and target
    features_renamed = [
        f + "_visit1" if f == target_visit2 else f for f in features_visit1
    ]
    target_renamed = target_visit2 + "_visit2"

    X = merged[features_renamed]
    y = merged[target_renamed]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVR
    svr = SVR(kernel=kernel, C=C, epsilon=epsilon)
    svr.fit(X_train_scaled, y_train)

    # Predict
    y_train_pred = svr.predict(X_train_scaled)
    y_test_pred = svr.predict(X_test_scaled)

    # Metrics
    metrics = {
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "train_r2": r2_score(y_train, y_train_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "test_r2": r2_score(y_test, y_test_pred),
    }

    # Plotting
    if plot:
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_test_pred, alpha=0.7, edgecolors="k")
        plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
        plt.xlabel("Actual MMSE at Visit 2 (Test Set)")
        plt.ylabel("Predicted MMSE at Visit 2")
        plt.title(
            f"SVR on Test Set\nMSE={metrics['test_mse']:.2f}, R²={metrics['test_r2']:.2f}"
        )
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return svr, scaler, merged, metrics
