"""Processors for the data cleaning step of the worklow.

The processors in this step, apply the various cleaning steps identified
during EDA to create the training datasets.
"""
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow

from ta_lib.core.api import (
    load_dataset,
    register_processor,
    save_dataset,
)


@register_processor("data_cleaning", "sales")
def clean_sales_table(context, params):
    """Clean the ``sales`` data table.

    The table containts the sales data and has information
    on the sales value on product level etc.
    """

    input_dataset = "raw/sales"
    output_dataset = "cleaned/sales"

    # load dataset
    sales_df = load_dataset(context, input_dataset)

    sales_df_clean = (
        sales_df.copy()
        # set dtypes
        .change_type(
            ["sales_dollars_value", "sales_units_value", "sales_lbs_value"], np.float64
        )
        .change_type("product_id", np.int64)
        # set dtypes
        .to_datetime("system_calendar_key_N", format="%Y%m%d")
        .to_datetime("system_calendar_key_N", format="%d-%m-%y", dayfirst=True)
        # clean column names
        .rename_columns({"system_calendar_key_N": "date"})
        .clean_names(case_type="snake")
    )

    # save dataset
    save_dataset(context, sales_df_clean, output_dataset)
    return sales_df_clean


@register_processor("data_cleaning", "social_media")
def clean_social_media_table(context, params):
    """Clean the ``social_media`` data table.

    The table containts the social media data and has information
      on the number of posts on theme level.
    """

    input_dataset = "raw/social_media"
    output_dataset = "cleaned/social_media"

    # load dataset
    social_media_df = load_dataset(context, input_dataset)

    social_media_df_clean = (
        social_media_df.copy()
        # set dtypes
        .change_type("total_post", np.int64)
        # set dtypes
        .to_datetime("published_date", dayfirst=True)
        # clean column names
        .rename_columns({"published_date": "date", "Theme Id": "claim_id"}).clean_names(
            case_type="snake"
        )
    )

    # save dataset
    save_dataset(context, social_media_df_clean, output_dataset)
    return social_media_df_clean


@register_processor("data_cleaning", "google_search")
def clean_google_search_table(context, params):
    """Clean the ``google_search`` data table.

    The table containts the google search data and has information
      on the search volume on theme level etc.
    """

    input_dataset = "raw/google_search"
    output_dataset = "cleaned/google_search"

    # load dataset
    google_search_df = load_dataset(context, input_dataset)

    google_search_df_clean = (
        google_search_df.copy()
        # set dtypes
        .change_type(["searchVolume", "week_number", "year_new"], np.int64)
        # set dtypes
        .to_datetime("date", dayfirst=True)
        # clean column names
        .rename_columns({"year_new": "year"}).clean_names(case_type="snake")
    )

    # save dataset
    save_dataset(context, google_search_df_clean, output_dataset)
    return google_search_df_clean


@register_processor("data_cleaning", "product_manufacturer")
def clean_product_manufacturer_table(context, params):
    """Clean the ``product_manufacturer`` data table.

    The table containts the product's manufacturer data
    and has information on the vendor of the product.
    """

    input_dataset = "raw/product_manufacturer"
    output_dataset = "cleaned/product_manufacturer"

    # load dataset
    product_manufacturer = load_dataset(context, input_dataset)

    product_manufacturer_clean = (
        product_manufacturer.copy()
        # clean column names
        .clean_names(case_type="snake")
    )

    # save dataset
    save_dataset(context, product_manufacturer_clean, output_dataset)
    return product_manufacturer_clean


@register_processor("data_cleaning", "Theme_list")
def clean_Theme_list_table(context, params):
    """Clean the ``Theme_list`` data table.

    The table containts the Theme data and has information on the Theme ID and its name.
    """

    input_dataset = "raw/Theme"
    output_dataset = "cleaned/Theme"

    # load dataset
    Theme_list = load_dataset(context, input_dataset)

    Theme_list_clean = (
        Theme_list.copy()
        # clean column names
        .clean_names(case_type="snake")
    )

    # save dataset
    save_dataset(context, Theme_list_clean, output_dataset)
    return Theme_list_clean


@register_processor("data_cleaning", "Theme_product_list")
def clean_Theme_product_list_table(context, params):
    """Clean the ``Theme_product_list`` data table.

    The table containts the Theme product data and has information
    on which product falls under which theme.
    """

    input_dataset = "raw/Theme_product"
    output_dataset = "cleaned/Theme_product"

    # load dataset
    Theme_product_list = load_dataset(context, input_dataset)

    Theme_product_list_clean = (
        Theme_product_list.copy()
        # clean column names
        .clean_names(case_type="snake")
    )

    # save dataset
    save_dataset(context, Theme_product_list_clean, output_dataset)
    return Theme_product_list_clean


@register_processor("data_cleaning", "sales_with_theme_product")
def clean_sales_with_theme_product_table(context, params):
    """Clean the ``sales_with_theme_product`` data table.

    The table is a summary table obtained by doing a ``inner`` join
    of all the other tables, based on certain key columns and creating
    some business intution features.
    """
    input_sales_ds = "/cleaned/sales"
    input_social_media_ds = "/cleaned/social_media"
    input_google_search_ds = "/cleaned/google_search"
    input_product_manufacturer_ds = "/cleaned/product_manufacturer"
    input_Theme_list_ds = "/cleaned/Theme"
    input_Theme_product_list_ds = "/cleaned/Theme_product"
    output_dataset = "cleaned/sales_with_theme_product"

    # load datasets
    sales_df_clean = load_dataset(context, input_sales_ds)
    social_media_df_clean = load_dataset(context, input_social_media_ds)
    google_search_df_clean = load_dataset(context, input_google_search_ds)
    product_manufacturer_clean = load_dataset(context, input_product_manufacturer_ds)
    Theme_list_clean = load_dataset(context, input_Theme_list_ds)
    Theme_product_list_clean = load_dataset(context, input_Theme_product_list_ds)

    # Merging sales data with product manufacturer data to get Vendor
    sales_with_product_manf = sales_df_clean.merge(
        product_manufacturer_clean, on="product_id", how="inner"
    )

    # Merging sales data with theme product list to get the claim id
    sales_with_theme_product = sales_with_product_manf.merge(
        Theme_product_list_clean, on="product_id", how="inner"
    )

    # Merging sales data with theme list to get the claim name
    sales_with_theme_product = sales_with_theme_product.merge(
        Theme_list_clean, on="claim_id", how="inner"
    )

    # Creating Business Inutution Features
    # Unit Price
    sales_with_theme_product["unit_price"] = (
        sales_with_theme_product["sales_dollars_value"]
        / sales_with_theme_product["sales_units_value"]
    )

    # Price Per Pound
    sales_with_theme_product["price_per_pound"] = (
        sales_with_theme_product["sales_dollars_value"]
        / sales_with_theme_product["sales_lbs_value"]
    )

    # Year
    sales_with_theme_product["year"] = sales_with_theme_product["date"].dt.year
    social_media_df_clean["year"] = social_media_df_clean["date"].dt.year

    # Week Number
    sales_with_theme_product["week_number"] = (
        sales_with_theme_product["date"].dt.isocalendar().week
    )
    social_media_df_clean["week_number"] = (
        social_media_df_clean["date"].dt.isocalendar().week
    )

    # Imputing missing values in social media for claim_id
    # by taking mode on month-year level
    # Getting month-year column from date
    social_media_df_clean["month_year"] = social_media_df_clean["date"].dt.to_period(
        "M"
    )

    # Calculate the mode of "claim_id" grouped by "month_year"
    mode_theme_id = social_media_df_clean.groupby("month_year")["claim_id"].apply(
        lambda x: x.mode()[0]
    )

    # Fill the null values in "claim_id" with the corresponding mode values
    social_media_df_clean["claim_id"].fillna(
        social_media_df_clean["month_year"].map(mode_theme_id), inplace=True
    )

    # Getting social media data on weekly level by aggregating total_post
    social_media_df_clean["weekly_total_post"] = social_media_df_clean.groupby(
        ["claim_id", "week_number", "year"]
    )["total_post"].transform("sum")
    social_media_df_clean.drop(
        ["date", "total_post", "month_year"], axis=1, inplace=True
    )
    social_media_df_clean = social_media_df_clean.drop_duplicates()

    # Getting google search data on weekly level by aggregating search_volume
    google_search_df_clean["weekly_search_volume"] = google_search_df_clean.groupby(
        ["claim_id", "week_number", "year"]
    )["search_volume"].transform("sum")
    google_search_df_clean.drop(
        ["date", "search_volume", "platform"], axis=1, inplace=True
    )
    google_search_df_clean = google_search_df_clean.drop_duplicates()

    # Ranking claim id(s)
    sales_with_theme_product["rank_by_claim"] = sales_with_theme_product.groupby(
        [
            "date",
            "sales_dollars_value",
            "sales_units_value",
            "sales_lbs_value",
            "product_id",
            "vendor",
            "unit_price",
            "price_per_pound",
        ]
    )["claim_id"].rank(ascending=False)

    # Filtering sales data for claim id(s) having rank of 1
    sales_with_theme_product = sales_with_theme_product[
        sales_with_theme_product["rank_by_claim"] == 1.0
    ]
    sales_with_theme_product = sales_with_theme_product.drop("rank_by_claim", axis=1)

    # Filtering out the data with 0 sales value
    sales_with_theme_product = sales_with_theme_product[
        sales_with_theme_product["sales_dollars_value"] != 0
    ]
    sales_with_theme_product = sales_with_theme_product[
        sales_with_theme_product["sales_lbs_value"] != 0
    ]

    # Merging all the datasets to get a single dataset
    # Merging sales data with social media data
    final_data = sales_with_theme_product.merge(
        social_media_df_clean, on=["claim_id", "year", "week_number"], how="inner"
    )

    # Merging sales data with google search data
    final_data = final_data.merge(
        google_search_df_clean, on=["claim_id", "year", "week_number"], how="inner"
    )

    # Getting multiplier for weighted weekly post and weighted weekly search volume
    final_data["claim_total_sales_units"] = final_data.groupby(["claim_id"])[
        "sales_units_value"
    ].transform("sum")
    final_data["weighted_sales_by_product"] = (
        final_data["sales_units_value"] / final_data["claim_total_sales_units"]
    )

    # Calculating weighted weekly post and weighted weekly search volume
    final_data["weekly_total_post"] = (
        final_data["weekly_total_post"] * final_data["weighted_sales_by_product"]
    )
    final_data["weekly_search_volume"] = (
        final_data["weekly_search_volume"] * final_data["weighted_sales_by_product"]
    )

    # Dropping unncessary columns
    final_data.drop(
        [
            "date",
            "product_id",
            "claim_total_sales_units",
            "weighted_sales_by_product",
            "sales_dollars_value",
            "sales_lbs_value",
            "claim_id",
        ],
        axis=1,
        inplace=True,
    )

    # Convert "week_number" column to int64 data type
    final_data["week_number"] = final_data["week_number"].astype("int64")

    # Filtering data only for Vendor A and low carb theme
    final_data = final_data[
        (final_data["vendor"] == "A") & (final_data["claim_name"] == "low carb")
    ]
    final_data.drop(["vendor", "claim_name"], axis=1, inplace=True)

    save_dataset(context, final_data, output_dataset)
    return final_data


@register_processor("data_cleaning", "train-test")
def create_training_datasets(context, params):
    """Split the ``SALES`` table into ``train`` and ``test`` datasets."""

    input_dataset = "cleaned/sales_with_theme_product"
    output_train_features = "train/sales/features"
    output_train_target = "train/sales/target"
    output_test_features = "test/sales/features"
    output_test_target = "test/sales/target"

    # load dataset
    sales_df_processed = load_dataset(context, input_dataset)

    # split the data
    X = sales_df_processed.drop(params["target"], axis=1)
    y = sales_df_processed[params["target"]]

    # split test dataset into features and target
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=params["test_size"], random_state=0
    )

    # save the train dataset
    save_dataset(context, train_X, output_train_features)
    save_dataset(context, train_y, output_train_target)

    # save the test datasets
    save_dataset(context, test_X, output_test_features)
    save_dataset(context, test_y, output_test_target)

    # Configuring mlflow
    mlflow.set_experiment("Sales Prediction")
    with mlflow.start_run(run_name="Data Cleaning"):
        mlflow.log_params(params)
