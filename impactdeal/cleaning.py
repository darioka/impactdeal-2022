import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .config.column_names import NUMERICAL, CATEGORICAL, DATES


def normalize_missing(df, columns):
    """
    Returns a copy of the input dataframe,
    where known patterns for missing values are replaced with `np.nan`.

    Parameters
    ----------

    df (pd.DataFrame): input dataframe

    columns (list): name of the columns where missing values will be checked

    Returns
    -------

    df (pd.DataFrame)
    """

    missing_patterns = [
        "invalid!",
        "no data!",
        "nodata!",
        "not defined",
        "unknown",
        "sap05",
    ]

    new_df = df.copy()

    new_df[columns] = new_df[columns].applymap(
        lambda x: np.nan
        if any([str(x).lower().startswith(pattern) for pattern in missing_patterns])
        else x
    )

    return new_df


def clean_age_band(df):
    """Returns a copy of the input dataframe,
    with a cleaned version of the column CONSTRUCTION_AGE_BAND

    Parameters
    ----------

    df (pd.DataFrame): input dataframe

    Returns
    -------

    df (pd.DataFrame)
    """

    df = df.copy()

    def _clean(x):
        x = str(x).replace("England and Wales: ", "")

        # cannot trust dates after 2007, as there is a collective category "2007 onwards"
        if x in ["2012 onwards", "2007-2011"]:
            return "2007 onwards"
        elif x == "before 1900":
            return "1899 and earlier"

        try:
            x = int(x)
        except ValueError:
            if not isinstance(x, str):
                print(x, type(x))
            return x

        if x < 1900:
            return "1899 and earlier"
        if x < 1930:
            return "1900-1929"
        if x < 1950:
            return "1930-1949"
        if x < 1967:
            return "1950-1966"
        if x < 1976:
            return "1967-1975"
        if x < 1982:
            return "1976-1982"
        if x < 1991:
            return "1983-1990"
        if x < 1996:
            return "1991-1995"
        if x < 2003:
            return "1996-2002"
        if x < 2007:
            return "2003-2006"
        else:
            return "2007 onwards"

    df.loc[~df["CONSTRUCTION_AGE_BAND"].isnull(), "CONSTRUCTION_AGE_BAND"] = df.loc[
        ~df["CONSTRUCTION_AGE_BAND"].isnull(), "CONSTRUCTION_AGE_BAND"
    ].apply(_clean)
    return df


def clean_floor_level(df):
    """Returns a copy of the input dataframe,
    with a cleaned version of the column FLOOR_LEVEL
    
    Parameters
    ----------

    df (pd.DataFrame): input dataframe

    Returns
    -------

    df (pd.DataFrame)
    """

    def _clean(x):
        if x == "21st or above":
            return "above 20th"
        try:
            x = int(str(x)[:2])
        except ValueError:
            return x

        x = str(x)
        if x in ["0", "-1"]:
            return x
        elif x.endswith("1"):
            return x + "st"
        elif x.endswith("2"):
            return x + "nd"
        elif x.endswith("3"):
            return x + "rd"
        else:
            return x + "th"

    df = df.copy()

    df["FLOOR_LEVEL"] = (
        df["FLOOR_LEVEL"]
        .str.strip()
        .str.lower()
        .replace({"ground": 0, "ground floor": 0, "-1": "basement"})
        .apply(_clean)
    )

    return df


def clean_mainheat(df):
    """Returns a copy of the input dataframe,
    with a cleaned version of the column MAIN_HEATING_CONTROLS
    
    Parameters
    ----------

    df (pd.DataFrame): input dataframe

    Returns
    -------

    df (pd.DataFrame)
    """

    def _clean(x):
        try:
            return str(int(x))
        except ValueError:
            return np.nan

    df.copy()
    df["MAIN_HEATING_CONTROLS"] = df["MAIN_HEATING_CONTROLS"].apply(_clean)

    return df


class Cleaner(TransformerMixin, BaseEstimator):
    """
    Clean the EPC rating dataset

    The input to this transformer must be a pandas.DataFrame.

    Parameters
    ----------

    text_features: bool, default=False
        Whether to keep text columns
    
    missing_threshold: float, default=0.7
        Threshold of percentage of missing values in each feature
        beyond which the feature will be discarderd.

    Attributes
    ----------

    feature_names_in_: list of strings
        Names of the input columns
    
    feature_names_out_: list of strings
        Names of the output columns
    """

    _numerical_features = NUMERICAL
    _categorical_features = [
        x for x in CATEGORICAL if not x.lower().endswith("description")
    ]
    _text_features = [x for x in CATEGORICAL if x.lower().endswith("description")]

    def __init__(self, text_features=False, missing_threshold=0.7):
        if text_features in [True, False]:
            self.text_features = text_features
        else:
            raise ValueError("text_features must be either True or False")

        if (
            np.asarray(missing_threshold).dtype.kind in ["i", "f"]
            and missing_threshold > 0
            and missing_threshold <= 1
        ):
            self.missing_threshold = missing_threshold
        else:
            raise ValueError("missing threshold must be 0 < x <= 1")

    def fit(self, X, y=None):
        """
        Fit the Cleaner to X. Compute percentages of missing values
        and choose the columns to keep.

        Parameters
        ----------

        X : pd.DataFrame

        y: None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.
        """
        X = self._check_X(X)

        X = normalize_missing(X, self._categorical_features)
        if self.text_features:
            X = normalize_missing(X, self._text_features)

        # drop where too many missing
        columns_to_drop = [
            i for i, x in X.isnull().mean().iteritems() if x > self.missing_threshold
        ]
        # drop variables with informative content similar to other features
        columns_to_drop += [
            "LOW_ENERGY_FIXED_LIGHT_COUNT",
        ]

        self.feature_names_in_ = X.columns.tolist()
        self.feature_names_out_ = [x for x in X.columns if x not in columns_to_drop]

        return self

    def _check_X(self, X):
        # accept dataframes only
        if not (hasattr(X, "iloc") and getattr(X, "ndim", 0) == 2):
            raise ValueError("X must be a pandas dataframe")

        all_features = self._numerical_features + self._categorical_features
        if self.text_features:
            all_features += self._text_features

        try:
            cols = X[all_features].columns
        except KeyError:
            raise KeyError("X must have all NUMERICAL and CATEGORICAL features")

        if len(cols) != len(all_features):
            raise KeyError("X has duplicated columns")

        return X.loc[:, all_features]

    def transform(self, X, y=None):
        """
        Apply custom cleaning functions to specific columns.
        Filter out columns with too many missings.

        Parameters
        ----------

        X : pd.DataFrame

        y: None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.
        """

        check_is_fitted(self, ["feature_names_out_"])

        X = self._check_X(X)
        X = normalize_missing(X, self._categorical_features)
        X = clean_age_band(X)
        X = clean_floor_level(X)
        X = clean_mainheat(X)
        X = X.loc[:, self.feature_names_out_]

        return X
