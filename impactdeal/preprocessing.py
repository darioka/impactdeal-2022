import numpy as np
import pandas as pd
from warnings import warn

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_X_y, check_array
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import make_pipeline


class CategoryReducer(TransformerMixin, BaseEstimator):
    """
    Reduce the number of categories, replacing all instances
    of rare classes with an encoded value.

    The input to this transformer must be a pandas.DataFrame.
    All columns will be interpreted as and converted to string.

    Parameters
    ----------

    size: int or float, default=10
        Minimum number of samples for each category (if int)
        or minimum frequency for each category (if float)
    
    encoded_value: str or np.nan
        The replacement value for classes with n_samples < size
    
    Attributes
    ----------

    value_counts_: dict of pandas DataFrames
        Dataframes with count and frequencies for each feature, computed
        during ``fit`` from the input dataframe (in order of
        the features in X and corresponding with the output of ``transform``).
    """

    def __init__(self, size=10, encoded_value=np.nan):

        self._validate_size(size)
        self.size = size

        self.encoded_value = encoded_value

    def _validate_size(self, size):

        size_dtype = np.asarray(size).dtype.kind

        if (size_dtype == "i" and size <= 0) or (
            size_dtype == "f" and (size <= 0 or size >= 1)
        ):
            raise ValueError(
                "size={0} should be either positive or a float in the "
                "(0, 1) range".format(size)
            )

        if size_dtype not in ("i", "f"):
            raise ValueError("Invalid value for size: {}".format(size))

    def fit(self, X, y=None):
        """
        Fit the CategoryReducer to X, computing counts and frequencies.

        Parameters
        ----------

        X : pd.DataFrame

        y: None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.
        """

        X = self._check_X(X)

        n_samples, n_features = X.shape

        self.value_counts_ = {
            col: X[col].value_counts().reset_index().rename(columns={col: "count"})
            for col in X
        }
        for col, df in self.value_counts_.items():
            df["freq"] = df["count"] / n_samples
            df.set_index("index", drop=True, inplace=True)

        return self

    def _check_X(self, X):

        # accept dataframes only
        if not (hasattr(X, "iloc") and getattr(X, "ndim", 0) == 2):
            raise ValueError("X must be a pandas dataframe")

        return pd.DataFrame(
            check_array(X, dtype="object", force_all_finite=False),
            columns=X.columns,
            index=X.index,
        )

    def transform(self, X, y=None):
        """
        Encoded all rare categories in X with the encoded value.

        Parameters
        ----------

        X : pd.DataFrame

        y: None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.
        """

        check_is_fitted(self, ["value_counts_"])

        X = self._check_X(X)

        threshold_type = "count" if np.asarray(self.size).dtype.kind == "i" else "freq"
        for col in self.value_counts_:
            mapping = X[col].map(self.value_counts_[col][threshold_type])
            X[col] = X[col].mask(mapping < self.size, self.encoded_value)

        return X


class RoomsImputer(TransformerMixin, BaseEstimator):
    """
    Use linear regressions on TOTAL_FLOOR_AREA to input missing values in
    NUMBER_HABITABLE_ROOMS and NUMBER_HEATED_ROOMS.

    The input to this transformer must be a pandas.DataFrame.
    Output is a dataframe with NUMBER_HABITABLE_ROOMS and NUMBER_HEATED_ROOMS.
    
    Attributes
    ----------

    feature_names_in_: list of strings
        ["TOTAL_FLOOR_AREA", "NUMBER_HABITABLE_ROOMS", "NUMBER_HEATED_ROOMS"]
    
    feature_names_out_: list of strings
        ["NUMBER_HABITABLE_ROOMS", "NUMBER_HEATED_ROOMS"]
    
    imputers_: dict
        Contains the sklearn models for each feature
    """

    _x_cols = ["TOTAL_FLOOR_AREA"]
    _y_cols = ["NUMBER_HABITABLE_ROOMS", "NUMBER_HEATED_ROOMS", "FIXED_LIGHTING_OUTLETS_COUNT"]

    def fit(self, X, y=None):
        """
        Fit multiple linear regressions on TOTAL_FLOOR_AREA to
        predict NUMBER_HABITABLE_ROOMS and NUMBER_HEATED_ROOMS,
        then compute cross-validated scores.

        The model is a linear regression with log-transformed target
        and power-transformed feature.

        Parameters
        ----------

        X : pd.DataFrame

        y: None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.
        """
        x, ys = self._check_X(X)
        self.feature_names_in_ = self._x_cols + self._y_cols
        self.feature_names_out_ = self._y_cols

        self.imputers_ = {}
        self.mean_absolute_error_ = {}
        for i, y_col in enumerate(self._y_cols):
            pt = PowerTransformer()
            tt = TransformedTargetRegressor(
                regressor=LinearRegression(),
                func=lambda x: np.log(1 + x),
                inverse_func=lambda x: np.exp(x) - 1,
            )
            pipeline = make_pipeline(pt, tt)

            pipeline.fit(x, ys[:, i])
            self.imputers_[y_col] = pipeline

    def _check_X(self, X, dropna=True):
        # accept dataframes only
        if not (hasattr(X, "iloc") and getattr(X, "ndim", 0) == 2):
            raise ValueError("X must be a pandas dataframe with 3 columns")

        try:
            x = X.loc[:, self._x_cols + self._y_cols]
        except KeyError:
            raise ValueError(
                f"X must have the following columns: {self._x_cols + self._y_cols}"
            )

        if dropna:
            x = x.dropna()
        ys = x.loc[:, self._y_cols]
        x = x.drop(columns=self._y_cols)

        x = check_array(x, dtype="numeric", force_all_finite=dropna)
        ys = np.stack(
            [
                check_array(
                    ys[col], force_all_finite=dropna, ensure_2d=False, dtype="numeric"
                )
                for col in ys
            ],
            axis=-1,
        )
        return x, ys

    def transform(self, X, y=None):
        """
        Replace missing values in X with predictions from the fitted models.

        Output only ["NUMBER_HABITABLE_ROOMS", "NUMBER_HEATED_ROOMS"]
        independently from the additional columns that may be present in X.

        Parameters
        ----------

        X : pd.DataFrame

        y: None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.
        """
        check_is_fitted(self, ["imputers_"])
        x, ys = self._check_X(X, dropna=False)

        ys_pred = pd.DataFrame(ys, index=X.index, columns=self._y_cols)
        for i, y_col in enumerate(self._y_cols):
            na_indeces = ys_pred[y_col].isnull()
            imputed_y = self.imputers_[y_col].predict(x[na_indeces]).round()
            imputed_y[imputed_y < 0] = 0
            ys_pred.loc[na_indeces, y_col] = imputed_y

        return ys_pred
