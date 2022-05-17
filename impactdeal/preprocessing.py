import numpy as np
import pandas as pd
from warnings import warn

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array


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

        if (size_dtype == "i" and size <= 0) or (size_dtype == "f" and (size <= 0 or size >= 1)):
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

        self.value_counts_ = {col: X[col].value_counts().reset_index().rename(columns={col: "count"}) for col in X}
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
            columns = X.columns, index=X.index
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
