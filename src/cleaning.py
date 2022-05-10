import pandas as pd

def normalize_missing(df, columns):
    """
    Returns a copy of the input dataframe,
    where known patterns for missing values are replaced with `None`.
    """
    
    missing_patterns = [
        'invalid!',
        'no data!',
        'nodata!',
        'not defined',
        'unknown',
        'sap05',
    ]
    
    new_df = df.copy()
    
    new_df[columns] = new_df[columns].applymap(
        lambda x: None if any([str(x).lower().startswith(pattern) for pattern in missing_patterns]) else x)
    
    return new_df


def clean_age_band(df):
    """Returns a copy of the input dataframe,
    with a cleaned version of the column CONSTRUCTION_AGE_BAND"""

    df = df.copy()
   
    def _clean(x):
        x = str(x).replace("England and Wales: ", "")

        # cannot trust dates after 2007, as there is a collective category "2007 onwards"
        if x in ["2012 onwards", "2007-2011"]:
            return "2007 onwards"
        try:
            x = int(x)
        except ValueError:
            if not isinstance(x, str):
                print(x, type(x))
            return x

        if x < 1900:
            return "before 1900"
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
        ~df["CONSTRUCTION_AGE_BAND"].isnull(), "CONSTRUCTION_AGE_BAND"].apply(_clean)
    return df


def clean_floor_level(df):
    """Returns a copy of the input dataframe,
    with a cleaned version of the column FLOOR_LEVEL"""
    
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
    
    df["FLOOR_LEVEL"] = df["FLOOR_LEVEL"].str.strip().str.lower().replace(
        {"ground": 0, "ground floor": 0, "-1": "basement"}).apply(_clean)
    
    
    return df