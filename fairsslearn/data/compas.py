# https://github.com/propublica/compas-analysis

from pandas import read_csv, to_datetime, get_dummies, concat
from numpy import nanmedian, array
from sklearn.preprocessing import MinMaxScaler


def get_compas_data(path):

    df = read_csv(path)

    df = df.loc[df["days_b_screening_arrest"] <= 30, :]
    df = df.loc[df["days_b_screening_arrest"] >= -30, :]
    df = df.loc[df["is_recid"] != -1, :]
    df = df.loc[df["c_charge_degree"] != "O", :]
    df["c_jail_in"] = to_datetime(df["c_jail_in"])
    df["c_jail_out"] = to_datetime(df["c_jail_out"])
    df["c_offense_date"] = to_datetime(df["c_offense_date"])
    df["c_arrest_date"] = to_datetime(df["c_arrest_date"])
    df["c_jain_days"] = (df["c_jail_out"] - df["c_jail_in"]).dt.days
    df["c_offence_year"] = df["c_offense_date"].dt.year
    df["c_offence_month"] = df["c_offense_date"].dt.month
    df["c_arrest_year"] = df["c_arrest_date"].dt.year
    df["c_arrest_month"] = df["c_arrest_date"].dt.month

    df["race"] = df["race"].isin(["African-American"]) + 0

    df_dummies = get_dummies(df.loc[:, ["sex", "c_charge_degree"]])

    df = df.loc[:, [
        "age", 
        "race",
        "juv_fel_count", 
        "juv_misd_count", 
        "juv_other_count", 
        "priors_count", 
        "days_b_screening_arrest",
        "c_jain_days", 
        "c_offence_year", 
        "c_offence_month", 
        "c_arrest_year", 
        "c_arrest_month", 
        #"is_recid", 
        #"is_violent_recid",  
        "two_year_recid"
    ]]

    df["c_offence_missing"] = df["c_offence_year"].isnull() + 0
    df["c_arrest_missing"] = df["c_arrest_year"].isnull() + 0

    df["c_offence_year"] = df["c_offence_year"]\
        .fillna(nanmedian(df["c_offence_year"]))
    df["c_offence_month"] = df["c_offence_month"]\
        .fillna(nanmedian(df["c_offence_month"]))
    df["c_arrest_year"] = df["c_arrest_year"]\
        .fillna(nanmedian(df["c_arrest_year"]))
    df["c_arrest_month"] = df["c_arrest_month"]\
        .fillna(nanmedian(df["c_arrest_month"]))

    df = concat((df_dummies, df), axis=1)

    X = array(df.loc[:, df.columns != "two_year_recid"])
    y = array(df["two_year_recid"])
    s = array(df["race"])

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return X, y, s