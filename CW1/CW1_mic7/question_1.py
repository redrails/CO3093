import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
Args:
    dataframe: A dataframe of type pandas.DataFrame
Returns:
    A clean version of the dataframe by removing all empty/NaN cols
Raises:
    TypeError: If wrong type of argument is specified
"""
def clean(dataframe):
    assert isinstance(dataframe, pd.DataFrame), 'Argument of wrong type!'
    return dataframe.dropna(axis=1, how='all')

"""
Filter out the columns that are required in a given dataframe to only keep the relevant information.
Args:
    dataframe: A dataframe of type pandas.DataFrame
    cols: A list of column names to be filtered out
Returns:
   A filtered version of the data frame as a data frame
Raises:
    TypeError: If wrong type of argument is specified
"""
def filterCols(dataframe, cols):
    assert isinstance(dataframe, pd.DataFrame), 'Argument of wrong type!'
    assert isinstance(cols, list), 'Argument of wrong type!'
    return dataframe[cols]

rteams = ["Man United", "Man City"] # Teams we care about
data = pd.read_csv("resources/PremierLeague1718.csv") # Read the data set file
data = clean(data) # Clean the data up to remove empty and NaN cols
colsToKeep = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "FTR"] # List of cols we want to keep
data = filterCols(data, colsToKeep) # Filter relevant cols

teams = {"ManUnitedHome": "", "ManCityHome": "", "ManUnitedAway": "", "ManCityAway": ""} # A dict to contain relevant team dataframes

teams["ManUnitedHome"] = data.loc[data["HomeTeam"] == "Man United"]
teams["ManUnitedAway"] = data.loc[data["AwayTeam"] == "Man United"]
teams["ManCityHome"] = data.loc[data["HomeTeam"] == "Man City"]
teams["ManCityAway"] = data.loc[data["AwayTeam"] == "Man City"]

"""
Used to summarise the data, relevant to question 1.1
"""
def summarisedata():
    print("Description:")
    print(data.describe())

    print("\nNumber of wins at home")
    print(data.groupby("HomeTeam")["FTR"].apply(lambda x: x[x == 'H'].count()))

    print("\nNumber of wins away")
    print(data.groupby("AwayTeam")["FTR"].apply(lambda x: x[x == 'A'].count()))

    print("\nNumber of losses at home")
    print(data.groupby("HomeTeam")["FTR"].apply(lambda x: x[x == 'A'].count()))

    print("\nNumber of losses away")
    print(data.groupby("AwayTeam")["FTR"].apply(lambda x: x[x == 'H'].count()))

"""
Information about the data from the two teams relevant to question 1.2
"""
def datainfo():
    print("Man United Home:")
    print(teams["ManUnitedHome"].describe())
    print("\nMan City Home")
    print(teams["ManCityHome"].describe())
    print("-------------")
    print("Man United Away:")
    print(teams["ManUnitedAway"].describe())
    print("\nMan City Away")
    print(teams["ManCityAway"].describe())



summarisedata()

# fig, ax = plt.subplots()
# index = np.arange(2)
# bar_width = 0.35
# opacity = 0.8
#
#
# rects1 = plt.bar(index, mutd_scores, bar_width,
#                  alpha=opacity,
#                  color='b',
#                  label='Manchester United')
#
# rects2 = plt.bar(index + bar_width, manc_scores, bar_width,
#                  alpha=opacity,
#                  color='g',
#                  label='Manchester City')
#
#
# plt.xlabel('Home/Away Team')
# plt.ylabel('Goals')
# plt.title('Scores by Team')
# plt.xticks((index + bar_width/2), ('Home Goals', 'Away Goals',))
# plt.legend()
#
# plt.tight_layout()
# print(data_mutd_home)
#
# plt.show()

# print(data.describe())

