import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import poisson, skellam
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

def mergeteamdata():
    df = teams["ManUnitedHome"].append(teams["ManUnitedAway"])
    df = df.append((teams["ManCityHome"].append(teams["ManCityAway"])))
    return df

"""
Renaming columns given a dict and a dataframe
"""
def renameCols(df, colsdict):
    return df.reset_index().rename(columns=colsdict)

"""
Used to summarise the data, relevant to question 1.1
"""
def summarisedata():
    print("Description:")
    print(data.describe())

    groups = [0,0,0,0]
    groups[0] = renameCols(data.groupby("HomeTeam")["FTR"].apply(lambda x: x[x == 'H'].count()), {"HomeTeam": "Team", "FTR": "Wins at home"})
    groups[1] = renameCols(data.groupby("AwayTeam")["FTR"].apply(lambda x: x[x == 'A'].count()), {"AwayTeam": "Team", "FTR": "Wins away"})
    groups[2] = renameCols(data.groupby("HomeTeam")["FTR"].apply(lambda x: x[x == 'A'].count()), {"HomeTeam": "Team", "FTR": "Losses at home"})
    groups[3] = renameCols(data.groupby("AwayTeam")["FTR"].apply(lambda x: x[x == 'H'].count()), {"AwayTeam": "Team", "FTR": "Losses away"})

    merged = groups[0].merge(groups[1], on="Team").merge(groups[2], on="Team").merge(groups[3], on="Team")

    print("Data on wins/losses")
    print(merged)

    print("Averages of the wins/losses")
    print(merged.describe())

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

"""
Comparing the defensive and offensive stats between the two teams.
Using matplotlib to visualise the data.
Also using the poisson distribution model.
Relevant to question 1.3
"""
def comparestats():
    c_goals = mergeteamdata()
    c_goals = c_goals[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
    c_goals = c_goals.rename(columns={"FTHG": "HomeGoals", "FTAG": "AwayGoals"})
    c_goals.head()

    poisson_p = np.column_stack([[poisson.pmf(i, c_goals.mean()[j]) for i in range(8)] for j in range(2)])
    plt.hist(c_goals[["HomeGoals", "AwayGoals"]].values, range(9), alpha=0.7, label=["Home", "Away"], normed=True, color=["#FFA07A", "#20B2AA"])

    pois1, = plt.plot([i-0.5 for i in range(1,9)], poisson_p[:,0], linestyle="-", marker="o", label="Home", color="#CD5C5C")
    pois2, = plt.plot([i-0.5 for i in range(1,9)], poisson_p[:,1], linestyle="-", marker="o", label="Away", color="#006400")
    leg=plt.legend(loc="upper right", fontsize=13, ncol=2)

    leg.set_title("Poisson               Actual            ", prop = {"size": "14", "weight":"bold"})
    plt.xticks([i - 0.5 for i in range(1, 9)], [i for i in range(9)])
    plt.xlabel("Goals per Match", size=13)
    plt.ylabel("Proportion of Matches", size=13)
    plt.title("Number of Goals per Match", size=14, fontweight='bold')
    plt.ylim([-0.004, 0.4])
    plt.tight_layout()
    plt.show()

summarisedata()
#comparestats()
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

