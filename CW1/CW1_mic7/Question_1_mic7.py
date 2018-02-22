# Question 1 - Predicting the winner

# Firstly we need to import some modules to help us along the way.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import poisson, skellam
import matplotlib.mlab as mlab
import math
from collections import Counter


# Then we want to read the data file into our program.

data = pd.read_csv("resources/PremierLeague1718.csv") # Read the data set file


# The data file consists of empty columns and rows which we want to clean up so that we are working with _clean_ data. To do this we can use the below function.

def clean(dataframe):
    assert isinstance(dataframe, pd.DataFrame), 'Argument of wrong type!'
    return dataframe.dropna(axis=1, how='all')


# And then we can apply this to our data.

data = clean(data) # Clean the data up to remove empty and NaN cols


# Because the data file consists of a lot of columns, some of them are irrelevant to us, the below function can help eliminate these columns, to do this we can specify only the columns we want to keep as a list of column names.

def filterCols(dataframe, cols):
    assert isinstance(dataframe, pd.DataFrame), 'Argument of wrong type!'
    assert isinstance(cols, list), 'Argument of wrong type!'
    return dataframe[cols]


# We can then use this function to filter our data.

colsToKeep = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "FTR"] # List of cols we want to keep
data = filterCols(data, colsToKeep) # Filter relevant cols


# To get an idea of how the data looks like, we can use the `describe` function to give us various information. (This is helpful to answer Question 1, part 1.)

print("Description:")
print(data.describe())


# Furthermore, we can also group the data in a way so that we can see an outlook of how many goals were scored and conceded by each team over the whole league.


def renameCols(df, colsdict):
    return df.reset_index().rename(columns=colsdict)

def creategroups():
    groups = [0,0,0,0]
    groups[0] = renameCols(data.groupby("HomeTeam")["FTR"].apply(lambda x: x[x == 'H'].count()), {"HomeTeam": "Team", "FTR": "Wins at home"})
    groups[1] = renameCols(data.groupby("AwayTeam")["FTR"].apply(lambda x: x[x == 'A'].count()), {"AwayTeam": "Team", "FTR": "Wins away"})
    groups[2] = renameCols(data.groupby("HomeTeam")["FTR"].apply(lambda x: x[x == 'A'].count()), {"HomeTeam": "Team", "FTR": "Losses at home"})
    groups[3] = renameCols(data.groupby("AwayTeam")["FTR"].apply(lambda x: x[x == 'H'].count()), {"AwayTeam": "Team", "FTR": "Losses away"})

    return groups

groups = creategroups()
merged = groups[0].merge(groups[1], on="Team").merge(groups[2], on="Team").merge(groups[3], on="Team")

print("Data on wins/losses")
print(merged)

print("Averages of the wins/losses")
print(merged.describe())


# We can create a data structure containing information about the teams we care about. In this instance Manchester United and Manchester City. This can help us answer Question 1 part 2.

rteams = ["Man United", "Man City"] # Teams we care about

teams = {"ManUnitedHome": "", "ManCityHome": "", "ManUnitedAway": "", "ManCityAway": ""} # A dict to contain relevant team dataframes

teams["ManUnitedHome"] = data.loc[data["HomeTeam"] == "Man United"]
teams["ManUnitedAway"] = data.loc[data["AwayTeam"] == "Man United"]
teams["ManCityHome"] = data.loc[data["HomeTeam"] == "Man City"]
teams["ManCityAway"] = data.loc[data["AwayTeam"] == "Man City"]

def mergeteamdata():
    df = teams["ManUnitedHome"].append(teams["ManUnitedAway"])
    df = df.append((teams["ManCityHome"].append(teams["ManCityAway"])))
    return df

print("Man United Home:")
print(teams["ManUnitedHome"].describe())
print("\nMan City Home")
print(teams["ManCityHome"].describe())
print("-------------")
print("Man United Away:")
print(teams["ManUnitedAway"].describe())
print("\nMan City Away")
print(teams["ManCityAway"].describe())


# To get a better visualisation of how these teams stack up against eachother we can plot these values onto a graph t compare Manchester United's home defence against Manchester City's away offence and Manchester City's home defence against Manchester United's away offence.

def plotstats(stat):
    fig = plt.figure(1)
    fig.set_size_inches(12, 8)
    for i in stat:
        plt.subplot(i.plots[0])
        i.df.boxplot(i.col, vert=False)
        plt.subplot(i.plots[1])
        temp = i.df[i.col].as_matrix()
        plt.hist(temp, bins=20, alpha=1, label=i.glabel)
        plt.xlabel(i.xlabel)
        plt.ylabel(i.ylabel)
        plt.legend()
        plt.xticks(np.arange(min(temp), max(temp)+1, 1.0))
        plt.yticks(np.arange(0, len(temp)+1, 1.0))
    plt.show()


# We can create a simple object to use with the above function because it gives us a dynamic way of displaying information for different teams. 

class PlotObject:
    def __init__(self, df, col, glabel, xlabel, ylabel, plots):
        self.df     = df
        self.col    = col
        self.glabel = glabel
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plots  = plots


# We can then plot these teams and see how the data looks like. This will help answer Question 1 part 3. 

# ### Manchester United's home defence vs Manchester City's away offence


obj = []
obj.append(PlotObject(teams["ManUnitedHome"], "FTAG", "Man Utd Home", "Number of goals conceded", "Number of Matches", [221,223]))
obj.append(PlotObject(teams["ManCityAway"], "FTAG", "Man City Away", "Number of goals scored", "Number of Matches", [222,224]))
plotstats(obj)


# ### Manchester City's home defence vs Manchester United's away offence


obj = []
obj.append(PlotObject(teams["ManCityHome"], "FTAG", "Man City Home", "Number of goals conceded", "Number of Matches", [221, 223]))
obj.append(PlotObject(teams["ManUnitedAway"], "FTAG", "Man Utd Away", "Number of goals scored", "Number of Matches", [222, 224]))
plotstats(obj)


# ## Simulation

# To simulate the future matches, we can use the poisson distribution to randomly generate future scores and see how the teams stack up against eachother.

def sim_poisson(nums, mean):
    gen = np.random.poisson(lam = mean, size = nums)
    return gen


# And then we can use this function to generate some scores for the following cases:
# Man Utd Home vs Man City Away
# Man City Home vs Man Utd Away

def generate_scores():
    gen_scores = []
    gen_scores.append(sim_poisson(1000, teams["ManUnitedHome"]["FTHG"].mean()))
    gen_scores.append(sim_poisson(1000, teams["ManUnitedAway"]["FTAG"].mean()))
    gen_scores.append(sim_poisson(1000, teams["ManCityHome"]["FTHG"].mean()))
    gen_scores.append(sim_poisson(1000, teams["ManCityAway"]["FTAG"].mean()))

    MUHVSMCA = list(map(list,zip(gen_scores[0],gen_scores[3])))
    MCHVSMUA = list(map(list,zip(gen_scores[2],gen_scores[1])))
    
    return {"ManUtdHvsMC": MUHVSMCA, "ManCityHvsMU": MCHVSMUA}

genscores = generate_scores()


# We can summarise this data by looking at who won each game.


def getresult(arr):
    if arr[0] > arr[1]:
        return "H"
    elif arr[0] < arr[1]:
        return "A"
    else:
        return "D"


ManUtdHvsMC = [getresult(i) for i in (genscores["ManUtdHvsMC"])]

ManCityHvsMU = [getresult(i) for i in (genscores["ManCityHvsMU"])]

home_stats = {
    "MU Win": Counter(ManUtdHvsMC)["H"],
    "MC Win":  Counter(ManCityHvsMU)["H"],
}

away_stats = {
    "MU Win": Counter(ManCityHvsMU)["A"],
    "MC Win": Counter(ManUtdHvsMC)["A"],
}


draws = Counter(ManCityHvsMU)["D"] + Counter(ManUtdHvsMC)["D"]

df = pd.DataFrame(home_stats, index=["Home"])
df = df.append(pd.DataFrame(away_stats, index=["Away"]))
print("Total number of simulations: ",len(ManUtdHvsMC)+len(ManCityHvsMU))
print("\nWins/Losses simulated: \n",df)
print("\nTotal draws in simulation: ",Counter(ManUtdHvsMC)["D"] + Counter(ManCityHvsMU)["D"])


def findoverallprob(team):
    total_matches = len(ManUtdHvsMC) + len(ManCityHvsMU)
    if team == "MU":
        team_win = home_stats["MU Win"] + away_stats["MU Win"]
    elif team == "MC":
        team_win = home_stats["MC Win"] + away_stats["MC Win"]
    elif team == "D":
        team_win = draws
    return team_win/float(total_matches)


# From this we can observe the following results:


print("Probablity of Manchester United winning: ", findoverallprob("MU"))
print("Probablity of Manchester City winning: ", findoverallprob("MC"))
print("Probability of drawing: ", findoverallprob("D"))




def findprob(team, side):
    if side == "home":
        if team == "MU":
            total_matches = home_stats["MU Win"] + away_stats["MC Win"]
            return home_stats["MU Win"]/float(total_matches)
        elif team == "MC":
            total_matches = home_stats["MC Win"] + away_stats["MU Win"]
            return home_stats["MC Win"]/float(total_matches)
    elif side == "away":
        if team == "MU":
            total_matches = away_stats["MU Win"] + home_stats["MC Win"]
            return away_stats["MU Win"]/float(total_matches)
        elif team == "MC":
            total_matches = away_stats["MC Win"] + home_stats["MU Win"]
            return away_stats["MC Win"]/float(total_matches)


# And we can see the following probabilities from this:

print("Probability of MU win given they play home: ", findprob("MU", "home"))
print("Probability of MC win given they play home: ", findprob("MC", "home"))
print("Probability of MU win given they play away: ", findprob("MU", "away"))
print("Probability of MC win given they play away: ", findprob("MC", "away"))


# From this we can observe that Manchester City has a higher chance of winning both playing home or away compared to Manchester United. From this we can confidently predict that in a match up of Manchester United vs Manchester City in the Premier League, based on the historical data; Manchester City will win the game. 


# Generate 10 games between Chelsea and Arsenal

chelsea = [data.query("HomeTeam == 'Chelsea'"), data.query("AwayTeam == 'Chelsea'")]
arsenal = [data.query("HomeTeam == 'Arsenal'"), data.query("AwayTeam == 'Arsenal'")]

print(chelsea[0].describe())
print(arsenal[0].describe())

print(chelsea[1].describe())
print(arsenal[1].describe())

CHAW = data.query("AwayTeam == 'Arsenal' & HomeTeam == 'Chelsea'")
AHCW = data.query("AwayTeam == 'Chelsea' & HomeTeam == 'Arsenal'")
# print("Chelsea home vs Arsenal Away\n", CHAW)
# print("Arsenal Home vs Chelsea Away\n", AHCA)

g_CH = sim_poisson(10, chelsea[0]["FTHG"].mean())
g_CA = sim_poisson(10, chelsea[1]["FTAG"].mean())

g_AH = sim_poisson(10, arsenal[0]["FTHG"].mean())
g_AA = sim_poisson(10, arsenal[1]["FTAG"].mean())


#g_AHCW = sim_poisson(10, arsenal["FTHG"].mean())

CHAA = zip(g_CH, g_AA)
AHCA = zip(g_AH, g_CA)

print("Chelsea Home vs Arsenal Away: \n")
print("Actual score: ", CHAW)
print("Simulated match results (C = chelsea win, A = Arsenal win, D = Draw): ", ["C" if getresult(x) == "H" else "A" if getresult(x) == "A" else "D" for x in CHAA])

print("\n-------------------------\n")

print("Arsenal Home vs Chelsea Away: \n")
print("Actual score: ", AHCW)
print("Simulated match results (C = chelsea win, A = Arsenal win, D = Draw): ", ["C" if getresult(x) == "H" else "A" if getresult(x) == "A" else "D" for x in AHCA])

