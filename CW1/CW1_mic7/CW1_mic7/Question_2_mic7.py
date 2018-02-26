import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm, feature_selection, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

data = pd.read_csv("resources/AAPL.csv") # Read the data set file

# The data file consists of empty columns and rows which we want to clean up so that we are working with _clean_ data. To do this we can use the below function.
def clean(dataframe):
    assert isinstance(dataframe, pd.DataFrame), 'Argument of wrong type!'
    return dataframe.dropna() # This will drop any rows that contain NaN values. Important for training.
data = clean(data)


# Creating a timeseries graph, to do this we will make date the key so that the data is structured with date.
data["Date"] = pd.to_datetime(data["Date"]) # Convert the Date column into DateTime object
data.index = data["Date"]                   # Set the index of the dataset to be the Date column
del data["Date"]                            # Remove the Date column


# Plot timeseries.
def timeseries(df):
    assert isinstance(df, pd.DataFrame), 'Argument of wrong type!'
    ts = pd.Series(df["Adj Close"])
    #ts = ts.cumsum()
    fig = plt.figure()
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price")
    fig.set_size_inches(12,8)
    ts.plot()
    plt.show()
timeseries(data)

# Normalise data
def normalize(df):
    assert isinstance(df, pd.DataFrame), 'Argument of wrong type!'
    num_cols = df.select_dtypes(include=[np.number]).copy()
    df_norm = ((num_cols-num_cols.min())/(num_cols.max()-num_cols.min()))
    df_norm = df_norm.dropna()
    return df_norm

# Feature selection
def selectfeatures(dfn, toexclude, topredict):
    assert isinstance(dfn, pd.DataFrame), 'Argument of wrong type!'
    assert isinstance(toexclude, list), 'Argument of wrong type!'
    assert isinstance(topredict, str), 'Argument of wrong type!'

    dfn = dfn.select_dtypes(include=[np.number]).copy()
    feature_cols = [x for x in dfn.columns.values.tolist() if x not in toexclude] # exclude features we are not predicting
    print(feature_cols)
    XO = dfn[feature_cols]
    YO = dfn[topredict]
    estimator = svm.SVR(kernel="linear")
    selector = feature_selection.RFE(estimator, 5, step=1)
    selector = selector.fit(XO, YO)

    # From the ranking you can select your predictors with rank 1
    # Model 1; let us select the folowing features as predictors:
    select_features = np.array(feature_cols)[selector.ranking_ == 1].tolist()
    print("Features: ", select_features)
    return select_features

def linearModel(dfn, features)
    assert isinstance(dfn, pd.DataFrame), 'Argument of wrong type!'
    assert isinstance(features, list), 'Argument of wrong type!'

    X = dfn[features]
    Y = dfn["Adj Close"]

    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.25)
    lm = linear_model.LinearRegression()
    lm.fit(trainX, trainY)

    # Inspect the calculated model equations
    print("Y-axis intercept {}".format(lm.intercept_))
    print("Weight coefficients:")

    for feat, coef in zip(features, lm.coef_):
        print(" {:>20}: {}".format(feat, coef))

    # The value of R^2
    print("R squared for the training data is {}".format(lm.score(trainX, trainY)))
    print("Score against test data: {}".format(lm.score(testX, testY)))
    pred_trainY = lm.predict(trainX)
    pred_testY = lm.predict(testX)
    print("MSE: {}".format(mse(trainY, pred_trainY)))
    return {"trainX": trainX, "trainY": trainY, "testX": testX, "testY": testY, "pred_trainY": pred_trainY, "pred_testY": pred_testY}


def plotResiduals(A, B, title, xlabel, ylabel):
    assert isinstance(A, int), 'Argument of wrong type!'
    assert isinstance(B, int), 'Argument of wrong type!'
    assert isinstance(title, str), 'Argument of wrong type!'
    assert isinstance(xlabel, str), 'Argument of wrong type!'
    assert isinstance(ylabel, str), 'Argument of wrong type!'

    for X,Y in zip(A, B):
        plt.plot(X, Y-X, 'o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def linearplot(obsy, predy):
    assert isinstance(obsy, list), 'Argument of wrong type!'
    assert isinstance(predy, list), 'Argument of wrong type!'

    plt.figure(figsize=(14, 8))
    plt.plot(obsy, predy, 'o')
    plt.xlabel('Actual Prices')
    plt.ylabel("Predicted Prices")
    plt.title("Plot of predicted vs actual prices")
    plt.show()


df = data.copy()    # Create a copy of the data


#Initial model without anything extra
def step1():
    dfn = normalize(df)
    select_features = selectfeatures(dfn, ["Adj Close"], "Adj Close")

    lm = linearModel(dfn, select_features)
    linearplot(lm["trainY"], lm["pred_trainY"])
    plotResiduals(lm["trainY"], lm["pred_trainY"], "Actual prices vs Predicted Prices residuals (Training Data)", "Adjusted Close", "Residuals")

# Adding a 5 day shift column
def step2():
    df["FDS"] = df["Adj Close"].shift(-5) # Shifting by -5 to move data upwards, i.e. to make i+5th date to the ith date
    df.dropna()                           # Drop empty columns at the end of the shift as the last 5 days will lose values

    dfn = normalize(df)
    select_features = selectfeatures(dfn, ["Adj Close"], "Adj Close")

    lm = linearModel(dfn, select_features)
    linearplot(lm["trainY"], lm["pred_trainY"])
    plotResiduals(lm["trainY"], lm["pred_trainY"], "Actual prices vs Predicted Prices residuals (Training Data)", "Adjusted Close", "Residuals")

# Finally adding a Moving Average to the five-day-shift
def step3():
    FDS = df["Adj Close"].shift(-5) # Create a column for shifting by 5 days to help predict the prices over the next 5 days.
    df["FDSMA"] = FDS.rolling(window=5).mean()
    df.dropna() # Because we are shifting upwards, there will be lost rows at the bottom of the dataset.

    dfn = normalize(df)

    select_features = selectfeatures(dfn, ["Adj Close"], "Adj Close")

    lm = linearModel(dfn, select_features)
    linearplot(lm["trainY"], lm["pred_trainY"])

    print(dfn.tail(10))
    print(df.tail(10))


# Please be careful running these functions as they can be slow to run. The residuals can take a while to process
# please give enough time for it to plot the residuals.

#step1()
#step2()
#step3()
