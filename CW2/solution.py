import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import numpy as np
import matplotlib.mlab as mlab
from sklearn import linear_model, preprocessing, cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn import feature_selection, metrics
from sklearn.cluster import KMeans
from sklearn import datasets 
from sklearn.decomposition import PCA
import seaborn

data = pd.read_csv("diabetic_data.csv") # Firstly read the data-set csv

'''
Cleaning the data:
- Converting ? or spsace ' ' values to NaN
- Checking for missing values in the dataframe
- Mapping the IDs given in the ID csv to the IDs we are interested in
- Converting the predicting column to YES NO
- Dropping rows if more than 75% of values are missing from a column to eliminate columns that may not help us in the future
- Dropping rows with more than 40 missing columns, total being 50
'''
def cleanData(df):
	df.replace(regex=r'\?', value=np.nan, inplace=True) # Convert ? to NaN 
	pd.isnull(df).values.sum() # Check for missing values

	# Create a mapping to replace IDs in admission type, as this col might be useful later on
	adm_mapping = {1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn", 5: "Not Available", 6: np.NaN, 7: "Trauma Center", 8: np.NaN}

	df = df.replace({"admission_type_id": adm_mapping}) # Apply this mapping to the column

	df.loc[df.readmitted != "NO", "readmitted"] = "YES"	# Convert all non-"NO" values to "YES" for readmitted

	df = df.dropna(axis=1, thresh=(len(df)/4)*3)
	df = df.dropna(axis=0, thresh=40)	
	return df


# Normalise function to normalise the data
def normalize(df):
    assert isinstance(df, pd.DataFrame), 'Argument of wrong type!'
    num_cols = df.select_dtypes(include=[np.number]).copy()
    df_norm = ((num_cols-num_cols.min())/(num_cols.max()-num_cols.min()))
    df_norm = df_norm.dropna()
    return df_norm


'''
Function to drop list of columns from a given dataframe df
'''
def dropAll(df, todrop):
	for i in todrop:
		try:
			del df[i]
		except:
			continue
	return df

def filterCols(dataframe, cols):
	return dataframe[cols]

data = cleanData(data)
print(data)

'''
Function for plotting bloxplots
df - dataframe to plot
plot_cols - columns to plot 
by - Column(Y axis) to plot against
'''
def boxplot(df, plot_cols, by):
	for col in plot_cols:
		fig = plt.figure(figsize=(6, 5))
		ax = fig.gca()
		df.boxplot(column = col, by = by, ax = ax)
		ax.set_title("Box plots of {} by {}".format(col, by))
		plt.suptitle("")
		ax.set_ylabel(col)
		plt.show()

# Plotting the numerical columns we are interested in
boxplot(data, ["number_emergency", "num_medications", "num_procedures", "number_diagnoses"], "readmitted")

# Making a copy of the dataframe to remove outliers and plot again to see if it helps.
xdata = data.copy()
xdata["readmitted"] = xdata["readmitted"].map({"NO": 0, "YES": 1})
xdata = xdata.select_dtypes(include=[np.number])

xdata = xdata[(np.abs(stats.zscore(xdata)) < 4).all(axis=1)]

print("outlier removed count: ", len(xdata))

# Replot the data after normalising it
xdata = normalize(xdata)
boxplot(xdata, ["number_emergency", "num_medications", "num_procedures", "number_diagnoses"], "readmitted")

# Create a new column for normalised data, dropping the ids and other cols we are not interested in
df_norm = normalize(data)
df_norm = dropAll(df_norm, ["encounter_id", "patient_nbr", "discharge_disposition_id", "admission_source_id"])
print(df_norm.columns)

# Filter the columns that are relavent and we want to analyse further
filtered = filterCols(data, ["race", "gender", "age", "time_in_hospital", "num_procedures", "number_diagnoses", "readmitted", "admission_type_id"])

print(filtered.describe())
# print("Skewness of num_procedures: ", stats.normaltest(filtered["num_procedures"]))

'''
Function to plot pie charts
dct - data groups to plot 
xlabel - label of pie chart
'''
def createPieChart(dct, xlabel):
	labels = list(dct.keys())
	sizes  = [dct[x] for x in labels]

	plt.pie(sizes, labels=['' for x in labels], autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.1 , startangle=140)
	plt.legend( loc = 'right', labels=['%s, %d' % (l, s) for l, s in zip(labels, sizes)])
	plt.axis('equal')
	plt.xlabel(xlabel)
	plt.show()


'''
Function to plot a bar chart
df - dataframe to plot
cols - columns from the dataframe to plot
xcol - x axis column
ycol - y axis column
xlabel - x axis label
ylabel - y axis label
'''
def createBar(df, cols, xcol, ycol, xlabel, ylabel):
	ser = df.value_counts().to_frame().reset_index().sort_values(by="index")
	ser.columns = cols
	ser.plot(x=cols[xcol], y=cols[ycol], kind='bar', legend=None, color=["C0"])
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()

# Get the groups from the dataframe to create pie charts for checking proportionality
race = filtered.race.value_counts()
createPieChart(race, "Diabetic patients by race")

gender = filtered.gender.value_counts()
del gender["Unknown/Invalid"]
createPieChart(gender, "Diabetic pateints by gender")

age = filtered.age.value_counts()
createPieChart(age, "Diabetic patients by age")


readmitted = filtered.readmitted.value_counts()
createPieChart(readmitted, "Readmitted")

admission_type = filtered.admission_type_id.value_counts()
createPieChart(admission_type, "Admission Type")


# See the correlation of admission and readmission
grp1 = filtered[["readmitted", "admission_type_id"]].groupby(["admission_type_id", "readmitted"], as_index=False).size()
print(grp1)

# See the correlation of age and readmission
grp2 = filtered[["readmitted", "age"]].groupby(["age", "readmitted"], as_index=False).size()
print(grp2)

createBar(filtered.time_in_hospital, ["days in hospital", "number of patients"], 0, 1, "Days spent in hospital", "Number of patients")

createBar(filtered.number_diagnoses, ["number diagnoses", "number of patients"], 0, 1, "Number of Diagnoses", "Number of patients")

# Pairing the data to create a scatter matrix to see correlations between different numeric columns
def pairs(plot_cols, df):
	from pandas.tools.plotting import scatter_matrix
	fig = plt.figure(1, figsize=(12, 12))
	fig.clf()
	ax = fig.gca()
	scatter_matrix(df[plot_cols], alpha=0.3, diagonal='kde', ax = ax)
	plt.show()
	return('Done')



plot_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_diagnoses"]
pairs(plot_cols, data)

'''
Function to plot clusters
df1 - x axis dataframe to plot
df2 - y axis dataframe to plot 
xlabel - x axis label
ylabel - y axis label
'''
def plotclusters(df1, df2, xlabel, ylabel):
	plt.plot(df1, df2,'ro')
	plt.title(xlabel+" vs "+ylabel)
	plt.xlabel(xlabel) 
	plt.ylabel(ylabel)
	plt.show()


plotclusters(data["num_medications"], data["readmitted"], "# of Medications", "Readmitted")
plotclusters(data["num_lab_procedures"], data["readmitted"], "# of Lab Procedures", "Readmitted")
plotclusters(data["number_outpatient"], data["readmitted"], "# Outpatient", "Readmitted")
plotclusters(data["number_emergency"], data["readmitted"], "# Emergency", "Readmitted")
plotclusters(data["number_inpatient"], data["readmitted"], "# Inpatient", "Readmitted")


# Create a new dataframe for building a predictive model
_df = data.copy()

# Convert the readmitted column to 0,1 NO,YES respectively
_df["readmitted"] = _df["readmitted"].map({"NO": 0, "YES": 1})

# Remove all the outliers, using the same technique as before
_df = _df[(np.abs(stats.zscore(_df.select_dtypes(exclude='object'))) < 3).all(axis=1)]

# Using a logistic regression model to create a model based on the features we are interested in
clf = linear_model.LogisticRegression()

# List of columns to consider - numerical
f_cols = ['number_emergency', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_diagnoses', 'number_inpatient', 'number_outpatient']
X = _df[f_cols]
Y = _df['readmitted']
clf.fit(X, Y)

# Obtaining the score for the model, and coefficients
print("Model score:\n {}".format(clf.score(X,Y)))
print("Intercept:\n {}".format(clf.intercept_))
print("Coefficients:\n")
for feat, coef in zip(f_cols, clf.coef_[0]):
	print(" {:>20}: {}".format(feat, coef))	

# Validate the model with test and training set to see how it performs
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)
clf = linear_model.LogisticRegression()
clf.fit(X_train, Y_train)
print("Score against training data: {}".format(clf.score(X_train, Y_train)))
print("Score against test data: {}".format(clf.score(X_test, Y_test)))

# List of categorical columns we are interested in based on our analysis
cat_cols = ["age", "admission_type_id", "diabetesMed", "change", "insulin"]
t_df = _df[f_cols+cat_cols]
df_cat = pd.get_dummies(t_df, columns=cat_cols)

# Running the logistic regression again to pick features that we can use to build up our model further
model = linear_model.LogisticRegression()
X0 = df_cat.ix[:, df_cat.columns != 'readmitted']
Y0 = _df['readmitted']	
selector = feature_selection.RFE(model, n_features_to_select = 20, step=1)
selector = selector.fit(X0, Y0)
selected_features = df_cat.ix[:, selector.support_]
print("Selected features:\n{}".format(',\n'.join(list(selected_features))))

# Use the selected freatures to rerun the model and see if that wil help for improvement and revalidate this against the test set
X = selected_features
Y = _df['readmitted']
trainX, testX, trainY, testY = cross_validation.train_test_split(X, Y, test_size=0.3, random_state=0)
clf = linear_model.LogisticRegression()
clf.fit(trainX, trainY)
predicted = clf.predict(testX)
print("Mean hits: {}".format(np.mean(predicted==testY)))
print("Accuracy score: {}".format(metrics.accuracy_score(testY, predicted)))
scores = cross_validation.cross_val_score(
linear_model.LogisticRegression(), X, Y, scoring='accuracy', cv=8)
print("Cross validation mean scores: {}".format(scores.mean()))

# Creating an ROC curve to get a better sense of the linear model
prob = np.array(clf.predict_proba(testX)[:, 1])
testY += 1
fpr, sensitivity, _ = metrics.roc_curve(testY, prob, pos_label=2)
print("AUC = {}".format(metrics.auc(fpr, sensitivity)))
plt.scatter(fpr, fpr, c='b', marker='s')
plt.scatter(fpr, sensitivity, c='r', marker='o')
plt.show()


#########
## call KMeans algo with 6 clusters
model = KMeans(n_clusters=6)
model.fit(df_norm)
    ## J score
print('J-score = ', model.inertia_)
print(' score = ', model.score(df_norm))
    ## include the labels into the data
print(model.labels_)
labels = model.labels_
md = pd.Series(labels)
df_norm['clust'] = md
df_norm.head(5)
    ## cluster centers 
centroids = model.cluster_centers_
print ('centroids', centroids)
    ## histogram of the clusters
plt.hist(df_norm['clust'])
plt.title('Histogram of Clusters')
plt.xlabel('Cluster')
plt.ylabel('Frequency')
plt.show()
## means of the clusters
print ('clustered data', df_norm)
print (df_norm.groupby('clust').mean())

######## 2D plot of the clusters
pca_data = PCA(n_components=6).fit(df_norm)
pca_2d = pca_data.transform(df_norm)
plt.scatter(pca_2d[:,0], pca_2d[:,1], c=labels)
plt.title('Readamission clusters')
plt.show()