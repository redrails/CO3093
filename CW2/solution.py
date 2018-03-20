import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import numpy as np
import matplotlib.mlab as mlab
import ast
from sklearn import linear_model, preprocessing, cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn import feature_selection, metrics
from sklearn.cluster import KMeans
from sklearn import datasets 
from sklearn.decomposition import PCA

data = pd.read_csv("diabetic_data.csv")

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

def normalize(df):
    assert isinstance(df, pd.DataFrame), 'Argument of wrong type!'
    num_cols = df.select_dtypes(include=[np.number]).copy()
    df_norm = ((num_cols-num_cols.min())/(num_cols.max()-num_cols.min()))
    df_norm = df_norm.dropna()
    return df_norm

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
print(data.columns)


def boxplot(df, plot_cols, by):
	for col in plot_cols:
		fig = plt.figure(figsize=(9, 6))
		ax = fig.gca()
		df.boxplot(column = col, by = by, ax = ax)
		ax.set_title("Box plots of {} bg {}".format(col, by))
		ax.set_ylabel(col)
		plt.show()

boxplot(data, ["number_emergency", "num_medications", "num_procedures", "number_diagnoses"], "readmitted")


df_norm = normalize(data)

df_norm = dropAll(df_norm, ["encounter_id", "patient_nbr", "discharge_disposition_id", "admission_source_id"])

print(df_norm.columns)

filtered = filterCols(data, ["race", "gender", "age", "time_in_hospital", "num_procedures", "number_diagnoses", "readmitted", "admission_type_id"])

print(filtered.describe())
# print("Skewness of num_procedures: ", stats.normaltest(filtered["num_procedures"]))

def createPieChart(dct, xlabel):
	labels = list(dct.keys())
	sizes  = [dct[x] for x in labels]

	plt.pie(sizes, labels=['' for x in labels], autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.1 , startangle=140)
	plt.legend( loc = 'right', labels=['%s, %d' % (l, s) for l, s in zip(labels, sizes)])
	plt.axis('equal')
	plt.xlabel(xlabel)
	plt.show()

def createBar(df, cols, xcol, ycol, xlabel, ylabel):
	ser = df.value_counts().to_frame().reset_index().sort_values(by="index")
	ser.columns = cols
	ser.plot(x=cols[xcol], y=cols[ycol], kind='bar', legend=None, color=["C0"])
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()


race = filtered.race.value_counts()
#createPieChart(race, "Diabetic patients by race")

gender = filtered.gender.value_counts()
del gender["Unknown/Invalid"]
#createPieChart(gender, "Diabetic pateints by gender")

age = filtered.age.value_counts()
#createPieChart(age, "Diabetic patients by age")


readmitted = filtered.readmitted.value_counts()
#createPieChart(readmitted, "Readmitted")

admission_type = filtered.admission_type_id.value_counts()
#createPieChart(admission_type, "Admission Type")


# See the correlation of admission and readmission
grp1 = filtered[["readmitted", "admission_type_id"]].groupby(["admission_type_id", "readmitted"], as_index=False).size()
print(grp1)

# See the correlation of age and readmission
grp2 = filtered[["readmitted", "age"]].groupby(["age", "readmitted"], as_index=False).size()
print(grp2)

#createBar(filtered.time_in_hospital, ["days in hospital", "number of patients"], 0, 1, "Days spent in hospital", "Number of patients")

#createBar(filtered.number_diagnoses, ["number diagnoses", "number of patients"], 0, 1, "Number of Diagnoses", "Number of patients")

# def pairs(plot_cols, df):
# 	from pandas.tools.plotting import scatter_matrix
# 	fig = plt.figure(1, figsize=(12, 12))
# 	fig.clf()
# 	ax = fig.gca()
# 	scatter_matrix(df[plot_cols], alpha=0.3, diagonal='kde', ax = ax)
# 	plt.show()
# 	return('Done')



# plot_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_diagnoses"]
# pairs(plot_cols, data)

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

_df = data.copy()

_df["readmitted"] = _df["readmitted"].map({"NO": 0, "YES": 1})

_df = normalize(_df)

clf = linear_model.LogisticRegression()
f_cols = ['number_emergency', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_diagnoses', 'number_inpatient', 'number_outpatient']
X = _df[f_cols]
Y = _df['readmitted']
clf.fit(X, Y)
print("Model score:\n {}".format(clf.score(X,Y)))
print("Intercept:\n {}".format(clf.intercept_))
print("Coefficients:\n")
for feat, coef in zip(f_cols, clf.coef_[0]):
	print(" {:>20}: {}".format(feat, coef))	

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)
clf = linear_model.LogisticRegression()
clf.fit(X_train, Y_train)
print("Score against training data: {}".format(clf.score(X_train, Y_train)))
print("Score against test data: {}".format(clf.score(X_test, Y_test)))

cat_cols = ["age", "admission_type_id", "diabetesMed", "change", "insulin"]
t_df = _df[f_cols+cat_cols]
df_cat = pd.get_dummies(t_df, columns=cat_cols)

#to_remove = ["encounter_id", "patient_nbr", "race", "gender", "admission_source_id", "discharge_disposition_id", "payer_code", "diag_1", "diag_2", "diag_3"]
#to_keep = f_cols+["readmitted"]
#cdf = _df[_df.columns.difference(to_remove)]
#cdf = _df[to_keep]
#df_cat = pd.get_dummies(cdf, columns=cdf.ix[:, cdf.columns != 'readmitted'].select_dtypes(include=[np.object]))

model = linear_model.LogisticRegression()
X0 = df_cat.ix[:, df_cat.columns != 'readmitted']
Y0 = _df['readmitted']
selector = feature_selection.RFE(model, n_features_to_select = 20, step=1)
selector = selector.fit(X0, Y0)
selected_features = df_cat.ix[:, selector.support_]
print("Selected features:\n{}".format(',\n'.join(list(selected_features))))

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
plt.title('Wine clusters')
plt.show()