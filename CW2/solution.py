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

data = pd.read_csv("diabetic_data.csv")


print(data)
def cleanData(df):
	df.replace('?', np.nan, inplace = True) # Convert ? to NaN 
	pd.isnull(df).values.sum() # Check for missing values

	# Create a mapping to replace IDs in admission type, as this col might be useful later on
	adm_mapping = {1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn", 5: "Not Available", 6: np.NaN, 7: "Trauma Center", 8: np.NaN}
	df = df.replace({"admission_type_id": adm_mapping}) # Apply this mapping to the column

	df.loc[df.readmitted != "NO", "readmitted"] = "YES"	# Convert all non-"NO" values to "YES" for readmitted

	df.dropna(axis = 0, how = 'all', inplace = True)	# Drop rows that have NaN or no values

	return df


def filterCols(dataframe, cols):
	return dataframe[cols]

data = cleanData(data)

print(data)

filtered = filterCols(data, ["race", "gender", "age", "time_in_hospital", "num_procedures", "number_diagnoses", "readmitted", "admission_type_id"])

print(filtered.describe())
print("Skewness of num_procedures: ", stats.normaltest(filtered["num_procedures"]))

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
print(gender)
del gender["Unknown/Invalid"]
#createPieChart(gender, "Diabetic pateints by gender")

age = filtered.age.value_counts()
#createPieChart(age, "Diabetic patients by age")


readmitted = filtered.readmitted.value_counts()
#createPieChart(readmitted, "Readmitted")

admission_type = filtered.admission_type_id.value_counts()
#createPieChart(admission_type, "Admission Type")


grp1 = filtered[["readmitted", "admission_type_id"]].groupby(["admission_type_id", "readmitted"], as_index=False).size()
print(grp1)


grp2 = filtered[["readmitted", "age"]].groupby(["age", "readmitted"], as_index=False).size()
print(grp2)

#createBar(filtered.time_in_hospital, ["days in hospital", "number of patients"], 0, 1, "Days spent in hospital", "Number of patients")

#createBar(filtered.number_diagnoses, ["number diagnoses", "number of patients"], 0, 1, "Number of Diagnoses", "Number of patients")

def pairs(plot_cols, df):
	from pandas.tools.plotting import scatter_matrix
	fig = plt.figure(1, figsize=(12, 12))
	fig.clf()
	ax = fig.gca()
	scatter_matrix(df[plot_cols], alpha=0.3, diagonal='kde', ax = ax)
	plt.show()
	return('Done')



#plot_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_diagnoses"]
#pairs(plot_cols, data)

_df = data.copy()
_df["readmitted_num"] = _df["readmitted"].map({"NO": 0, "YES": 1})

del _df["medical_specialty"]

# age_dict = {"[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95}
# _df["age_num"] = _df["age"].map(age_dict)


clf = linear_model.LogisticRegression()
f_cols = ['number_emergency', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_diagnoses', ]
X = _df[f_cols]
Y = _df['readmitted_num']
clf.fit(X, Y)
print("Model score:\n {}".format(clf.score(X,Y)))
print("Intercept:\n {}".format(clf.intercept_))
print("Coefficients:\n")
for feat, coef in zip(f_cols, clf.coef_[0]):
	print(" {:>20}: {}".format(feat, coef))	

# cat_cols = ["age", "admission_type_id", "diabetesMed", "change", "insulin"]
# t_df = _df[f_cols+cat_cols]
# df_cat = pd.get_dummies(t_df, columns=cat_cols)

to_remove = ["encounter_id", "patient_nbr", "race", "gender", "admission_source_id", "discharge_disposition_id", "payer_code", "diag_1", "diag_2", "diag_3", ]

cdf = _df[_df.columns.difference(_df.columns[_df.isna().any()].tolist()+to_remove)]

df_cat = pd.get_dummies(cdf, columns=cdf.ix[:, cdf.columns != 'readmitted_num'].select_dtypes(include=[np.object]))

model = linear_model.LogisticRegression()
X0 = df_cat.ix[:, df_cat.columns != 'readmitted_num']
Y0 = cdf['readmitted_num']
selector = feature_selection.RFE(model, n_features_to_select=12, step=1)
selector = selector.fit(X0, Y0)
selected_features = df_cat.ix[:, selector.support_]
print("Selected features:\n{}".format(',\n'.join(list(selected_features))))

try:
    del selected_features["readmitted_NO"]
    del selected_features["readmitted_YES"]
except KeyError:
    print("Did not remove a key")

X = selected_features
Y = cdf['readmitted_num']
trainX, testX, trainY, testY = cross_validation.train_test_split(
X, Y, test_size=0.3, random_state=0)
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