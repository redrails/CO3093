import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


data = pd.read_csv("diabetic_data.csv")

data = data.dropna()


def filterCols(dataframe, cols):
	return dataframe[cols]

filtered = filterCols(data, ["race", "gender", "age", "time_in_hospital", "num_procedures"])

print(filtered.describe())
print("Skewness of time_in_hospital: ", stats.normaltest(filtered["time_in_hospital"]))

def createPieChart(dct, xlabel):
	labels = list(dct.keys())
	sizes  = [dct[x] for x in labels]

	plt.pie(sizes, labels=['' for x in labels], autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.1 , startangle=140)
	plt.legend( loc = 'right', labels=['%s, %d' % (l, s) for l, s in zip(labels, sizes)])
	plt.axis('equal')
	plt.xlabel(xlabel)
	plt.show()


race = filtered.race.value_counts()
#createPieChart(race, "Diabetic patients by race")

gender = filtered.gender.value_counts()
del gender["Unknown/Invalid"]
#createPieChart(gender, "Diabetic pateints by gender")

age = filtered.age.value_counts()
#createPieChart(age, "Diabetic patients by age")

tih = filtered.time_in_hospital.value_counts()
print(tih)
tih.hist(grid=False, bins=20)
plt.show()