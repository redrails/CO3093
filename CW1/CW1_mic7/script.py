import csv
from collections import OrderedDict
import json
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

data = []

# Read the csv file.
with open("resources/PremierLeague1718.csv") as f:
	reader = csv.reader(f)
	d_headers = next(reader)

	for row in reader:
		data.append(OrderedDict(zip(d_headers, row)))

for i in data:
	i.pop('Div');
	i.pop('B365H')
	i.pop('B365D')
	i.pop('B365A')
	i.pop('BWH')
	i.pop('BWD')
	i.pop('BWA')
	i.pop('IWH')
	i.pop('LBH')
	i.pop('LBD')
	i.pop('LBA')
	i.pop('PSH')
	i.pop('PSD')
	i.pop('PSA')
	i.pop('WHH')
	i.pop('WHD')
	i.pop('WHA')
	i.pop('VCH')
	i.pop('VCD')
	i.pop('VCA')
	i.pop('Bb1X2')
	i.pop('BbMxH')
	i.pop('BbAvH')
	i.pop('BbMxD')
	i.pop('BbAvD')
	i.pop('BbMxA')
	i.pop('BbAvA')
	i.pop('BbOU')
	i.pop('BbMx>2.5')
	i.pop('BbAv>2.5')
	i.pop('BbMx<2.5')
	i.pop('BbAv<2.5')
	i.pop('BbAH')
	i.pop('BbAHh')
	i.pop('BbMxAHH')
	i.pop('BbAvAHH')
	i.pop('BbMxAHA')
	i.pop('BbAvAHA')
	i.pop('PSCH')
	i.pop('PSCD')
	i.pop('PSCA')
	
	for k in i.keys():
		try:
			if len(i[k]) < 1:
				del i[k]
		except: pass

my_teams = ["Man United", "Man City"]
data_mod = [i for i in data if i["HomeTeam"] in my_teams or i["AwayTeam"] in my_teams]


m_c_h_stats = []
m_c_a_stats = []
m_u_h_stats = []
m_u_a_stats = []

my_dates = []

for i in data_mod:

	_date = dt.datetime.strptime(i["Date"], '%d/%m/%y').date()

	if i["HomeTeam"] == "Man City":
		#m_c_h_stats.append(i["FTHG"])
		m_c_h_stats.append([_date, int(i["FTHG"])])
	elif i["AwayTeam"] == "Man City":
		#m_c_a_stats.append(i["FTAG"])
		m_c_a_stats.append([_date, int(i["FTAG"])])
	elif i["HomeTeam"] == "Man United":
		#m_u_h_stats.append(i["FTHG"])
		m_u_h_stats.append([_date, int(i["FTHG"])])
	elif i["AwayTeam"] == "Man United":
		#m_u_a_stats.append(i["FTAG"])
		m_u_a_stats.append([_date, int(i["FTAG"])])
	else:
		continue

	my_dates.append(_date)


set(my_dates)
#print(np.array(m_c_h_stats))
#print(my_dates)

# plt.plot(m_c_h_stats, [i[1] for i in m_c_h_stats])
# plt.plot(m_c_a_stats, [i[1] for i in m_c_a_stats])
# plt.plot(m_u_h_stats, [i[1] for i in m_u_h_stats])
# plt.plot(m_u_a_stats, [i[1] for i in m_u_a_stats])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([x[0] for x in m_c_h_stats], [x[1] for x in m_c_h_stats], linestyle="-", label='MC Home')
ax.plot([x[0] for x in m_c_a_stats], [x[1] for x in m_c_a_stats], linestyle="-", label='MC Away')
ax.plot([x[0] for x in m_u_h_stats], [x[1] for x in m_u_h_stats], linestyle="-", label='MU Home')
ax.plot([x[0] for x in m_u_a_stats], [x[1] for x in m_u_a_stats], linestyle="-", label='MU Away')

print(m_c_h_stats)
print(m_c_a_stats)
print(m_u_h_stats)
print(m_u_a_stats)
# plt.plot(m_c_a_stats, linestyle="-", label='MC Away')
# plt.plot(m_u_h_stats, linestyle="-", label='MU Home')
# plt.plot(m_u_a_stats, linestyle="-", label='MU Away')
plt.legend()
plt.show()
#print(json.dumps(data_mod, indent=4))