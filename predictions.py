from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_children = pd.read_csv("data/Statewise Cases Reported of Crimes Committed Against Children 1994-2016.csv", header=None)
children_states=[i for i in data_children[0][1:].unique()]
children_crimes=[i for i in data_children[1][1:].unique()]
children_years=[2017,2018,2019,2020,2021,2022,2023,2024,2025,2026,2027,2028,2029,2030]
data_women = pd.read_csv("data/statewise_crime_against_women_2001_15.csv")
women_states=[i for i in data_women['STATE/UT'].unique()]
women_crimes=[i for i in data_women][2:]
women_years=[2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026,2027,2028,2029,2030]

def children_prediction(state, year, crime):
    global data_children
    data = data_children
    year = int(year)

    # Extract years from the first row
    X = data.iloc[0, 2:].values  # First row contains years
    X_train = np.array([int(i) for i in X])

    # Filter data by state and crime
    data = data[(data[0] == state) & (data[1] == crime)]
    
    # Check if any data is found
    if data.empty:
        return None, None, f"No data available for state: {state} and crime: {crime}"

    # Extract the crime data for that state and crime
    y = data.iloc[0, 2:].values
    y_train = np.array([int(i) for i in y])

    linear_regression = LinearRegression()
    linear_regression.fit(X_train.reshape(-1, 1), y_train)
    score = linear_regression.score(X_train.reshape(-1, 1), y_train)

    if score < 0.60:
        return y, X_train, "Can't predict further"

    # Generate predictions for the future years
    for j in range(2017, year + 1):
        prediction = linear_regression.predict(np.array([[j]]))
        prediction = max(0, prediction)  # Avoid negative predictions
        y = np.append(y, prediction)

    # Extend years array accordingly
    years = np.array([str(i) for i in range(1994, year + 1)])
    
    return y.tolist(), years.tolist(), ""

# You can now test this with proper inputs and catch errors like data not being found.

def women_prediction(state,year,crime):
	global data_women
	data=data_women
	print(year)
	year=int(year)
	data =data[data['STATE/UT']==state]
	X=[i for i in data['Year']]
	X_train=np.array([int(i) for i in X])
	y=[i for i in data[crime]]
	y_train=np.array([int(i) for i in y])
	linear_regression=LinearRegression()
	linear_regression.fit(X_train.reshape(-1,1),y_train)
	print(len(X_train),len(y_train))
	score=linear_regression.score(X_train.reshape(-1,1),y_train)
	b=np.array([])
	if score < 0.60:
		b=np.array([str(i) for i in range(2001,2016)])
		y = list(y)
		years = list(b)
		year = 2015
		output = "Can't predict"
	else:
		for j in range(2016,year+1):
			prediction = linear_regression.predict(np.array([[j]]))
			if(prediction < 0):
				prediction = 0
			y = np.append(y,prediction)
		b=np.array([str(i) for i in range(2001,year+1)])
		y = list(y)
		years = list(b)
		output = ""
	if output:
		print(output)
	else:
		print(y)
	print(b)
	return (y,years,output)

def pred_crime_plot(state,crime,x,y):
    plt.figure(figsize=(10,10)) 
    plt.grid(True)
    plt.xticks(fontsize=8)
    plt.plot(y,x)
    plt.xlabel('Years')
    plt.ylabel('No. of '+crime+' Cases in '+state)
    plt.title(crime)
    plt.savefig('static/images/plot.png')
