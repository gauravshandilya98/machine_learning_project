# machine_learning_project
A Study of societal Impact on Academic Research Using Machine Learning Algorithm
# impot_csv_file
import pandas as pd
data=pd.read_excel(io='/content/projectdata.xls' )
print(data)
# read content of file
df1=pd.read_excel(io='/content/projectdata.xls')
# read and display content of file
df=pd.read_excel(io='/content/projectdata.xls')
display(df)
# which sponsoring gave most funds
import pandas as pd

# Define a dictionary containing employee data
data=pd.read_excel(io='/content/projectdata.xls')


# Convert the dictionary into DataFrame
df = pd.DataFrame(data)

# select two columns
df[['Sponsoring Agency', 'Sanctioned Fund']]
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a dataset named 'funding_data.csv' with columns 'Agency' and 'Funding'
df=pd.read_excel(io='/content/sponsered1.xls' )

# Group the data by 'Agency' and calculate the total funding for each agency
agency_funding = df.groupby('Sponsoring Agency')['Sanctioned Fund'].sum().nlargest(20)

# Plotting the bar graph
plt.bar(agency_funding.index, agency_funding.values)
plt.xlabel('Sponsoring Agency')
plt.ylabel('Sanctioned Fund')
plt.title('Top 20 Sponsoring Agencies by Funding')
plt.xticks(rotation=90)

# Display the graph
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a dataset named 'projects.csv' with columns 'Department' and 'Project'
df=pd.read_excel(io='/content/sponsered1.xls' )

# Group the data by 'Department' and count the number of projects in each department
department_projects = df['Emp Department'].value_counts()

# Plotting the bar graph
plt.bar(department_projects.index, department_projects.values)
plt.xlabel('Department')
plt.ylabel('Number of Projects')
plt.title('Number of Projects by Department')
plt.xticks(rotation=90)

# Display the graph
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a dataset named 'projects.csv' with columns 'Research Area' and 'Project'
df=pd.read_excel(io='/content/sponsered1.xls' )

# Group the data by 'Research Area' and count the number of projects in each area
research_area_projects = df['Main Research Area(s)'].value_counts().nlargest(10)

# Plotting the bar graph
plt.bar(research_area_projects.index, research_area_projects.values)
plt.xlabel('Research Area')
plt.ylabel('Number of Projects')
plt.title('Top 10 Research Areas by Number of Projects')
plt.xticks(rotation=90)

# Display the graph
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read the dataset from a CSV file
df1=pd.read_excel(io='/content/sponsered2.xls')

# Select the attributes for clustering
attributes = ['Sanctioned Fund', 'Year']

# Extract the attribute data for clustering
X = df[attributes]

# Specify the number of clusters
k = 2

# Perform K-means clustering
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(X)

# Get the cluster labels
labels = kmeans.labels_

# Add the cluster labels to the dataframe
df['Cluster'] = labels

# Visualize the clusters
plt.scatter(df['Sanctioned Fund'], df['Year'], c=labels, cmap='viridis')
plt.xlabel('Sanctioned Fund')
plt.ylabel('Year')
plt.title('K-means Clustering')
plt.colorbar(label='Cluster')
plt.show()
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Read the dataset from a CSV file
df1=pd.read_excel(io='/content/sponsered2.xls')

# Select the predictor variables (independent variables)
predictors = ['Sanctioned Fund']

# Select the target variable (dependent variable)
target = 'Year'

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df[predictors], df[target], test_size=0.2, random_state=0)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Print the evaluation metrics
print('Mean Squared Error:', mse)
print('R-squared:', r2)
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Read the dataset from a CSV file
df1=pd.read_excel(io='/content/sponsered2.xls')

# Select the predictor variables (independent variables)
predictors = ['Sanctioned Fund']

# Select the target variable (dependent variable)
target = 'Year'

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df[predictors], df[target], test_size=0.2, random_state=0)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Plot the predicted and actual values
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, predictions, color='red', label='Predicted')
plt.xlabel('Sanctioned Fund')
plt.ylabel('Year')
plt.title('Linear Regression: Predicted vs Actual')
plt.legend()
plt.show()
