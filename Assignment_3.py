
# import the standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit

# this function returns a dataframe and its transpose
def read_data_excel(excel_url, sheet_name, drop_cols):
    """
    This function defines all the necessary attributes for reading the excel files and transposing it:
    excel_url: depicts the downloaded link of the file,
    sheet_name: states the name of the excel sheet
    drop_cols: columns to be dropped in the dataframe
    """
    df = pd.read_excel(excel_url, sheet_name=sheet_name, skiprows=3)
    df = df.drop(columns=drop_cols)
    df.set_index('Country Name', inplace=True)
    return df, df.transpose()


# the url below indicates agricultural land (% of land area)
url_agric_land= 'https://api.worldbank.org/v2/en/indicator/AG.LND.AGRI.ZS?downloadformat=excel'
# the excel sheet name
sheet_name = 'Data'
# the columns to be dropped
drop_cols = ['Country Code', 'Indicator Name', 'Indicator Code', '1960','2021']
# the parameters are passed into the function and it returns the dataframe and its transpose
agric_land, agric_land_transpose = read_data_excel(url_agric_land, sheet_name, drop_cols)
agric_land.head()

# Year 1961 and 2020 are sliced for clustering
data_agric_land = agric_land.loc[:, ['1961','2020']]
print(data_agric_land)

# this function preprocesses data
def null_values(df):
    """"
    The function accepts the dataframe and it returns:
    the sum of the null values in each column, and
    the countries with null values
    """
    df_null_sum = df.isnull().sum()
    df_null_values = df[df.isna().any(axis=1)]
    return df_null_sum, df_null_values

# the sum of the null values in each column and the countries with null values are determined
agric_null_sum, agric_null_values = null_values(data_agric_land)
print('The sum of the null values are')
print(agric_null_sum)

print('The countries with null values in each column are:')
print(agric_null_values)

# this function returns a dataframe without null values
def dropna(df):
    """"This function drops the null values in a dataframe"""
    df = df.dropna()
    return df

# the dataframe is passed into the dropna function and a new dataframe without null values are returned
data_new =  dropna(data_agric_land)
print(data_new)

# the statistics of the new data is determined
data_new.describe()

# this function plots a scatter plot
def scatterplot(data, title, xlabel, ylabel, color):
    """
    The function defines the parameters the parameters for creating a scatter plot:
    data: the dataframe to be plot
    title: the title of the plot
    xlabel: this is the label of the x-axis
    ylabel: this is the y-axis label
    color: this is the color of the points on the plot
    """
    plt.figure(figsize=(16,12), dpi=200)
    plt.scatter(data[0], data[1], color=color)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.show()
    return

# the parameters for creating the scatter plot are listed and passed into the function
data = [data_new['1961'], data_new['2020']]
title = 'Scatterplot of Agricultural land of world countries between 1995 and 2015'
xlabel = 'Year 1961'
ylabel = 'Year 2020'
color = 'blue'
scatterplot(data, title, xlabel, ylabel, color)

# a boxplot of each years are created to show if there are outliers
col = ['1961', '2020']
data_new.boxplot(col)

# a function is created which uses min-max normalization
def scaled_data(data_array):
    """"The function accepts the dataframecand it normalises the data and returns the points from 0 to 1  """
    min_val = np.min(data_array)
    max_val = np.max(data_array)
    scaled = (data_array-min_val) / (max_val-min_val)
    return scaled

# a function is created which scales each column of the dataframe
def norm_data(data):
    """"The function accepts the dataframe and it return the scaled inputs of each column"""
    for col in data.columns:
        data[col] = scaled_data(data[col])
    return data

# a copy of the new data is created
data_copy = data_new.copy()

# the new data is passed into the norm_data function and returns the scaled data points
data_norm = norm_data(data_new)
data_norm

# the scaled data frame is converted into an array
x_data = data_norm[['1961', '2020']].values
print(x_data)

# a function is created for the elbow method and uses the within cluster sum of squares to determine ideal number of clusters
def elbow_method(x_data, title, xlabel, ylabel):
    """
    This function creates the elbow graph which determines the ideal number of clusters and it accepts:
    x_data: this is the array which is to be used for clustering
    title: this is the title of the elbow
    xlabel: this is the label of the x-axis
    ylabel: this is the y-axis label
    """    
    wcss = [] # within cluster sum of squares
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=200, n_init=10, random_state=2)
        kmeans.fit(x_data)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10,8), dpi=200)
    plt.plot(range(1, 11), wcss) # visualize the elbow
    plt.title(title, fontsize=30)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.show()
    return

# the parameters for the elbow are passed into the function and the elbow is created
title = 'Elbow method for agricultural land'
xlabel = 'Number of clusters'
ylabel = 'wcss'

elbow_method(x_data, title, xlabel, ylabel)

# the ideal number of clusters is 4 and it is used to cluster the data points using the KMeans function
kmeans = KMeans(n_clusters=4, random_state=2)
kmeans.fit(x_data)
kmeans_y = kmeans.fit_predict(x_data)

# the centroid of the clusters are determined
centroid = kmeans.cluster_centers_
print('Centroid:')
print(centroid)

# the silhoutte_score is also determined
data_silhouette_score = silhouette_score(x_data, kmeans_y)
print('silhouette_score:')
print(data_silhouette_score)

# a scatterplot is plot to visualize the clusters and the centroids
plt.figure(figsize=(16,12), dpi=200)
plt.scatter(x_data[kmeans_y == 0, 0], x_data[kmeans_y == 0, 1], s = 50, c='black', label='cluster 1')
plt.scatter(x_data[kmeans_y == 1, 0], x_data[kmeans_y == 1, 1], s = 50, c='orange', label='cluster 2')
plt.scatter(x_data[kmeans_y == 2, 0], x_data[kmeans_y == 2, 1], s = 50, c='green', label='cluster 3')
plt.scatter(x_data[kmeans_y == 3, 0], x_data[kmeans_y == 3, 1], s = 50, c='blue', label='cluster 4')
plt.scatter(centroid[:, 0], centroid[:,1], s = 200, c = 'red', marker='x', label = 'Centroids')
plt.title('Scatterplot showing clusters and centroids of agricultural land between 1961 and 2020', fontsize=20, fontweight='bold')
plt.xlabel('Year 1961', fontsize=20)
plt.ylabel('Year 2020', fontsize=20)
plt.legend(bbox_to_anchor=(1.11,1.01))
plt.show()

# a dataframe called clusters is created which stores the clusters of each countries
data_copy['clusters'] = kmeans_y
print(data_copy)

# five countries from the first cluster is extracted for analysis
cluster1 = data_copy[(data_copy['clusters']==0)]
cluster_bar1 = cluster1.iloc[:5,:]
print(cluster_bar1)

# five countries from the second cluster is extracted 
cluster2 = data_copy[(data_copy['clusters']==1)]
cluster_bar2 = cluster2.iloc[:5,:]
print(cluster_bar2)

# five countries from the third cluster is extracted
cluster3 = data_copy[(data_copy['clusters']==2)]
cluster_pie1 = cluster3.iloc[:5,:]
cluster_pie1

# five countries from the fourth cluster is extracted
cluster4 = data_copy[(data_copy['clusters']==3)]
cluster_pie2 = cluster4.iloc[:5,:]
print(cluster_pie2)

# this function creates a multiple bar plot
def barplot(labels_array, width, y_data, y_label, label, title):
   """
   This function defines a grouped bar plot and it takes the following attributes:
   labels_array: these are the labels of barplots of the x-axis which depicts countries of the indicator to be determined
   width: this is the size of the bar
   y_data: these are the data to be plotted
   y_label: this is the label of the y-axis
   label: these are the labels of each grouped plots which depicts the years of the indicator 
   title: depicts the title of the bar plot.
   """
   
   x = np.arange(len(labels_array)) # x is the range of values using the length of the label_array
   fig, ax  = plt.subplots(figsize=(16,12))
   
   plt.bar(x - width, y_data[0], width, label=label[0]) 
   plt.bar(x, y_data[1], width, label=label[1])
   
   
   plt.title(title, fontsize=20)
   plt.ylabel(y_label, fontsize=20)
   plt.xlabel(None)
   plt.xticks(x, labels_array)
   
   sns.despine(bottom=True) #seaborn function despine is used to take away the top and the right spine of the function


   plt.legend()
   ax.tick_params(bottom=False, left=True)

   plt.show()
   return

# the parameters for producing grouped bar plots of Agricultural land (% of land area)
labels_array = ['Aruba', 'UAE', 'American Samoa', 'Antigua and Barbuda', 'Bahrain']
width = 0.2 
y_data = [cluster_bar1['1961'], 
          cluster_bar1['2020']]
y_label = '% Agricultural land'
label = ['Year 1961', 'Year 2020']
title = 'Agricultural land (% of land Area) of five countries for cluster 0'

# the parameters are passed into the defined function and produces the desired plot
barplot(labels_array, width, y_data, y_label, label, title)


# the parameters for producing grouped bar plots of Agriicultural land (% of land area)
labels_array = ['Africa Western and Central', 'Arab World', 'Austria', 'Benin', 'Burkina Faso']
width = 0.2 
y_data = [cluster_bar2['1961'], 
          cluster_bar2['2020']]
y_label = '% Agricultural land'
label = ['Year 1961', 'Year 2020']
title = 'Agricultural land (% of land Area) of five countries for cluster 1'

# the parameters are passed into the defined function and produces the desired plot
barplot(labels_array, width, y_data, y_label, label, title)


# this function creates a pie chart
def pie_chart(pie_data, explode, label, title, color):
    """a function is defined for a pie chart which accepts pie_data, explode, label, title and color as parameters """
    plt.figure(figsize=(10,8))
    plt.title(title, fontsize=20)
    plt.pie(pie_data, explode = explode, labels=label, colors=color, autopct='%0.2f%%')
    plt.legend(bbox_to_anchor=(1.01,1.01))
    plt.show()
    
    return

# the parameters for the pie chart of the third cluster for the year 1961 are passed into the function
pie_data = cluster_pie1['1961']
label = cluster_pie1.index
title = 'Pie chart for Year 1961 on third cluster'
color = ['blue', 'red', 'yellow', 'indigo', 'green']
explode = (0, 0, 0, 0.2 , 0)

pie_chart(pie_data, explode, label, title, color)

# the parameters for the pie chart of the fourth cluster for the year 2020 are passed into the function
pie_data = cluster_pie2['2020']
label = cluster_pie2.index
title = 'Pie chart for Year 1961 on fourth cluster'
color = ['blue', 'red', 'yellow', 'indigo', 'green']
explode = (0, 0.1, 0, 0, 0)

#The defined pie chart function is then passed for visualization
pie_chart(pie_data, explode, label, title, color)

# DATA FITTING 

# a dataframe is created using using United Kingdom as its reference
df_uk = pd.DataFrame({
    'Year' : agric_land_transpose.index,
    'UK' : agric_land_transpose['United Kingdom']
})
df_uk.reset_index(drop=True)

# the data type is printed out
df_uk.info()

# the data type of the year column is changed from object to int64
df_uk['Year'] = np.asarray(df_uk['Year'].astype(np.int64))

plt.figure(figsize=(12,8), dpi=100)
plt.plot(df_uk.Year, df_uk.UK,)
plt.title('Time series plot showing change in Agricultural land in the United Kingdom', fontsize=20)
plt.xlabel('Year', fontsize=15)
plt.ylabel('% agricultural land', fontsize=15)
plt.show()

# this function returns a polynomial 
def polynomial(x, w, y, z):
    """
    Calculates a polynomial function which accepts:
    x: this is the years of the data
    w,y,z are the constants which define the equation
    """
    f = w + y*x + z*x**2 
    return f

# the curve_fit function is used to fit the data points 
# and it accepts the polynomial function, years and United Kingdom Column as parameters
param, cov = curve_fit(polynomial, df_uk['Year'], df_uk['UK'])
print('Param:')
print(param)

print('Covariance:')
print(cov)

# the function calculates the error ranges
def err_ranges(x, y):
    """
    This function calculates the confidence intervals of the data points
    it returns the lower and upper limits of the data points
    """
    ci = 1.96 * np.std(y)/np.sqrt(len(x))
    lower = y - ci
    upper = y + ci
    return lower, upper

x = df_uk['Year']
y = df_uk['UK']

# an array of years ranging from 1961 to 2041 is created for forecast
year = np.arange(1961, 2041)
# the forecast for the next 20 years is calculated using the polynomial function
# the polynomial function accepts year and *param as arguments
forecast = polynomial(year, *param)
# the lower and upper limits of the forecast are calculated using the err_ranges function
lower, upper = err_ranges(year, forecast)

# this is a multiple plot which shows the change in agricultural land in UK
# it also shows the forecast for the year 20 years
# finally, it shows the confidence intervals across 191 to 2041
plt.figure(figsize=(16,12))
plt.plot(x, y, label='UK')
plt.plot(year, forecast, label='forecast')
#plt.fill_between(df_uk['Year'], lower, upper, color="blue", alpha=0.2)
plt.fill_between(year, lower, upper, color="blue", alpha=0.2)
plt.title('A plot showing the predictions and the error ranges of Agricultural land of United Kingdom between 1961 and 2041', fontsize=20)
plt.xlabel('year', fontsize=20)
plt.ylabel('Population', fontsize=20)
plt.legend()
plt.show()

# a dataframe is created for the forecast of the agricultural land in United Kingdom
df_forecast = pd.DataFrame({'Year': year,
                            'Forecast': forecast
                           })

# the next forecast for the next 20 years is extracted 
df_forecast_twenty_years = df_forecast.iloc[60:,:]
print(df_forecast_twenty_years)




