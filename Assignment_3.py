#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score




def read_data_excel(excel_url, sheet_name, drop_cols):
    df = pd.read_excel(excel_url, sheet_name=sheet_name, skiprows=3)
    df = df.drop(columns=drop_cols)
    df.set_index('Country Name', inplace=True)
    return df, df.transpose()




url_agric_land= 'https://api.worldbank.org/v2/en/indicator/AG.LND.AGRI.ZS?downloadformat=excel'
sheet_name = 'Data'
drop_cols = ['Country Code', 'Indicator Name', 'Indicator Code', '1960','2021']
agric_land, agric_land_transpose = read_data_excel(url_agric_land, sheet_name, drop_cols)
agric_land.head()


data_agric_land = agric_land.loc[:, ['1961','2020']]
print(data_agric_land)



def null_values(df):
    df_null_sum = df.isnull().sum()
    df_null_values = df[df.isna().any(axis=1)]
    return df_null_sum, df_null_values



agric_null_sum, agric_null_values = null_values(data_agric_land)
print('The sum of the null values are')
print(agric_null_sum)
print(agric_null_values)



def dropna(df):
    df = df.dropna()
    return df




data_new =  dropna(data_agric_land)
print(data_new)



data_new.describe()



def scatterplot(data, title, xlabel, ylabel, color):
    plt.figure(figsize=(16,12))
    plt.scatter(data[0], data[1], color=color)
    plt.title(title, fontsize=30)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.show()
    return

data = [data_new['1961'], data_new['2020']]
title = 'Scatterplot of Agricultural land of world countries between 1995 and 2015'
xlabel = 'Year 1961'
ylabel = 'Year 2020'
color = 'blue'
scatterplot(data, title, xlabel, ylabel, color)


col = ['1961', '2020']
data_new.boxplot(col)


def scaled_data(data_array):
    min_val = np.min(data_array)
    max_val = np.max(data_array)
    scaled = (data_array-min_val) / (max_val-min_val)
    return scaled

def norm_data(data):
    for col in data.columns:
        data[col] = scaled_data(data[col])
    return data


data_copy = data_new.copy()


data_norm = norm_data(data_new)
print(data_norm)

x_data = data_norm[['1961', '2020']].values
print(x_data)


def elbow_method(x_data, title, xlabel, ylabel):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=200, n_init=10, random_state=2)
        kmeans.fit(x_data)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10,8), dpi=100)
    plt.plot(range(1, 11), wcss) # visualize the elbow
    plt.title(title, fontsize=30)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.show()
    return


title = 'Elbow method for agricultural land'
xlabel = 'Number of clusters'
ylabel = 'wcss'

elbow_method(x_data, title, xlabel, ylabel)

kmeans = KMeans(n_clusters=4, random_state=2)
kmeans.fit(x_data)
kmeans_y = kmeans.fit_predict(x_data)

print(data_copy)


centroid = kmeans.cluster_centers_
print(centroid)

data_silhouette_score = silhouette_score(x_data, kmeans_y)
print(data_silhouette_score)


plt.figure(figsize=(16,12))
plt.scatter(x_data[kmeans_y == 0, 0], x_data[kmeans_y == 0, 1], s = 50, c = 'black', label='cluster 1')
plt.scatter(x_data[kmeans_y == 1, 0], x_data[kmeans_y == 1, 1], s = 50, c = 'orange', label='cluster 2')
plt.scatter(x_data[kmeans_y == 2, 0], x_data[kmeans_y == 2, 1], s = 50, c = 'green', label='cluster 3')
plt.scatter(x_data[kmeans_y == 3, 0], x_data[kmeans_y == 3, 1], s = 50, c = 'blue', label='cluster 4')
plt.scatter(centroid[:, 0], centroid[:,1], s = 200, c = 'red', marker='x', label = 'Centroids')
plt.title('Scatterplot showing clusters and centroids of agricultural land between 1961 and 2020', fontsize=20, fontweight='bold')
plt.xlabel('Year 1961', fontsize=20)
plt.ylabel('Year 2020', fontsize=20)
plt.legend(bbox_to_anchor=(1.11,1.01))
plt.show()

data_copy['clusters'] = kmeans_y
cluster1 = data_copy[(data_copy['clusters']==0)]
cluster_bar1 = cluster1.iloc[:5,:]
cluster_bar1

cluster2 = data_copy[(data_copy['clusters']==1)]
cluster_bar2 = cluster2.iloc[:5,:]
cluster_bar2

cluster3 = data_copy[(data_copy['clusters']==2)]
cluster_pie1 = cluster3.iloc[:5,:]
cluster_pie1

cluster4 = data_copy[(data_copy['clusters']==3)]
cluster_pie2 = cluster4.iloc[:5,:]
cluster_pie2

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

# the parameters for producing grouped bar plots of Urban population growth (annual %)
labels_array = ['Aruba', 'UAE', 'American Samoa', 'Antigua and Barbuda', 'Bahrain']
width = 0.2 
y_data = [cluster_bar1['1961'], 
          cluster_bar1['2020']]
y_label = '% Agricultural land'
label = ['Year 1961', 'Year 2020']
title = 'Agricultural land (% of land Area) of five countries for cluster 0'

# the parameters are passed into the defined function and produces the desired plot
barplot(labels_array, width, y_data, y_label, label, title)

# the parameters for producing grouped bar plots of Urban population growth (annual %)
labels_array = ['Africa Western and Central', 'Arab World', 'Austria', 'Benin', 'Burkina Faso']
width = 0.2 
y_data = [cluster_bar2['1961'], 
          cluster_bar2['2020']]
y_label = '% Agricultural land'
label = ['Year 1961', 'Year 2020']
title = 'Agricultural land (% of land Area) of five countries for cluster 1'

# the parameters are passed into the defined function and produces the desired plot
barplot(labels_array, width, y_data, y_label, label, title)

def pie_chart(pie_data, explode, label, title, color):
    """ Here I defined a function for a pie chart which accept pie_data, explode, label, title and color as parameters """
    plt.figure(figsize=(10,8))
    plt.title(title, fontsize=20)
    plt.pie(pie_data, explode = explode, labels=label, colors=color, autopct='%0.2f%%')
    plt.legend(bbox_to_anchor=(1.01,1.01))
    plt.show()
    
    return

pie_data = cluster_pie1['1961']
label = cluster_pie1.index
title = 'Pie chart for Year 1961 on third cluster'
color = ['blue', 'red', 'yellow', 'indigo', 'green']
explode = (0, 0, 0, 0.2 , 0)

#The defined pie chart function is then passed for visualization
pie_chart(pie_data, explode, label, title, color)

pie_data = cluster_pie2['1961']
label = cluster_pie2.index
title = 'Pie chart for Year 1961 on fourth cluster'
color = ['blue', 'red', 'yellow', 'indigo', 'green']
explode = (0, 0, 0.2, 0, 0)

#The defined pie chart function is then passed for visualization
pie_chart(pie_data, explode, label, title, color)

# DATA FITTING

df_uk = pd.DataFrame({
    'Year' : agric_land_transpose.index,
    'UK' : agric_land_transpose['United Kingdom']
})
df_uk.reset_index(drop=True)


df_uk.info()


df_uk['Year'] = np.asarray(df_uk['Year'].astype(np.int64))


plt.plot(df_uk.Year, df_uk.UK)

def polynomial(x, w, y, z):
    """Calculates a polynomial function which accepts:
    x: this is the years of the data
    w,y,z are the constants which define the equation
    """
    f = w + y*x + z*x**2 
    return f

from scipy.optimize import curve_fit
param, cov = curve_fit(polynomial, df_uk['Year'], df_uk['UK'])
print(param)
print(cov)

def err_ranges(x, y):
    ci = 1.96 * np.std(y)/np.sqrt(len(x))
    lower = y - ci
    upper = y + ci
    return lower, upper

x = df_uk["Year"]
y = df_uk["UK"]
lower, upper = err_ranges(x, y)

year = np.arange(1961, 2041)

forecast = polynomial(year, *param)

plt.figure(figsize=(16,12))
plt.plot(df_uk["Year"], df_uk["UK"], label="UK")
plt.plot(year, forecast, label="forecast")
plt.fill_between(df_uk['Year'], lower, upper, color="blue", alpha=0.2)
plt.title('A plot showing the predictions and the error ranges of Population of world countries between 1961 and 2041', fontsize=20)
plt.xlabel("year", fontsize=20)
plt.ylabel("Population", fontsize=20)
plt.legend()
plt.show()

df_forecast = pd.DataFrame({'Year': year,
                            'Forecast': forecast
                           })


df_forecast_twenty_years = df_forecast.iloc[60:,:]
df_forecast_twenty_years


