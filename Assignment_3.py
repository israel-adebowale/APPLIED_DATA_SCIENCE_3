#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
data_copy['clusters'] = kmeans_y
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

# DATA FITTING

df_uk = pd.DataFrame({
    'Year' : agric_land_transpose.index,
    'UK' : agric_land_transpose['United Kingdom']
})
df_uk.reset_index(drop=True)


df_uk.info()


df_uk['Year'] = np.asarray(df_uk['Year'].astype(np.int64))


plt.plot(df_uk.Year, df_uk.UK)

def exponential(x, a, b):
    """Calculates exponential function with scale factor n0 and population growth  rate g."""
    x = x - 1961.0
    f = a * np.exp(b*x)
    return f


from scipy.optimize import curve_fit
param, cov = curve_fit(exponential, df_uk['Year'], df_uk['UK'],
                                                p0=(81.841855, 0.03))
print(param)
print(cov)


df_uk["fit"] = exponential(df_uk["Year"], *param)

plt.plot(df_uk['Year'], df_uk['UK'])
plt.plot(df_uk['Year'], df_uk['fit'])
plt.show()


