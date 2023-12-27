import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

raw_data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv') # Read data from URL

#print(raw_data.head()) # print first 5 rows of data
#print(raw_data.dtypes)
#count_row = raw_data.shape[0]  # Get the number of rows
#print(count_row)

# Check any NA (missing) values
raw_data_ = raw_data.copy()

raw_data_.replace('nan', np.nan, inplace=True) # Replace all 'nan' strings with np.nan, best practice for handling missing values as lots of pandas functions can handle np.nan.
for col in raw_data_.columns: # Loop through each column.
    print(f"{col}: {raw_data_[col].isna().sum()}") # Check for missing values in each column.

print(raw_data_.describe(include='all'))


# Let's find the average sepal length for each species
sepal_length_mean = raw_data_.groupby('species')['sepal_length'].mean() # Group the data by deal category, then find the mean of order cost for each group
#print(sepal_length_mean) # Print the means
sepal_width_mean = raw_data_.groupby('species')['sepal_width'].mean()

# Let's continue on and explore if there is a relationship between species and sepal_length, sepal_width, petal_width, petal_length.
plt.bar(raw_data_['species'].unique(), sepal_length_mean) 
plt.xlabel('Species') # Label the x axis.
plt.ylabel('Mean Sepal Length') # Label the y axis.
plt.title('Mean Sepal Length by Species') # Title the plot.
#plt.show() # Show the plot.

corr_value = np.corrcoef(raw_data_['sepal_width'], raw_data_['petal_length']) # Find the correlation coefficient between 
print(f"Correlation coefficient sepal_width to petal_length: {corr_value[0, 1]}") 

corr_value2 = np.corrcoef(raw_data_['sepal_length'], raw_data_['petal_width']) # Find the correlation coefficient between order cost and deal category.
print(f"Correlation coefficient sepal_length to petal_width: {corr_value2[0, 1]}") 

corr_value3 = np.corrcoef(raw_data_['sepal_length'], raw_data_['petal_length']) # Find the correlation coefficient between order cost and deal category.
print(f"Correlation coefficient sepal_length to petal_length: {corr_value3[0, 1]}") 

corr_value4 = np.corrcoef(raw_data_['sepal_width'], raw_data_['petal_width']) # Find the correlation coefficient between order cost and deal category.
print(f"Correlation coefficient sepal_width to petal_width: {corr_value4[0, 1]}") 

raw_data_.boxplot(column='petal_length', by='species') 
plt.ylabel('Petal Length') # Set the y-axis label
plt.xlabel('Species') # Set the x-axis label
plt.xticks(rotation=90) # Rotate the x-axis labels by 90 degrees
plt.subplots_adjust(bottom=0.4) # Adjust the bottom of the plot to make room for the x-axis labels
plt.suptitle('') # Remove the default super-title
plt.title('Boxplot of Petal Length by Species') # Set the title of the plot
plt.show() # Show the plot

colours = {'setosa':'red', 'versicolor':'blue', 'virginica':'green'} # Create a dictionary of colours for each species
plt.scatter(raw_data_['petal_length'], raw_data_['petal_width'], c=raw_data_['species'].map(colours))
#plt.scatter(raw_data_['petal_length'], raw_data_['petal_width'], c=raw_data_['species'].apply(lambda x: colours[x]))
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Petal Length vs Petal Width')
plt.show()