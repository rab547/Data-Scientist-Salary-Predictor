import pandas
import numpy

#Returns a pandas dataframe from a csv file located at filePath
def loadDataFrame(filePath):
    return pandas.read_csv(filePath)  

#Removes all rows within a dataframe which have null values
def removeNullsWithoutReplacement (dataframe):
    return dataframe.dropna()

# When null values exist, if it is numerical it will be replaced with the median of that column, 
# if it is categorical a value will be selected for it with the same probabilitydistribution as 
# the overall categorical values
def replaceNullsWithMedian (dataframe):
    #Replaces numerical columns with median
    for column in dataframe.select_dtypes(include=['number']).columns:
        dataframe[column] = dataframe[column].fillna(dataframe[column].median())
    
    #Replaces categorical columns with a random (matches how common each corresponding category is) value 
    for column in dataframe.select_dtypes(include=['object', 'category']).columns:
        value_counts = dataframe[column].value_counts(normalize=True)  
        categories = value_counts.index.tolist() 
        probabilities = value_counts.values.tolist() 
        
        dataframe[column] = dataframe[column].apply(
            lambda x: numpy.random.choice(categories, p=probabilities) if pandas.isna(x) else x
        )
    return dataframe

#Will replace categorical columns with One-Hot encoded columns
def columnOneHotEncoding(column, labels):  
    encodedColumn = pandas.get_dummies(column)  
    return encodedColumn.reindex(columns=labels, fill_value=0) * 1

#Will fill in any null values in the table and then return formatted summary statistics
def cleanDataAndReturnSummaryStatistics(dataframe):
    dataframe = replaceNullsWithMedian(dataframe)
    summaryStatistics = {}
    for column in dataframe.columns:
        currColumn = dataframe[column]
        if dataframe[column].dtype in ['object', 'category']:  

            summaryStatistics[column] = {
                'unique_values': currColumn.unique().tolist(),
                'value_counts': currColumn.value_counts().to_dict() 
            }
        else:  
            summaryStatistics[column] = {
                'mean': currColumn.mean(),
                'median': currColumn.median(),
                'min': currColumn.min(),
                'max': currColumn.max(),
                'std': currColumn.std()
            }
    return summaryStatistics