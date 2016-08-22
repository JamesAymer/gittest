import pandas, plotly

__version__ = 2.1

def version():
    return "data_handler.py V{}".format(__version__)

#crops the string date label and casts into a float
def convert_date_to_year(string):
    return float(string[0:4])

'''Returns a sorted reindexed dataframe'''
def sort_descending(input_dataframe):
    
    #sorts
    output_dataframe = input_dataframe.sort_index(ascending=False)
    
    #reindexes
    output_dataframe.reset_index(inplace=True, drop=True)
    
    return output_dataframe

'''Inserts new points between each datapoint, and uses interpolation to approximate those new values'''
def insert_n(input_dataframe, n=10):
    
    #this function is designed around DataFrames
    import pandas
    
    #finds the length of the dataset, and subtracts 2 in order to skip the last datapoint
    length = len(input_dataframe)-2
    
    #insertion works only by column, therefore the dataset is first transposed
    inserted = input_dataframe.T
    
    #calculates the distance between each inserted point
    step = 1/n
    
    #iterates through the dataset point by point (date by date)
    for date_i in range(length):
        
        #finds the id (name) prefix of the new column
        date_i = length-date_i
        
        #for each datapoint, insert n-1 (omitting the last point because it would overlap with the next point)
        for j in range(n-1):
            
            #finds id suffix of the new column
            i = j*step+step
            
            #does the insertion
            inserted.insert(date_i,str(date_i+(1-i)),[inserted[date_i]['Date']+(i), None])
            
    #converts the dataset back into a 2 dimentional dataframe
    inserted = inserted.T.reset_index(drop=True)

    #sorts the dates to make sure the data is in order
    inserted.sort_values("Date")
    inserted.reset_index(drop=True, inplace=True)
    
    #does the interpolation
    output = inserted.interpolate()
    
    return output

"""Splits the input(X), target(Y) and time(T) matrices into a training and testing set"""
def train_test_split(X, Y, T, split, overlap=0):
    train = round(len(X)*split)
    test = len(X)
    
    X_train = X.loc[0:train,:]
    Y_train = Y.loc[0:train]
    T_train = T.loc[0:train]

    X_test  = X.loc[train+overlap:test,:]
    Y_test = Y.loc[train+overlap:]
    T_test = T.loc[train+overlap:]
    
    return (X_train, Y_train, T_train, X_test, Y_test, T_test)
    
"""Creates a list of shifted data matrices (X,Y and T)"""
def time_shift(X, Y, T, shift=10):

    X_z = list()
    Y_z = list()
    T_z = list()

    #insert a time shift
    for i in range(shift):
        X_z.append(X.loc[i:,:].reset_index())
        del X_z[i]["index"]
        Y_z.append(Y[i:].reset_index())
        del Y_z[i]["index"]
        T_z.append(T[i:].reset_index())
        del T_z[i]["index"]
        
    #all lists must have the same length
    
    #new lists will all have the same length
    X_z_ = list()
    Y_z_ = list()
    T_z_ = list()
    
    
    #maximum size should be equal to the size of the smallest list
    max_list = min([len(X_z[i]) for i in range(len(X_z))])
    for X_ in X_z:
        while len(X_) > max_list:
            X_ = X_.loc[:len(X_)-2,:]
        X_z_.append( X_ )
        
    max_list = min([len(Y_z[i]) for i in range(len(Y_z))])  
    for i, Y_ in enumerate(Y_z):
        while len(Y_) > max_list:
            Y_ = Y_[:-1]
        Y_z_.append( Y_ )
        
    max_list = min([len(T_z[i]) for i in range(len(T_z))])  
    for i, T_ in enumerate(T_z):
        while len(T_) > max_list:
            T_ = T_[:-1]
        T_z_.append( T_ )
    
    return (X_z_, Y_z_, T_z_)    
    
    
'''LEGACY FUNCTION'''
'''NO LONGER REQUIRED'''
''' Imports data from the plot.ly server into a panda DataFrame '''
def plotly_Extraction(plotID, userID="jamo87", password="7kuadfxkyq", axis=None):
    
    plotly.plotly.sign_in(userID, password)   
    
    extracted_figure = plotly.plotly.get_figure(userID, plotID, raw=False)
    
    initial_dataframe = extracted_figure.to_dataframe()
    
    x_columns = dict()
    y_columns = dict()
    
    x_axis = []
    
    for i in range( len(initial_dataframe.axes[1]) ):
        
        if initial_dataframe.axes[1][i][-1] == 'x':
            label = initial_dataframe.axes[1][i][0:-2]
            values = initial_dataframe.values[:,i].tolist()
            x_columns[label] = values
            x_axis = values
            
        elif initial_dataframe.axes[1][i][-1] == 'y':
            label = initial_dataframe.axes[1][i][0:-2]
            values = initial_dataframe.values[:,i].tolist()
            y_columns[label] = values
            
    if axis == 'x':
        return pandas.DataFrame(data=x_columns)
    elif axis == 'y':
        return pandas.DataFrame(data=y_columns)
    else:
        return pandas.DataFrame(data=y_columns, index=x_axis)