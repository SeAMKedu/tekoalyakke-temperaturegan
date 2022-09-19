# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 12:56:49 2022

@author: Toni Takala
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import re

DEBUG = False

# Change the filepath if directory structure is changed after unzipping files...
DATA_FILE = "./data/Esimuotoiltu_data_csv_versio.csv"
initial_day = datetime.strptime("2021-05-01T00:00:01Z", '%Y-%m-%dT%H:%M:%SZ')

def readData(filename):
    print(f"Reading {filename}")
    df = pd.read_csv(filename)

    print(f"Parsing timedata")
    Parsed_timedata_from_csv = []
    for i in range(len(df.values)):
        dt = datetime.strptime(re.sub('\\..*','',df.values[i][0].split(';')[0]), '%Y-%m-%dT%H:%M:%S')
        Parsed_timedata_from_csv.append(dt)

    print(f"Classifying data")
    Unique_day_indexes = []
    Unique_day_index = 0
    Rolling_time_index = 0
    Prev_date = None
    for i in range(len(Parsed_timedata_from_csv)):
        dt = Parsed_timedata_from_csv[i]
        value = float(df.values[i][0].split(";")[1])
        
        # Set numeric classes for Temperature and Humidity
        T_or_H_value = -1
        if (df.values[i][0].split(";")[2] == 'T'):
            T_or_H_value = 0
        if(df.values[i][0].split(";")[2] == 'H'):
            T_or_H_value = 1
            
        # In the data, humidity changes to temperature at index 129122.
        # Only pick temperature measurements
        if Prev_date is not None and T_or_H_value == 0:
            if (Prev_date.year == dt.year) and (Prev_date.month == dt.month) and (Prev_date.day == dt.day):
                # Same date
                Rolling_time_index += 1
                Unique_day_indexes.append([Unique_day_index, Rolling_time_index, value, T_or_H_value] )            
            else:
                Unique_day_index += 1
                Rolling_time_index = 0
                # Add a record to table:
                Unique_day_indexes.append([Unique_day_index, Rolling_time_index, value, T_or_H_value])
        if Prev_date is None and T_or_H_value == 0:    
            # Add a record to table:
            Unique_day_indexes.append([Unique_day_index, Rolling_time_index, value, T_or_H_value])
    
        Prev_date = dt

    return Unique_day_indexes


def teachGAN(Unique_day_indexes, epochs = 125):
    print(f"Teaching GAN")
    TEMPERATURES = Unique_day_indexes[:]

    # GAN - Data Generator:
    from ctgan import CTGANSynthesizer
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    ganData = np.array(TEMPERATURES)
    ganData = ganData[0:288*10]

    # Because data is numpy array, columns that are discrete need to be noted by their indices
    discrete_columns = [ 0, 1, 3 ]

    # Hyperparameters adjustments are super important for the end results
    # Developer will always need to find most fitting parameters for training to get any good results!
    ctgan = CTGANSynthesizer(embedding_dim=128, discriminator_dim=(32, 16, 16, 4), generator_dim=(32,12,4), batch_size=100, verbose=True)
    ctgan.fit(ganData, discrete_columns=discrete_columns, epochs=epochs)

    return ctgan


def sortUniqueDayIndexData(Unique_day_indexes):
    print(f"Sorting indices")
    tempdata = []
    dateindexes = np.unique(np.round(np.array(Unique_day_indexes)).astype(int).T[0])
    a = {}

    for i in range(len(dateindexes)):
        for n in range(len(Unique_day_indexes)):
            if Unique_day_indexes[n][0] == dateindexes[i]:
                tempdata.append(Unique_day_indexes[n])
        
        # After setting up tempdata we add it into the dict:
        a[(i+1)] = tempdata
        tempdata = []
        
        if i%50==0:
            print("Sorted data for dates: ",(i+1))
        
    print(f"Sorted data for dates: {len(dateindexes)}")

    return a
    

def findTrueValueForDateAndTimeIndex(a, dateindex, timeindex):
    if DEBUG: print("Find for date and time index: ", dateindex, " & ", timeindex)
    if dateindex in a and timeindex in np.array(a[dateindex]).T[1].astype(int):
        return a[dateindex][timeindex]
    else:
        return None


def findSampleStddevFromGANResults(ctgan, a, max_attempts_to_find_value=25, sample_size_for_unique_vals=10):
    errors = {}
    
    synthetic_sample = ctgan.sample(sample_size_for_unique_vals)
    if DEBUG: print("Synthetic (original) data sample: ", synthetic_sample)
    synthetic_sample =  [np.round(np.array(synthetic_sample).T[0]).astype(int), np.round(np.array(synthetic_sample).astype(int).T[1]), np.array(synthetic_sample).T[2], np.array(synthetic_sample).astype(int).T[3]]
    
    synthetic_sample_uniques = getValuesOnce(synthetic_sample[0])
    if DEBUG: print("Unique day samples given from GAN: ", synthetic_sample_uniques)  
    if DEBUG: print("Synthetic data sample: ", synthetic_sample)
    found_value = None
    
    for i in range(len(synthetic_sample_uniques)):  
        found_value = None
        # print("Is in a -> ", synthetic_sample_uniques[i], " -> ", synthetic_sample_uniques[i] in a)
        if synthetic_sample_uniques[i] in a:
            value_index = np.where(synthetic_sample[0] == synthetic_sample_uniques[i])[0]
            found_value = findTrueValueForDateAndTimeIndex(a, synthetic_sample[0][value_index[0]], synthetic_sample[1][value_index[0]])
            if found_value != None: 
                
                if found_value[0] not in errors:
                    errors[found_value[0]] = []
                if DEBUG: print("Found value --> ", found_value, " on index(es) --> ", value_index)
                if DEBUG: print("Synthetic sample data: ", synthetic_sample)
                synth_value_squared = np.power(synthetic_sample[2][value_index[0]],2)
                found_value_squared = np.power(found_value[2],2)
                if DEBUG: print("Synthetic squared: ", synth_value_squared, " True value squared: ", found_value_squared)
                # Calculate squared error delta
                errors[found_value[0]].append(np.sqrt(np.absolute((synth_value_squared - found_value_squared))))
            else:
                if DEBUG: print("Did not find real value for GAN sample error calculation!")
                if DEBUG: print("Trying again... (With new temporary samples)")
                temp_synthetic_sample = ctgan.sample(500)
                #print("Synthetic (original) data sample: ", temp_synthetic_sample)
                temp_synthetic_sample =  [np.round(np.array(temp_synthetic_sample).T[0]).astype(int), \
                                          np.round(np.array(temp_synthetic_sample).astype(int).T[1]), \
                                          np.array(temp_synthetic_sample).T[2], \
                                          np.array(temp_synthetic_sample).astype(int).T[3] \
                                          ]
                    
                temp_synthetic_sample_uniques = getValuesOnce(temp_synthetic_sample[0])
                for n in range (max_attempts_to_find_value):
                    for k in range(len(temp_synthetic_sample_uniques)):  
                        found_value = None
                        # DEBUG print("Is (temp) in a -> ", temp_synthetic_sample_uniques[k], " : ", temp_synthetic_sample_uniques[k] in a)
                        if temp_synthetic_sample_uniques[k] in a:
                            value_index = np.where(temp_synthetic_sample[0] == temp_synthetic_sample_uniques[k])[0]
                            found_value = findTrueValueForDateAndTimeIndex(a, temp_synthetic_sample[0][value_index[0]], temp_synthetic_sample[1][value_index[0]])
                            # DEBUG print("Value found -->", found_value)
                            if found_value is not None:
                                
                                if found_value[0] not in errors:
                                    errors[found_value[0]] = []
                                whereindex = np.where(temp_synthetic_sample[0] == found_value[0])[0]
                                # DEBUG print("Where is: ", whereindex[0])
                                synth_value_squared = np.power(temp_synthetic_sample[2][whereindex[0]],2)
                                found_value_squared = np.power(found_value[2],2)
                                errors[found_value[0]].append(np.sqrt(np.absolute(synth_value_squared - found_value_squared)))
                                # DEBUG print("Value find iteration end")
                                break
                            else:
                                # DEBUG print("Passed iteration: ",n)
                                found_value=None
                    if found_value is not None:
                        break                                    
                            
    found_value = None
    if DEBUG: print("Errors dictionary: ", errors)
    return errors


# This function will run findSampleStddevFromGANResults() function 
# for multiple times to get multiple error samplesto get error averages...
def runErrorRounds(ctgan, a, rounds=1):
    errors = {}
    for i in range(rounds):
        if i%10==0:
            print("Gathering absolute error data... round: ",i)
        error_round = findSampleStddevFromGANResults(ctgan, a)
        for key in error_round:
            if key in errors:
                errors[key].append(error_round[key][0])
            else:
                errors[key] = error_round[key]
    return errors


def runErrorAveraging(error_dict):
    error_averages_dict = {}
    averaging_sample_counts_dict = {}
    for key in error_dict:
        error_averages_dict[key] = np.average(error_dict[key])
        averaging_sample_counts_dict[key] =  np.array(error_dict[key]).size
    return error_averages_dict, averaging_sample_counts_dict
        
    
def getValuesOnce(arr):
    array = arr
    countMap = {}
    
    for i in array:
        if countMap.get(i):
            countMap[i]+=1
        else:
            countMap[i]=1                
    
    duplicate = []
    unique = []
    superunique = []
    for i in countMap.keys():
        unique.append(i)
        if countMap[i] > 1:
            duplicate.append(i)
        if countMap[i] == 1:
            superunique.append(i)
    
    if DEBUG: print("Duplicate values : " , duplicate)  
    if DEBUG: print("Unique values : " ,unique)
    if DEBUG: print("Existing unique values :" , superunique)
    return unique


def printSampleDataComparison(ctgan, a):
    print("Sampled data comparison:\n")
    synthetic_df = ctgan.sample(10)
    # DEBUG print(synthetic_df)
    synthetic_data_check = [np.round(np.array(synthetic_df).T[0]).astype(int), np.round(np.array(synthetic_df).astype(int).T[1]), np.array(synthetic_df).T[2], np.array(synthetic_df).astype(int).T[3]]
    synthetic_data_check_dates = np.array(synthetic_data_check[0]).T
    synthetic_data_check_times = np.array(synthetic_data_check[1]).T
    truevals = []
    for d in range(len(synthetic_data_check_dates)):
        truevalue = findTrueValueForDateAndTimeIndex(a, synthetic_data_check_dates[d], synthetic_data_check_times[d])
        truevals.append(truevalue)
    if DEBUG: print(truevals)
    
    for i in range(len(truevals[0])):    
        time_start = initial_day + timedelta(days=int((synthetic_data_check_dates[i]-1))) + timedelta(minutes=int((synthetic_data_check_times[i]*5))-5)
        time_at = initial_day + timedelta(days=int((synthetic_data_check_dates[i]-1))) + timedelta(minutes=int((synthetic_data_check_times[i]*5)))
        timeinterval = ""+str(time_start)+" - "+str(time_at)
        print("Time: ",timeinterval," True value: ", truevals[i][2], " Generated value: ", round(synthetic_data_check[2][i], 1))
        
 
def run():
    Unique_day_indexes = readData(DATA_FILE)
    ctgan = teachGAN(Unique_day_indexes, 25)
    a = sortUniqueDayIndexData(Unique_day_indexes)
    errors = runErrorRounds(ctgan, a, 200)
    if DEBUG: 
        print("--- --- ---")
        print(errors)
        print("--- --- ---")
    err_avgs, sample_counts = runErrorAveraging(errors)
    print("Errors on average: ", err_avgs)
    print("Sample counts: ", sample_counts)
    print("Random pick test: ")
    printSampleDataComparison(ctgan, a)

run()
