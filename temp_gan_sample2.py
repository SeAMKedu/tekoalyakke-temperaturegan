# -*- coding: utf-8 -*-
"""

@author: Mika Valkama
"""

from sdv.tabular import CTGAN
from sdv.metrics.tabular import DiscreteKLDivergence, KSComplement
from sdv.evaluation import evaluate
from datetime import datetime, date, timedelta
import time
from random import sample, seed
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# matplotlib 3.6.0 has a new annoying warning, so hiding it for now,
# if you use newer, comment two lines below and fix the issue
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # limit the amount of debug messages from tensorflow


def readData(filename, fraction=0.01, random_state=1234):
    print(f"Reading {filename}.")
    df = pd.read_csv(filename, header=None, names=["Timestamp", "Value", "Type"], parse_dates=[0], delimiter=";")

    # Only keep temperature measurements, the data also has humidity values
    df = df[df.Type == "T"]

    allRows = df.shape[0]

    # Take a sample from whole set, sort it and reset index
    df = df.sample(frac=fraction, random_state=random_state)
    df.sort_values(by=["Timestamp"], inplace=True)
    df.reset_index(inplace=True, drop=True)

    print(f"Using {(fraction * 100):.2f}% or {df.shape[0]} rows of {allRows}.\n")

    # Separate date from timestamp
    df["Date"] = pd.to_datetime(df["Timestamp"]).dt.date

    # Add date index
    di, unique = pd.factorize(df["Date"])
    df["DateIndex"] = di

    # Split year, month and day of month to own columns
    df["Year"] = pd.to_datetime(df["Timestamp"]).dt.year
    df["Month"] = pd.to_datetime(df["Timestamp"]).dt.month
    df["Day"] = pd.to_datetime(df["Timestamp"]).dt.day    

    # Add day of year column
    df["DayOfYear"] = df["Timestamp"].dt.dayofyear

    # Add seconds from midnight
    df["SecondsMidnight"] = ((df["Timestamp"] - df["Timestamp"].dt.normalize()) / pd.Timedelta('1 second')).astype(int)

    if DEBUG: 
        print(df.head())
        print(df.dtypes)

    return df


def teachGAN(df, gan_field_names, disdim, gendim, epochs = 125, batch_size=100, verbose=False):
    print(f"Teaching GAN with {epochs} epochs.")

    start = time.time()
    ctgan = CTGAN(field_names=gan_field_names, discriminator_dim=disdim, generator_dim=gendim, epochs=epochs, batch_size=batch_size, verbose=verbose, cuda=True)
    ctgan.fit(df)
    
    print(f"Took (H:MM:SS) {str(timedelta(seconds=(time.time() - start)))}\n")

    return ctgan
    

def find_neighbours(value, df, colname, glitch=False):
    try:
        if len(df) == 1:
            return [df.first_valid_index()] if not glitch else df.first_valid_index()

        exactmatch = df[df[colname] == value]
        if not exactmatch.empty:
            return exactmatch.index
        else:

            if df[df[colname] > value][colname].size and df[df[colname] < value][colname].size:
                lowerneighbour_ind = df[df[colname] < value][colname].idxmax()
                upperneighbour_ind = df[df[colname] > value][colname].idxmin()
                return [lowerneighbour_ind, upperneighbour_ind] 
            elif df[df[colname] > value][colname].size and not df[df[colname] < value][colname].size:
                upperneighbour_ind = df[df[colname] > value][colname].idxmin()
                return [upperneighbour_ind] 
            elif not df[df[colname] > value][colname].size and df[df[colname] < value][colname].size:
                lowerneighbour_ind = df[df[colname] < value][colname].idxmax()
                return [lowerneighbour_ind] 
    except KeyError as ke:
        pass
        


def doyStyle(row, comp, errors, plotSynthDf, plotMeasuredDf, verbose=False):
    # find entries that are close to or on date
    daily_idx = find_neighbours(row['DayOfYear'], comp, 'DayOfYear')
    # find entries that are close to time on that date
    real_idx = find_neighbours(row['SecondsMidnight'], comp.iloc[daily_idx], 'SecondsMidnight')

    # calculate avg 
    measuredAvg = comp[['DayOfYear', 'SecondsMidnight', 'Value']].iloc[real_idx].mean(axis=0)
    
    # Assume year to be 2022
    year = 2022
    strt_date = datetime(int(year), 1, 1)
    delta = timedelta(days=int(row['DayOfYear']) - 1) + timedelta(seconds=int(row['SecondsMidnight']))
    synth_date = strt_date + delta
    synth_date_str = synth_date.strftime("%Y-%m-%d %H:%M")

    measured_date = strt_date + timedelta(days=int(measuredAvg['DayOfYear']) - 1, seconds=int(measuredAvg['SecondsMidnight']))
    measured_date_str = measured_date.strftime("%Y-%m-%d %H:%M")

    error = measuredAvg['Value'] - row['Value']

    errors.append(error)

    plotSynthDf.loc[len(plotSynthDf)] = [synth_date, row['Value']]
    plotMeasuredDf.loc[len(plotMeasuredDf)] = [measured_date, measuredAvg["Value"]]

    if verbose: print(f"{synth_date_str} {float(row['Value']):>8.2f} <> {measured_date_str} {float(measuredAvg['Value']):>8.2f}\t{error:8.2f}")


def domStyle(row, comp, errors, plotSynthDf, plotMeasuredDf, verbose=False):
    # find entries that are close to or on date
    sliced = comp[(comp['Year'] == row['Year']) & (comp['Month'] == row['Month'])]

    if sliced.empty:
        sliced = comp[(comp['Month'] == row['Month'])]
    
    if sliced.empty:
        # no monthly data to compare against
        return

    daily = sliced[sliced['Day'] == row['Day']]

    day_idx = None

    if daily.empty:
        day_idx = find_neighbours(row['Day'], sliced, 'Day', glitch=True)
        daily = comp[['Year', 'Month', 'Day', 'SecondsMidnight', 'Value']].iloc[day_idx].mean(axis=0)

    measuredAvg = None

    if len(daily) > 1:
        # find entries that are close to timme on that date
        real_idx = find_neighbours(row['SecondsMidnight'], daily, 'SecondsMidnight')
        if real_idx == None:
            # something went wrong
            return

        # calculate avg 
        measuredAvg = comp[['Year', 'Month', 'Day', 'SecondsMidnight', 'Value']].iloc[real_idx].mean(axis=0)
    else:
        measuredAvg = daily.iloc[0]

    try:
        synth_date = datetime(int(row['Year']), int(row['Month']), int(row['Day']))
        synth_date = synth_date + timedelta(seconds=int(row['SecondsMidnight']))
        synth_date_str = synth_date.strftime("%Y-%m-%d %H:%M")

        measured_date = datetime(int(measuredAvg['Year']), int(measuredAvg['Month']), int(measuredAvg['Day']))
        measured_date = measured_date + timedelta(seconds=int(measuredAvg['SecondsMidnight']))
        measured_date_str = measured_date.strftime("%Y-%m-%d %H:%M")

        error = measuredAvg['Value'] - row['Value']
        errors.append(error)

        plotSynthDf.loc[len(plotSynthDf)] = [synth_date, row['Value']]
        plotMeasuredDf.loc[len(plotMeasuredDf)] = [measured_date, measuredAvg["Value"]]

        if verbose: print(f"{synth_date_str} {float(row['Value']):>8.2f} <> {measured_date_str} {float(measuredAvg['Value']):>8.2f}\t{error:8.2f}")
    except ValueError as ve:
        # if impossible date is generated, we ignore it
        pass



def diStyle(row, comp, errors, plotSynthDf, plotMeasuredDf, verbose=False):
    # find entries that are close to or on date
    daily_idx = find_neighbours(row['DateIndex'], comp, 'DateIndex')
    # find entries that are close to time on that date
    real_idx = find_neighbours(row['SecondsMidnight'], comp.iloc[daily_idx], 'SecondsMidnight')

    # calculate avg 
    measuredAvg = comp[['DateIndex', 'SecondsMidnight', 'Value']].iloc[real_idx].mean(axis=0)

    error = measuredAvg['Value'] - row['Value']

    errors.append(error)

    plotSynthDf.loc[len(plotSynthDf)] = [row["DateIndex"], row['Value']]
    plotMeasuredDf.loc[len(plotMeasuredDf)] = [measuredAvg["DateIndex"], measuredAvg["Value"]]

    if verbose: print(f"{int(row['DateIndex'])} {float(row['Value']):>8.2f} <> {int(measuredAvg['DateIndex'])} {float(measuredAvg['Value']):>8.2f}\t{error:8.2f}")


def eval(df, ctgan, samplesize, style, plotname, verbose=False):
    print(f"Evaluating with sample size of {samplesize}\n")
    
    synth = ctgan.sample(samplesize)

    synth.sort_values(by=style["sort"], inplace=True)
    synth = synth.astype(style["astypes"])
    
    if DEBUG: print(synth)

    comp = df[style["origin"]]
    comp = comp.astype(style["originastypes"])

    evaluation = evaluate(synth, comp, metrics=['KSComplement'], aggregate=False)
    print(f"{evaluation}\n")

    errors = []

    plotSynthDf = pd.DataFrame(columns=style["plotcolumns"])
    plotMeasuredDf = pd.DataFrame(columns=style["plotcolumns"])

    print(f"Synthetic                    Closest measured             Error")
    for index, row in synth.iterrows():
        if style["name"] == "doy":
            doyStyle(row, comp, errors, plotSynthDf, plotMeasuredDf, verbose)
        elif style["name"] == "dom":
            domStyle(row, comp, errors, plotSynthDf, plotMeasuredDf, verbose)
        else:
            diStyle(row, comp, errors, plotSynthDf, plotMeasuredDf, verbose)

    avgError = np.mean(np.abs(errors))

    print(f"Average error: {avgError:0.4f}\n")

    fig = plt.figure(figsize=[5.0, 5.0])
    gs = fig.add_gridspec(1)
    ax = fig.add_subplot(gs[0])
    plotSynthDf.plot(ax=ax, x=style["plotcolumns"][0], y=style["plotcolumns"][1])
    plotMeasuredDf.plot(ax=ax, x=style["plotcolumns"][0], y=style["plotcolumns"][1])
    ax.legend(labels=["Synth", "Measured"])

    plt.savefig(plotname)
    plt.close()

    return avgError


def loop(fraction, style, seed=1234, epochs=10, samplesize=10, verbose=False):
    print(f"--------\nStarting loop, style: {style['name']}")
    start = time.time()

    data = readData(DATA_FILE, fraction, seed)

    ctgan = teachGAN(data, disdim=style["disdim"], gendim=style["gendim"], epochs=epochs, gan_field_names=style["gan_fieldnames"], verbose=verbose)

    avgError = eval(data, ctgan, samplesize, style, style["plotname"].format(fraction=fraction), verbose=verbose)

    print(f"Loop took (H:MM:SS) {str(timedelta(seconds=(time.time() - start)))}\n\n")

    return avgError

styles = [ 
    {
        "gan_fieldnames": ["DayOfYear", "SecondsMidnight", "Value"],
        "sort": ["DayOfYear", "SecondsMidnight"],
        "astypes": {"DayOfYear":"int", "SecondsMidnight":"int"},
        "origin": ["DayOfYear", "SecondsMidnight", "Value"],
        "originastypes": {"DayOfYear":"int", "SecondsMidnight":"int"},
        "plotcolumns": ["Date", "Value"],
        "name": "doy",
        "plotname": "img/plot-doy-{fraction}.png"
    },
    {
        "gan_fieldnames": ["Year", "Month", "Day", "SecondsMidnight", "Value"],
        "sort": ["Year", "Month", "Day", "SecondsMidnight"],
        "astypes": {"Year":"int", "Month":"int", "Day":"int", "SecondsMidnight":"int"},
        "origin": ["Year", "Month", "Day", "SecondsMidnight", "Value"],
        "originastypes": {"Year":"int", "Month":"int", "Day":"int", "SecondsMidnight":"int"},
        "plotcolumns": ["Date", "Value"],
        "name": "dom",
        "plotname": "img/plot-dom-{fraction}.png"
    },
    {
        "gan_fieldnames": ["DateIndex", "SecondsMidnight", "Value"],
        "sort": ["DateIndex", "SecondsMidnight"],
        "astypes": {"DateIndex":"int", "SecondsMidnight":"int"},
        "origin": ["DateIndex", "SecondsMidnight", "Value"],
        "originastypes": {"DateIndex":"int", "SecondsMidnight":"int"},
        "plotcolumns": ["DateIndex", "Value"],
        "name": "di",
        "plotname": "img/plot-di-{fraction}.png"
    },
]

# Change the filepath if directory structure is changed after unzipping files...
DATA_FILE = "./data/Esimuotoiltu_data_csv_versio.csv"

TEST = False
DEBUG = False

if not TEST:
    #fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
    fractions = [0.01, 0.05, 0.10, 0.25, 1.00]
    errors = pd.DataFrame(columns=["Fraction", "Style", "Error"])

    start = time.time()

    for f in range(len(fractions)):
        for s in range(len(styles)):
            # default layer config
            if not 'disdim' in styles[s].keys():
                styles[s]["disdim"] =  (32, 16, 16, 4)
                styles[s]["gendim"] =  (32, 12, 4)
                

            avgError = loop(fractions[f], epochs=100, samplesize=50, style=styles[s], verbose=True)
            errors.loc[len(errors)] = [fractions[f], styles[s]['name'], avgError]

    errors.sort_values(by=["Style", "Fraction"], inplace=True)
    print(errors)

    print(f"Process took (H:MM:SS) {str(timedelta(seconds=(time.time() - start)))}.\n\n")
else:
    ts = 0
    #styles[ts]["disdim"] =  (64, 32, 16, 4)
    #styles[ts]["gendim"] =  (64, 32, 16, 8, 4)
    if not 'disdim' in styles[ts].keys():
        styles[ts]["disdim"] =  (32, 16, 16, 4)
        styles[ts]["gendim"] =  (32, 12, 4)
    loop(0.01, epochs=100, style=styles[ts], verbose=True)
