import numpy as np
import pandas

def cleanFile():

    df = pandas.read_csv("StockData\SPX_DaySplit.csv")

    closes = df["Close"]

    closes = closes.shift(1)

    df["Open"] = closes

    #print(df.head)

    df.to_csv("Test.csv", index=False)

    return

def daySplit():

    df = pandas.read_csv("StockData\SPX_original.csv")

    df[['Month', 'Day', 'Year']] = df['Date'].str.split('/', n=2, expand=True)

    columns = ["Year", "Month", "Day", "Open", "High", "Low", "Close", "Volume"]

    df = df[columns]

    #print(df.head)

    df.to_csv("SPX_DaySplit.csv", index=False)

    return


if __name__ == "__main__":
    cleanFile()
    #daySplit()