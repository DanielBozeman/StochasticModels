import numpy as np
import matplotlib.pyplot as plt
import math

def dW(dt : float):
    return np.random.normal(loc=0.0, scale=np.sqrt(dt))

def makePath(intervalStart : float, intervalEnd : float, timeStep : float):
    #Start and end of the time interval
    timeInterval = [intervalStart, intervalEnd]

    #Numpy array for all the time discretization steps
    timeDiscretization = np.arange(timeInterval[0], timeInterval[1] + timeStep, timeStep)

    #Numpy array that will eventually have the brownian path
    brownianPath = np.zeros(timeDiscretization.size)

    for i in range(1, brownianPath.size):
        brownianPath[i] = brownianPath[i-1] + dW(timeStep)

    return timeDiscretization, brownianPath

def makeCorrelatedPaths(intervalStart : float, intervalEnd : float, timeStep : float, correlation : float):

    timeInterval, brownianPath1 = makePath(intervalStart, intervalEnd, timeStep)
    timeInterval, brownianPath2 = makePath(intervalStart, intervalEnd, timeStep)

    correlatedPath = np.zeros(brownianPath1.size)

    for i in range(correlatedPath.size):
        correlatedPath[i] = correlation * brownianPath1[i] + math.sqrt(1 - correlation **2 ) * brownianPath2[i]

    return timeInterval, brownianPath1, correlatedPath


if __name__ == "__main__":

    timeStep = pow(2, -10)

    timeInterval, brownianPath, brownianPath2 = makeCorrelatedPaths(0, 10, timeStep, -1)

    print(type(timeInterval))

    plt.plot(timeInterval, brownianPath)
    plt.plot(timeInterval, brownianPath2)

    plt.show()

