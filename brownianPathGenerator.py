import numpy as np
import matplotlib.pyplot as plt
import math

def dW(dt : float) -> float:
    """Generates a random Brownian noise

    Random noise over the timestep dt following dW ~ N (0,1) * sqrt(dt)

    Args:   
        dt (float): Timestep

    Returns:
        float : Change in the Brownian path over timestep
    """
    return np.random.normal(loc=0.0, scale=np.sqrt(dt))

def makePath(intervalStart : float, intervalEnd : float, timeStep : float) -> tuple[np.ndarray,np.ndarray]:
    """Generates a Brownian trajectory over the time interval specified with time discretization of timeStep

    Args:
        intervalStart (float): Start of the time interval
        intervalEnd (float): End of the time interval
        timeStep (float): Time discretization level

    Returns:
        tuple[np.ndarray,np.ndarray]: Time steps and Brownian path
    """
    #Start and end of the time interval
    timeInterval = [intervalStart, intervalEnd]

    #Numpy array for all the time discretization steps
    timeDiscretization = np.arange(timeInterval[0], timeInterval[1] + timeStep, timeStep)

    #Numpy array that will eventually have the brownian path
    brownianPath = np.zeros(timeDiscretization.size)

    for i in range(1, brownianPath.size):
        brownianPath[i] = brownianPath[i-1] + dW(timeStep)

    return timeDiscretization, brownianPath

def makeCorrelatedPaths(intervalStart : float, intervalEnd : float, timeStep : float, correlation : float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates 2 Brownian paths, which are correlated by the correlation coefficient provided

    Args:
        intervalStart (float): Start of the time interval
        intervalEnd (float): End of the time interval
        timeStep (float): Time discretization level
        correlation (float): Can be between 0-1 inclusive, correlation between the 2 Brownian paths   

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Timesteps, Brownian path 1, Brownian Path 2
    """
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

