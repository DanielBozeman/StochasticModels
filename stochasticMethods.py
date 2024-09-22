import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable
import math
import brownianPathGenerator

#The idea here is that the methods are going to take the required functions and paths as inputs,
#and they return the approximation path

class SDEModel():
    """SDEModel class that gives required function for EM and Milstein methods

    Attributes:
        constantsList (List) : List of constants that are used in the functions
    """

    constantsList = []

    def alphaFunction(self, model, value : float, time : float) -> float:
        """The function that is multiplied to dt in the SDE definition. 
        
        SHOULD BE OVERWRITTEN WHEN MODEL MADE

        Args:
            model (SDEModel): SDEModel object, for the constants list
            value (float): Current SDE value 
            time (float): Current time value

        Returns:
            float: Value of the alpha function at that time and value 
        """
        return (model.constantsList[0] * value)
    
    def betaFunction(self, model, value : float, time : float) -> float:
        """The function that is multiplied to dW in the SDE definition

        IF YOU ARE USING STOCHASTIC VOLATILITY ADD A VOLATILITY PARAMETER WHEN YOU OVERWRITE 

        Args:
            model (SDEModel): The SDE model, for the constant list
            value (float): Current SDE value
            time (float): Current time value

        Returns:
            float: Value of the beta function at that time and value
        """
        return (model.constantsList[1] * value)
    
    def betaPrimeFunction(self, model, value : float, time : float) -> float:
        """Derivative of the beta function, for use in the Milstein method

        SHOULD BE OVERWRITTEN WHEN SDEMODEL MADE
        IF USING STOCHASTIC VOLATILITY MAKE SURE TO ADD A VOLATILITY PARAMETER

        Args:   
            model (SDEModel): SDEModel object, for the constants list
            value (float): Current SDE value
            time (float): Current time value

        Returns:
            float: Value of the beta derivative at that time and value
        """
        return (model.constantsList[1])


    def __init__(self, 
                constants,
                alphaFunction : Callable[[float, float], float] = None,
                betaFunction : Callable[[float, float], float] = None,
                betaPrimeFunction : Callable[[float, float], float] = None):
        """Creates a SDEModel object, even if the alpha and beta functions aren't used here, make sure you change them

        Args:
            constants (List): List of constants used in the model
            alphaFunction (Callable[[float, float], float], optional): New definition for the alpha function. Defaults to None.
            betaFunction (Callable[[float, float], float], optional): New definition for the beta function. Defaults to None.
            betaPrimeFunction (Callable[[float, float], float], optional): New definition for the beta prime function. Defaults to None.
        """
        self.constantsList = constants
        
        if alphaFunction != None:
            self.alphaFunction = alphaFunction
        if betaFunction != None:
            self.betaFunction = betaFunction
        if betaPrimeFunction != None:
            self.betaPrimeFunction = betaPrimeFunction

#Performs the basic EM method using the functions defined in the model
def eulerMaruyama(model : SDEModel, 
           initialValue : float, 
           timeInterval : np.ndarray, 
           brownianPath : np.ndarray):
    """Performs the Euler-Maruyama method using the supplied model functions and path

    Args:
        model (SDEModel): Stochastic model, make sure that you have adjusted the functions
        initialValue (float): Initial value of the SDE
        timeInterval (np.ndarray): Time steps array, from the makePath method
        brownianPath (np.ndarray): Brownian trajectory to approximate with

    Returns:
        np.ndarray: Approximate solution of the SDE over the trajectory supplied
    """
    
    approximatePath = np.zeros(timeInterval.size)

    dt = timeInterval[1] - timeInterval[0]

    approximatePath[0] = initialValue

    for i in range(1,approximatePath.size):
        prevValue = approximatePath[i-1]
        prevTime = timeInterval[i-1]
        dW = brownianPath[i] - brownianPath[i-1]
        #print(alphaFunction(prevValue, prevTime))
        #print(betaFunction(prevValue, prevTime))
        approximatePath[i] = prevValue + model.alphaFunction(model, prevValue, prevTime) * dt +  model.betaFunction(model, prevValue, prevTime) * dW
        

    return approximatePath

#Performs the Milstein method with the functions defined in the model
def milstein(model : SDEModel,
      initialValue : float, 
      timeInterval : np.ndarray, 
      brownianPath : np.ndarray):
    """Performs the Milstein method with the provided model

    Args:
        model (SDEModel): Stochastic model, make sure you've adjusted the functions
        initialValue (float): Initial value of the SDE
        timeInterval (np.ndarray): Time steps array, from the makePath method
        brownianPath (np.ndarray): Brownian trajectory to approximate with

    Returns:
        np.ndarray : Approximate solution of the SDE over the trajectory supplied
    """
    
    approximatePath = np.zeros(timeInterval.size)

    dt = timeInterval[1] - timeInterval[0]

    approximatePath[0] = initialValue

    for i in range(1,approximatePath.size):
        prevValue = approximatePath[i-1]
        prevTime = timeInterval[i-1]
        dW = brownianPath[i] - brownianPath[i-1]
        approximatePath[i] = prevValue + model.alphaFunction(model, prevValue, prevTime) * dt + \
                             model.betaFunction(model, prevValue, prevTime) * dW + \
                             0.5 * model.betaFunction(model, prevValue, prevTime) * model.betaPrimeFunction(model, prevValue, prevTime) * ( dW**2 - dt)

    return approximatePath

#Performs the EM method like before but passes the current volatility to the beta function
def eulerMaruyamaStochasticVol(model : SDEModel, 
           initialValue : float, 
           timeInterval : np.ndarray, 
           brownianPath : np.ndarray,
           volatilityPath : np.ndarray) -> np.ndarray:
    """Performs the Euler-Maruyama, but with stochastic volatility

    Args:
        model (SDEModel): Stochastic model, make sure that the beta function takes a volatility parameter
        initialValue (float): Initial value of the SDE
        timeInterval (np.ndarray): Time step array from makePath method
        brownianPath (np.ndarray): Brownian trajectory to estimate with
        volatilityPath (np.ndarray): Stochastic volatility trajectory

    Returns:
        np.ndarray: Approximation of the SDE
    """
    approximatePath = np.zeros(timeInterval.size)

    dt = timeInterval[1] - timeInterval[0]

    approximatePath[0] = initialValue

    for i in range(1,approximatePath.size):
        prevValue = approximatePath[i-1]
        prevTime = timeInterval[i-1]
        prevVolatility = volatilityPath[i-1]
        dW = brownianPath[i] - brownianPath[i-1]
        approximatePath[i] = prevValue + model.alphaFunction(model, prevValue, prevTime) * dt +  model.betaFunction(model, prevValue, prevTime, prevVolatility) * dW
        

    return approximatePath

def runEM(model : SDEModel,
          initialValue : float,
          timeDiscretization : float,
          numSims : int,
          intervalStart : float,
          intervalEnd : float) -> tuple[np.ndarray, list]:
    
    approximations = []

    for i in range(numSims):
        times, brownianPath = brownianPathGenerator.makePath(intervalStart, intervalEnd, timeDiscretization)

        approximation = eulerMaruyama(model, initialValue, times, brownianPath)

        approximations.append(approximation)

    return times, approximations

def runEMStochasticVol(model : SDEModel,
                       initialValue : float,
                       timeDiscretization : float,
                       numSims : int,
                       intervalStart : float,
                       intervalEnd : float,
                       volatilityPaths : list) -> tuple[np.ndarray, list]:
    
    approximations = []

    for i in range(numSims):
        times, brownianPath = brownianPathGenerator.makePath(intervalStart, intervalEnd, timeDiscretization)

        approximation = eulerMaruyamaStochasticVol(model, initialValue, times, brownianPath, volatilityPaths[i])

        approximations.append(approximation)

    return times, approximations
    

if __name__ == "__main__":

    print("Running simulation")

    alpha = 2
    beta = 1

    constantsList = [alpha, beta]

    courseness = 10

    model = SDEModel(constants = constantsList)

    timeStep = pow(2, -10)

    times, path = brownianPathGenerator.makePath(0,1,timeStep)

    averagePath = np.zeros(path.size)
    averagePath2 = np.zeros(path.size)
    averagePath2 = averagePath2[::courseness]
    averagePath3 = np.zeros(path.size)
    averagePath3 = averagePath3[::courseness]

    numSims = 1

    for i in range(numSims):

        times, path = brownianPathGenerator.makePath(0,1,timeStep)

        path2 = path[::courseness]
        times2 = times[::courseness]

        approximate = stochasticExact(model, 1, times, path)
        approximate2 = milstein(model, 1, times2, path2)
        approximate3 = eulerMaruyama(model, 1, times2, path2)

        averagePath = np.add(averagePath, approximate)
        averagePath2 = np.add(averagePath2, approximate2)
        averagePath3 = np.add(averagePath3, approximate3)

    averagePath = np.divide(averagePath, numSims)
    averagePath2 = np.divide(averagePath2, numSims)
    averagePath3 = np.divide(averagePath3, numSims)

    
    plt.plot(times, averagePath, label = "exact")
    plt.plot(times2, averagePath2, label = "Milstein")
    plt.plot(times2, averagePath3, label = "EM")

    plt.legend(loc = "best")
    plt.show()


