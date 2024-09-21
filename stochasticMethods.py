import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable
import math
import brownianPathGenerator

#The idea here is that the methods are going to take the required functions and paths as inputs,
#and they return the approximation path

class SDEModel():

    constantsList = []

    def alphaFunction(self, model, value : float, time : float):
        return (model.constantsList[0] * value)
    
    def betaFunction(self, model, value : float, time : float):
        return (model.constantsList[1] * value)
    
    def betaPrimeFunction(self, model, value, time):
        return (model.constantsList[1])


    def __init__(self, 
                constants,
                alphaFunction : Callable[[float, float], float] = None,
                betaFunction : Callable[[float, float], float] = None):
        
        self.constantsList = constants
        
        if alphaFunction != None:
            self.alphaFunction = alphaFunction
        if betaFunction != None:
            self.betaFunction = betaFunction

#Performs the basic EM method using the functions defined in the model
def eulerMaruyama(model : SDEModel, 
           initialValue : float, 
           timeInterval : np.ndarray, 
           brownianPath : np.ndarray):
    
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
           volatilityPath : np.ndarray):
    
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

def stochasticExact(model : SDEModel,
             initialValue : float,
             timeInterval : np.ndarray,
             brownianPath : np.ndarray):
    
    approximatePath = np.zeros(timeInterval.size)

    dt = timeInterval[1] - timeInterval[0]

    approximatePath[0] = initialValue

    for i in range(1,approximatePath.size):
        prevValue = approximatePath[i-1]
        prevTime = timeInterval[i-1]
        dW = brownianPath[i] - brownianPath[i-1]
        approximatePath[i] = initialValue * math.exp((model.constantsList[0] - 0.5 * model.constantsList[1] **2 )* timeInterval[i] + model.constantsList[1] * brownianPath[i])

    return approximatePath
    

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


