import numpy as np
import matplotlib.pyplot as plt
import math
import brownianPathGenerator
import stochasticMethods
from stochasticMethods import SDEModel

#I envision that here every model will have a function that makes the SDEModels and then 
#a second function that actually runs that.
#Then in the main/another function you can just choose your model variables
#and run the model as many times as you want.

def OUProcess():

    theta = 0.7
    mu = 1.5
    sigma = 0.06

    timeStep = pow(2, -10)

    timeInterval = [0,7]

    numSims = 10

    constants = [theta, mu, sigma]

    def alphaFunction(model, value, time):
        return (model.constantsList[0] * (model.constantsList[1] - value))
    
    def betaFunction(model, value, time):
        return model.constantsList[2]
    
    model = SDEModel(constants=constants)

    model.alphaFunction = alphaFunction
    model.betaFunction = betaFunction

    for i in range(numSims):
        times, path = brownianPathGenerator.makePath(timeInterval[0], timeInterval[1], timeStep)

        approximation = stochasticMethods.eulerMaruyama(model, 0, times, path)

        plt.plot(times, approximation)

    plt.show()

#Generates CEV model, note that the Milstein beta prime isn't implemented
def CEVModel(interestRate : float, volatility : float, gamma : float) -> np.ndarray:

    constants = [interestRate, volatility, gamma]

    def alpha(model : SDEModel, value : float, time : float) -> float:
        return(model.constantsList[0] * value)
    
    def beta(model : SDEModel, value : float, time : float) -> float:
        return(model.constantsList[1] * pow(value,gamma))

    sdeModel = SDEModel(constants=constants, alphaFunction=alpha, betaFunction=beta)

    return sdeModel

#Generates Heston model, note that the Milstein beta prime isn't implemented
def hestonModel(interestRate : float, longVariance : float, reversionRate : float, volOfVol : float) -> tuple[stochasticMethods.SDEModel, stochasticMethods.SDEModel]:
    """Generates SDEModel objects for the Heston model, https://en.wikipedia.org/wiki/Heston_model

    Args:
        interestRate (float): Long term market interest rate
        longVariance (float): Long term variance
        reversionRate (float): Reversion rate of the variance
        volOfVol (float): Volatility of the volatility

    Returns:
        tuple[stochasticMethods.SDEModel, stochasticMethods.SDEModel]: Two SDEModel object, first for the overall 
        stock value, then for the volatility. From here generate a solution for the stochastic volatility and 
        couple that with the stock value model to get a final approximation of stock value.
    """
    
    #This section defines the variance model
    #################################################################################
    varianceConstants = [longVariance, reversionRate, volOfVol]

    def varianceAlpha(model : SDEModel, value : float, time : float):
        return (model.constantsList[1] * (model.constantsList[0] - value))

    def varianceBeta(model : SDEModel, value : float, time : float):
        return (model.constantsList[2] * math.sqrt(value))
    
    varianceModel = SDEModel(constants=varianceConstants, alphaFunction=varianceAlpha, betaFunction=varianceBeta)
    #################################################################################

    #This sections defines the stock model
    #################################################################################
    stockConstants = [interestRate]

    def stockAlpha(model : SDEModel, value : float, time : float):
        return(model.constantsList[0] * value)
    
    def stockBeta(model : SDEModel, value : float, time : float, volatility : float):
        return(math.sqrt(volatility) * value)

    stockModel = SDEModel(constants=stockConstants, alphaFunction=stockAlpha, betaFunction=stockBeta)
    #################################################################################

    return stockModel, varianceModel

#Generates SABR model, note that Milstein beta prime isn't implemented
def SABRModel(alpha : float, beta : float) -> tuple[np.ndarray, np.ndarray]:

    #This section defines the variance model
    #################################################################################
    varianceConstants = [alpha]

    def varianceAlpha(model : SDEModel, value : float, time : float):
        return (0)

    def varianceBeta(model : SDEModel, value : float, time : float):
        return (model.constantsList[0] * value)
    
    varianceModel = SDEModel(constants=varianceConstants, alphaFunction=varianceAlpha, betaFunction=varianceBeta)
    #################################################################################

    #This sections defines the stock model
    #################################################################################
    stockConstants = [beta]

    def stockAlpha(model : SDEModel, value : float, time : float):
        return(0)
    
    def stockBeta(model : SDEModel, value : float, time : float, volatility : float):
        return(volatility * pow(value, beta))

    stockModel = SDEModel(constants=stockConstants, alphaFunction=stockAlpha, betaFunction=stockBeta)
    #################################################################################

    return stockModel, varianceModel

#Runs the heston simulation
def runHeston():

    #Simulation Parameters
    numSims = 1000
    interval = [0,1]
    timeDiscretization = pow(2,-10)
    brownianCorrelation = 1

    #Stock model parameters
    interestRate = 0.1
    initialValue = 100

    #Variance model parameters
    reversionRate = 3.0
    longVariance = 0.2
    volOfVol = 0.1
    initialVariance = 0.1

    #Plot parameters
    numTransparent = numSims//4

    #Making model and generating approximations
    stockModel, varianceModel = hestonModel(interestRate, longVariance, reversionRate, volOfVol)
    times, values = stochasticMethods.runEMStochasticVol(stockModel, varianceModel, initialValue, initialVariance, timeDiscretization, numSims, interval[0], interval[1], brownianCorrelation)

    #Finding average path
    averageValues = values.sum(axis=0)
    averageValues = averageValues/numSims

    #Plotting paths
    for i in range(numTransparent):
        plt.plot(times, values[i], color = "gray", alpha=0.25)
    plt.plot(times, averageValues)
    plt.show()


    return()
    
if __name__ == "__main__":
    runHeston()
    

    