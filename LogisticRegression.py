
import numpy as np
import logging
import json
from utility import * #custom methods for data cleaning

FILE_NAME_TRAIN = 'train.csv' #replace this file name with the train file
FILE_NAME_TEST = 'test.csv' #replace
ALPHA = 2.0
EPOCHS = 40000#keep this greater than or equl to 5000 strictly otherwise you will get an error
MODEL_FILE = 'models/model'
train_flag = True


logging.basicConfig(filename='output.log',level=logging.DEBUG)

np.set_printoptions(suppress=True)
#################################################################################################
#####################################write the functions here####################################
#################################################################################################
#this function appends 1 to the start of the input X and returns the new array
def appendIntercept(X):
    #steps
    #make a column vector of ones
    #stack this column vector infront of the main X vector using hstack
    #return the new matrix
    #pass#remove this line once you finish writing
    size = X.shape[0]
    col = np.ones((size,1))
    col = np.hstack((col,X))
    return col


 #intitial guess of parameters (intialize all to zero)
 #this func takes the number of parameters that is to be fitted and returns a vector of zeros
def initialGuess(n_thetas):
    #pass
    theta = np.zeros(n_thetas)
    return theta



def train(theta, X, y, model):
    #J = [] #this array should contain the cost for every iteration so that you can visualize it later when you plot it vs the ith iteration
     #train for the number of epochs you have defined
    m = len(y)
     #your  gradient descent code goes here
     #steps
     #run you gd loop for EPOCHS that you have defined
        #calculate the predicted y using your current value of theta
        # calculate cost with that current theta using the costFunc function
        #append the above cost in J
        #calculate your gradients values using calcGradients function
        # update the theta using makeGradientUpdate function (don't make a new variable assign it back to theta that you received)
    for x in range(0,EPOCHS):
        y_pre = predict(X,theta)
        #costt = costFunc(m,y,y_pre)
        #J.append(costt)
        grads = calcGradients(X,y,y_pre,m)
        theta = makeGradientUpdate(theta,grads)
   # model['J'] = J
    model['theta'] = list(theta)
    return model


#this function will calculate the total cost and will return it
def costFunc(m,y,y_predicted):
    #takes three parameter as the input m(#training examples), (labeled y), (predicted y)
    #steps
    #apply the formula learnt
    #pass
    c = np.sum(np.square(y-y_predicted))
    return c/(m)



def calcGradients(X,y,y_predicted,m):
    #apply the formula , this function will return cost with respect to the gradients
    # basically an numpy array containing n_params
   #pass
   return (np.dot((y_predicted-y),X))/m

#this function will update the theta and return it
def makeGradientUpdate(theta, grads):
    #pass
    return theta - ALPHA*grads


#this function will take two paramets as the input
def predict(X,theta):
    #pass
    sigmoidVal = 1/(1+((np.exp(-np.dot(X,theta)))))
    return sigmoidVal




########################main function###########################################
def main():
    if(train_flag):
        model = {}
        X_df,y_df = loadData(FILE_NAME_TRAIN)
        X,y, model = normalizeData(X_df, y_df, model)
        X = appendIntercept(X)
        theta = initialGuess(X.shape[1])
        model = train(theta, X, y, model)
        X_test,y_test = loadData(FILE_NAME_TEST)
        X_test,y_test = normalizeTestData(X_test, y_test, model)
        X_test = appendIntercept(X_test)
        accuracy(X_test,y_test,model)
        with open(MODEL_FILE,'w') as f:
            f.write(json.dumps(model))

if __name__ == '__main__':
    main()
