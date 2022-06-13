from ast import excepthandler
import numpy as np
import pandas as pd
import re
import math
from  scipy.optimize import minimize as opt
import matplotlib.pyplot as plt
patternTime='^(?:[0-9]+)[:;,.](?:[0-9]+[:;,.])?(?:[0-9]+)$'
def loadData(file:str):
    df=pd.read_csv(file,delimiter=',',encoding_errors='ignore')
    return df
def rightBeginEnd(row):
    if (re.search(patternTime,str(row['start time']))!=None) and (re.search(patternTime,str(row['end time']))!=None):
        return True
    else:
        return False
def getSecond(s:str):
    s=s.replace(' ','')
    tempList=re.split("[:;,.]",s)
    power=1
    result=0
    try:
        for i in reversed(tempList):
            result+=int(i)*power
            power*=60
    except:
        print(s)
    return result
def getDuration(row):
    return getSecond(row['end time'])-getSecond(row['start time'])
def makeDuration(df:pd.DataFrame):
    count=0
    i=0
    df=df[[(rightBeginEnd(df.iloc[i,:]))for i in range(df.shape[0])]]
    df['duration']=df.apply(getDuration,axis=1)
    return df
#def cleanSpace(value):

def oneHot(df:pd.DataFrame):
    for i in ["venue","container"]:
        temp=pd.get_dummies(df[i])
        df.drop(i,axis=1,inplace=True)
        df=pd.concat([df,temp],axis=1)
    return df
def func(row):
    return str(row).replace(' ','').lower()
def cleanData(df:pd.DataFrame):
    df=df[['start time','end time','number of in','venue','container','describe how to make it',"viewer feeling of youtuber's style "]]
    df=makeDuration(df)
    df['venue']=df['venue'].apply(func)
    df['container']=df['container'].apply(func)
    temp=df["viewer feeling of youtuber's style "]
    df.drop("viewer feeling of youtuber's style ",axis=1,inplace=True)
    df=oneHot(df)
    df.insert(loc=28,column="viewer feeling",value=temp)
    df['describe how to make it']=pd.to_numeric(df['describe how to make it'],errors='coerce')
    df['viewer feeling']=pd.to_numeric(df['viewer feeling'],errors='coerce')
    df=df.dropna()
    return df

df=loadData(r"D:\học python\AIL\Annotation_AllVideos_FPT_Ver1.csv")
df=cleanData(df)
print(df.shape)
print(df.isna().any())
npData=(df.iloc[:,2:29]).to_numpy()
np.random.shuffle(npData)
def modelDataGen(j):
    global npData
    temp=np.zeros((npData.shape[0]))
    for i in range(npData.shape[0]):
        if npData[i,-1]==j:
            temp[i]=1
        else:
            temp[i]=0
    return temp
npData1=np.array([modelDataGen(j) for j in range(6)])
print(npData1.shape)
# print(npData1[0:4])
# print('----------------')
X_train=npData[0:3000,0:-1]
Y_train=npData1[:,0:3000]
# print(X[0:4])
# print('----------------')
# print(Y[0:4])
X_test=npData[3000:,0:-1]
Y_test=npData1[:,3000:]
print(X_train.shape)
"----------------------------------------------------------------"

# def g(z):
#     sigm = 1.0/(1.0+np.exp(-z))
#     return sigm
# print(in_theta.shape)
# def costFunction(theta, x, y):   
#     m = len(y)
#     h_theta = g(x.dot(theta))
#     J = (1.0/m)* (((-y).transpose()).dot(np.log(h_theta)) - (1.0 -y.transpose()).dot(np.log(1.0-h_theta)))
#     grad = grad = (1.0/m)* x.transpose().dot(h_theta - y)    
#     #return J, grad
#     print ('Cost at theta:', str(J[0,0]))
#     print ('Gradient at theta:','\n', str(grad[0,0]),'\n', str(grad[1,0]),'\n', str(grad[2,0]))
# def CostFunction(theta, x, y):
#     m = len(y)
#     h_theta = g(x.dot(theta))
#     J = (1.0/m)* (((-y).transpose()).dot(np.log(h_theta)) - (1.0 -y.transpose()).dot(np.log(1.0-h_theta)))
#     J = np.float64(J)
#     return J
# def Gradient(theta, x, y):
#     m = len(y)
#     n = x.shape[1]
#     theta = theta.reshape((n,1))
#     h_theta = g(x.dot(theta))
#     print(x.shape)
#     print(theta.shape)
#     print((h_theta-y).shape)
#     grad = (1.0/m)* (x.transpose().dot(h_theta - y)) 
#     print(grad.flatten().shape)
#     return grad.flatten()   
# Result = opt(fun = CostFunction, x0 = in_theta, args = (X, Y), method = 'TNC', jac = Gradient, options ={'maxiter':400})
# theta = Result.x
# print('Cost at theta:',Result.fun, '\n', 'Theta:', Result.x)
# def predict(theta, x):    
#     m = X.shape[0]
#     p = np.zeros((m,1))
#     n = X.shape[1]
#     theta = theta.reshape((n,1))
#     h_theta = g(X.dot(theta))    
#     for i in range(0, h_theta.shape[0]):
#         if h_theta[i] > 0.5:
#             p[i, 0] = 1
#         else:
#             p[i, 0] = 0
#     return p
# p = predict(theta, X_test)
# print ('Test Accuracy:', (Y_test[p == Y_test].size / float(Y_test.size)) * 100.0)


"----------------------------------------------------------"
class LogitRegression() :
    def __init__( self, learning_rate, iterations ) :        
        self.learning_rate = learning_rate        
        self.iterations = iterations
          
    # Function for model training    
    def fit( self, X, Y ) :        
        # no_of_training_examples, no_of_features        
        self.m, self.n = X.shape        
        # weight initialization        
        self.W = np.zeros( self.n )        
        self.b = 0        
        self.X = X        
        self.Y = Y
          
        # gradient descent learning
                  
        for i in range( self.iterations ) :            
            self.update_weights()            
        return self
      
    # Helper function to update weights in gradient descent
      
    def update_weights( self ) :           
        A = 1 / ( 1 + np.exp( - ( self.X.dot( self.W ) + self.b ) ) )
          
        # calculate gradients        
        tmp = ( A - self.Y.T )        
        tmp = np.reshape( tmp, self.m )        
        dW = np.dot( self.X.T, tmp ) / self.m         
        db = np.sum( tmp ) / self.m 
          
        # update weights    
        self.W = self.W - self.learning_rate * dW    
        self.b = self.b - self.learning_rate * db
          
        return self
      
    # Hypothetical function  h( x ) 
      
    def predict( self, X ) :    
        Z = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) )        
        Y = np.where( Z > 0.5, 1, 0 )        
        return Y
  
  
# Driver code


# Model training
model=[1,1,1,1,1,1,1]
for i in range(6):    
    model[i] = LogitRegression( learning_rate = 0.01, iterations = 1000 )
    print(Y_train[i])
    model[i].fit( X_train, Y_train[i] )    
    # Prediction on test set
    Y_pred = model[i].predict( X_test)    
    # measure performance    
    correctly_classified = 0    
    # counter    
    count = 0    
    for count in range( np.size( Y_pred ) ) :  
        if Y_test[i,count] == Y_pred[count] :            
            correctly_classified = correctly_classified + 1
    print( "Accuracy on test set by our model       :  ", i , ( 
        correctly_classified / count ) * 100 )
correctly=0
for i in range(X_test.shape[0]):
    max=-10
    maxPred=0
    for j in range(6):
        if (model[j].predict(X_test[i]))>max:
            max=(model[j].predict(X_test[i]))
            maxPred=j
    if (maxPred==npData[i+3000,-1]):
        correctly+=1
print("Overall accuracy :",correctly/X_test.shape[0]*100)
    