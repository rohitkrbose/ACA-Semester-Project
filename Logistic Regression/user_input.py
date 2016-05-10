import math
import numpy as np
import pandas as pd
     
#sigmoid function: g(z)=1/(1+exp(-z))

def sigmoid (C,x):
	return (1/(1+math.exp(-np.dot(C,x))))

#function to calculate derivative

def derivative(x,y,m,n,C,j):
	deriv=0.0
	for i in range(0,m):
		deriv+=(sigmoid(C,x[i])-y[i])*x[i][j]
	deriv/=m #from formula
	return deriv

#function to calculate error

def error(x,y,m,n,C):
	J=0.0 #calculating error as per definition
	for i in range (0,m):
		if (y[i]==1):
			J+=math.log(sigmoid(C,x[i]))
		else:
			J+=math.log(1-sigmoid(C,x[i]))
	J/=-m #note the negative sign
	return (J)

#feature scaling fucntion:

def feat_scale (x,m,n,dev,avg):
	dev=np.std(x,axis=0) #standard deviation of each column
	avg=np.mean(x,axis=0) #average of each column
	for i in range (1,n+1):
		x[:,i]=(x[:,i]-avg[i])/dev[i] #broadcasting: performing an operation to entire column
	return (dev,avg) #returning arrays

#gradient descent function

def grad_desc (x,y,m,n,C,L_Rate):
	flag=0
	while (flag!=1): #flag=1 implies that minima is reached
		J_prev=error(x,y,m,n,C)
		for j in range(0,n+1):
			C[j]=C[j]-L_Rate*derivative(x,y,m,n,C,j) #gradient descent
		J=error(x,y,m,n,C) #error
		if (abs(J-J_prev)<0.001): #small error means convergence!
			flag=1
	return C #coefficient vector is returned

#run-time code:

train_data=pd.read_csv("haberman.data",header=None,sep=",")
train_data.insert(0,'k',1.00)
n=len(train_data.columns)-2
m=len(train_data)
x=(train_data.ix[:,train_data.columns!=n]).values
y=(train_data.ix[:,train_data.columns==n]).values.flatten()
y=y-1
C=np.zeros(n+1) #coefficient vector
A=[] #feature values array, whose output is to be predicted
L_Rate=0.1 #reasonably small learning rate
dev=np.zeros(n+1)
avg=np.zeros(n+1)
dev,avg=feat_scale(x,m,n,dev,avg)
C=grad_desc(x,y,m,n,C,L_Rate) #coefficient vector is calculated properly
print ("\n*****************************************************************************\n")
print ("\nEnter feature values of something whose output you want to predict:\n")
A.append(1)
for i in range(1,n+1):
	inp=float(input("#F%d value = " %i)) #input features by user
	A.append(inp)
	if (dev[i]!=0):
		A[i]=(A[i]-avg[i])/dev[i]
A=np.asarray(A)
ans = np.dot(C,A) #calculating output based on calculated coefficient vector
if (ans>=0):
    print ("\nOutput = 2")
else:
    print ("\nOutput = 1")
