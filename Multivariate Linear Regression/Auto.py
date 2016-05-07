#Multivariate Linear Regression
#Rohit Kr. Bose

import math
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

E=[] #error list
I=[] #iteration number list

#function to calculate derivative:

def derivative(x,y,m,n,C,j):
	deriv=0.0
	for i in range(0,m):
		deriv+=(np.dot(C,x[i])-y[i])*x[i][j]
	deriv/=m #from formula
	return deriv	

#function to calculate error:

def error(x,y,m,n,C):
	J=0.0
	for i in range (0,m):
		J+=(np.dot(C,x[i])-y[i])**2/(2*m) 
	return J

#function to implement feature scaling:

def feat_scale (x,m,n,dev,avg):
	dev=np.std(x,axis=0) #standard deviation of each column
	avg=np.mean(x,axis=0) #average of each column
	for i in range (1,n+1):
		x[:,i]=(x[:,i]-avg[i])/dev[i] #broadcasting: performing an operation to entire column
	return (dev,avg) #returning arrays

#gradient descent function:

def grad_desc(x,y,m,n,C,L_Rate):
	print "Calculating...Please wait\n"
	#K is the learning rate alpha
	#C=[C0,C1,C2,...,Cn] (coefficient vector)
	#x=[{x0,x1,x2,...,xn}1,{x0,x1,x2,...,xn}2,...,{x0,x1,x2,...,xn}m] (feature matrix)
	#y=[y1,y2,y3,...,yn] (output vector)
	counter=0   #no. of iterations (for graph)
	last=0.0    #variable to store last error
	flag=0      #if error is considerably less, flag = 1
	flag_first=1 #value is 1 for first iteration only
	J_prev=0
	while (flag!=1): #flag=1 implies that cost error is quite less
		if (flag_first==0):
			J_prev=float(error(x,y,m,n,C))
		for j in range(0,n+1):
			C[j]=(C[j]-L_Rate*derivative(x,y,m,n,C,j))
		J=float(error(x,y,m,n,C)) #error
		if (flag_first==0):
			if (J>J_prev): #current error should be less than previous error
				L_Rate/=10.00 #anomaly, so decrease learning rate
		if (abs(J_prev-J)<0.001): #automatic convergence
			flag=1
		flag_first=0 #first iteration completed
		E.append(J) #error list (for graph)
		I.append(counter) #counter-th iteration (for graph)
		counter+=1
	return C #coefficient vector is returned

#function to plot graph of error vs iteration:

def plot_graph ():
	plt.plot(I,E)
	plt.show()

#Run-time code:

train_data=pd.read_csv("train.data", header=None, sep=r"\s+") #training data
train_data.insert(0,'k',1.00) #corresponding to x0=1
n=len(train_data.columns)-2 #no. of features
m=len(train_data) #no. of training cases
x=(train_data.ix[:,train_data.columns!=n]).values #feature matrix (train)
y=(train_data.ix[:,train_data.columns==n]).values.flatten() #outputs (train)
C=np.zeros(n+1) #coefficient vector
avg=np.zeros(n+1) #array of means of n features
dev=np.zeros(n+1) #deviation of n features
L_Rate=0.01 #reasonably small learning rate
dev, avg = feat_scale(x,m,n,dev,avg) #feature scaling
startTime=time.time()
C=grad_desc(x,y,m,n,C,L_Rate) #coefficient vector is calculated
elapsedTime=time.time()-startTime #time taken
plot_graph() #show graph
test_data=pd.read_csv("test.data", header=None, sep=r"\s+") #testing data
test_data.insert(0,'k',1.00) #corresponding to x0=1
m_T=len(test_data) #no. of testing cases
x_T=(test_data.ix[:,test_data.columns!=n]).values #feature matrix (test)
y_T=(test_data.ix[:,test_data.columns==n]).values.flatten() #outputs (test)
out=np.zeros(m_T) #predicted outputs
for j in range (1,n+1):
	if (dev[j]!=0):
		x_T[:,j]=(x_T[:,j]-avg[j])/dev[j] #manipulating as per feature scaling
print ("Original  |  Predicted")
print ("--------------------")
for i in range (0,m_T):
	out[i]=np.dot(C,x_T[i])
	print ("%f | %f" %(y[i], out[i])) #original and predicted outputs side by side
print ("\nLearning rate = %f" %L_Rate)
print ("Time taken by gradient descent = %f" %elapsedTime)
