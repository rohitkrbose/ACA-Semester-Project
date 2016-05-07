#Second program on gradient descent
#Uses linear regression in multiple variables
#Rohit Kr. Bose
#Feature scaling implemented

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
	dev=np.std(x,axis=0)
	avg=np.mean(x,axis=0)
	for i in range (1,n+1):
		x[:,i]=(x[:,i]-avg[i])/dev[i]
	return (dev,avg)

#gradient descent function: (m x n matrix)

def grad_desc(x,y,m,n,C,L_Rate):
	print "Calculating...Please wait"
	#K is the learning rate alpha
	#C=[C0,C1,C2,...,Cn]
	#x=[{x0,x1,x2,...,xn}1,{x0,x1,x2,...,xn}2,...,{x0,x1,x2,...,xn}m]
	#y=[y1,y2,y3,...,yn]
	counter=0
	last=0.0    #variable to store last error
	flag=0      #if error is considerably less, flag = 1
	flag_first=1 #is it the first iteration?
	J_prev=0
	while (flag!=1): #flag=1 implies that cost error is quite less
		if (flag_first==0):
			J_prev=float(error(x,y,m,n,C))
		for j in range(0,n+1):
			C[j]=(C[j]-L_Rate*derivative(x,y,m,n,C,j)) #temporarily stored for simultaneous update later
		J=float(error(x,y,m,n,C)) #error
		if (flag_first==0):
			if (J>J_prev):
				L_Rate/=10.00	
		if (abs(J_prev-J)<0.001):
			flag=1
		flag_first=0
		E.append(J)
		I.append(counter)
		counter+=1
	return C

#function to plot graph of error vs iteration:

def plot_graph ():
	plt.plot(I,E)
	plt.show()

#Run-time code:

data=pd.read_csv("housing.data", header=None, sep=r"\s+")
data.insert(0,'k',1.00)
n=len(data.columns)-2 #no. of features
m=len(data) #no. of training cases
x=(data.ix[:,data.columns!=n]).values
y=(data.ix[:,data.columns==n]).values.flatten()
C=np.zeros(n+1) #coefficient vector
avg=np.zeros(n+1) #array of means of n features
dev=np.zeros(n+1) #deviation of n features
A=[] 
L_Rate=0.01 #reasonably small learning rate
dev, avg = feat_scale(x,m,n,dev,avg) #feature scaling
startTime=time.time()
C=grad_desc(x,y,m,n,C,L_Rate) #coefficient vector is calculated
elapsedTime=time.time()-startTime #time taken
plot_graph() #plotting graph
print ("\nEnter feature values of something whose output you want to predict:\n")
A.append(1)
for i in range(1,n+1):
    inp=float(input("#F%d value = " %i)) #input features by user
    A.append(inp)
    if (dev[i]!=0):
        A[i]=(A[i]-avg[i])/dev[i] #manipulating A vector as per feature scaling
A=np.asarray(A)
ans = np.dot(C,A) #calculating output based on calculated coefficient vector
print ("\nEstimated output: %f" %ans)
print ("Time taken by gradient descent = %f" %elapsedTime)
