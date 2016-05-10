import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

E=[]
I=[]

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
	print "Calculating...Please wait\n"
	flag=0
	counter=0
	while (flag!=1): #flag=1 implies that minima is reached
		J_prev=error(x,y,m,n,C)
		for j in range(0,n+1):
			C[j]=C[j]-L_Rate*derivative(x,y,m,n,C,j) #gradient descent
		J=error(x,y,m,n,C) #error
		#print J
		if (abs(J-J_prev)<0.0001): #small error means convergence!
			flag=1
		E.append(J)
		I.append(counter)
		counter+=1
	return C #coefficient vector is returned

def plot_graph ():
	plt.plot(I,E)
	plt.show()

#run-time code:

train_data=pd.read_csv("haberman_train.data",header=None,sep=",")
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
plot_graph()
test_data=pd.read_csv("haberman_test.data",header=None,sep=",")
test_data.insert(0,'k',1.00)
m_T=len(test_data)
x_T=(test_data.ix[:,test_data.columns!=n]).values
y_T=(test_data.ix[:,test_data.columns==n]).values.flatten()
out=np.zeros(m_T)
for j in range (1,n+1):
	if (dev[j]!=0):
		x_T[:,j]=(x_T[:,j]-avg[j])/dev[j] #manipulating as per feature scaling	
print ("Original  |  Predicted")
print ("-------------------------")
acc=0
for i in range (0,m_T):
	ans=np.dot(C,x_T[i])
	if (ans<0):
		out[i]=1
	else:
		out[i]=2
	if (out[i]==y[i]):
		acc+=1
	print ("%d | %d" %(y_T[i], out[i]))
acc=acc/(float(len(y_T)))*100
print("Accuracy = %f", acc)
