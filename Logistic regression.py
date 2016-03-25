#Logistic regression that works only on y=0 and y=1 (binary case)
#Rohit Kr. Bose
#Feauture scaling not implemented

import math

#function to input values from user

def input_training(x,y,m,n):
    print ("\nEnter details of %d training cases:" %m)
    for i in range (0,m):
        print ("\nTraining case #%d:" %(i+1)) #particular training case
        x[i].append(1.0)
        for j in range (1,n+1):
            inp=float(input("F#%d value = " %j)) #feature value input
            x[i].append(inp)
        inp=float(input("Output (1 if true, 0 if false) = "))
        y.append(inp)
     
#sigmoid function: g(z)=1/(1+exp(-z))

def sigmoid (C,x,size):
    return (1/(1+math.exp(-dot_product(C,x,size))))

#function to calculate dot product of two vectors

def dot_product (v1,v2,size):
    prod=0.0
    for i in range (0,size):
        prod+=v1[i]*v2[i]
    return prod

#function to calculate derivative

def derivative(x,y,m,n,C,j):
    deriv=0.0
    for i in range(0,m):
        deriv+=(sigmoid(C,x[i],n+1)-y[i])*x[i][j]
    deriv/=m #from formula
    return deriv

#function to calculate error

def error(x,y,m,n,C):
    J=0.0 #calculating error as per defiinition
    for i in range (0,m):
        J+=y[i]*math.log(sigmoid(C,x[i],n+1))+(1-y[i])*math.log(1-sigmoid(C,x[i],n+1))
    J/=-m
    return J

#gradient descent function

def grad_desc (x,y,m,n,C,K):
    flag=0
    while (flag!=1): #flag=1 implies that minima is reached
        temp=[]
        for j in range(0,n+1):
            temp.append(C[j]-K*derivative(x,y,m,n,C,j)) #temporarily stored for simultaneous update later
        for j in range(0,n+1):
            C[j]=temp[j] #simultaneous update of C, corresponding to theta
        J=error(x,y,m,n,C) #error
        if (J<0.1): #small error means convergence!
            flag=1
        #print J
    return C #coefficient vector is returned


m=int(input("Enter number of training cases = "))
n=int(input("Enter number of variables per training case = "))
x=[] #array of training sets
y=[] #array outputs of training sets
C=[] #coefficient vector
A=[] #feature values array, whose output is to be predicted
for i in range (0,m):
    x.append([])  #creating list of lists
input_training(x,y,m,n) #calling function to input data
for i in range (0,n+1):
    C.append(0) #some random coefficient vector
L_Rate=0.1 #reasonably small learning rate
C=grad_desc(x,y,m,n,C,L_Rate) #coefficient vector is calculated properly
#print (C)
print ("\n*****************************************************************************\n")
print ("\nEnter feature values of something whose output you want to predict:\n")
A.append(1)
for i in range(1,n+1):
    inp=float(input("#F%d value = " %i)) #input features by user
    A.append(inp)
ans = dot_product(C,A,n+1) #calculating output based on calculated coefficient vector
if (ans>=0):
    print ("\nOutput = 1")
else:
    print ("\nOutput = 0")
