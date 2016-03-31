#Second program on gradient descent
#Uses linear regression in multiple variables
#Rohit Kr. Bose
#Feature scaling implemented


import math


#function to input values from user:

def input_training(x,y,m,n):
    print ("\nEnter details of %d training cases:" %m)
    for i in range (0,m):
        print ("\nTraining case #%d:" %(i+1)) #particular training case
        x[i].append(1.0)
        for j in range (1,n+1):
            inp=float(input("F#%d value = " %j)) #feature value input
            x[i].append(inp)
        inp=float(input("Output = "))
        y.append(inp)


#function to calculate dot product of two vectors:

def dot_product (v1,v2,size):
    prod=0.0
    for i in range (0,size):
        prod+=v1[i]*v2[i]
    return prod


#function to calculate derivative:

def derivative(x,y,m,n,C,j):
    deriv=0.0
    for i in range(0,m):
        deriv+=(dot_product(C,x[i],n+1)-y[i])*x[i][j]
    deriv/=m #from formula
    return deriv


#function to calculate error:

def error(x,y,m,n,C):
    J=0.0
    for i in range (0,m):
        J+=(dot_product(x[i],C,n+1)-y[i])**2/(2*m)
    return J


#function to implement feature scaling:

def feat_scale (x,m,n,avg,dev):
    #x0 is left as it is
    for i in range (1,n+1):
        large=x[0][i]
        small=x[0][i]
        s=0.0
        for j in range (0,m):
            s+=x[j][i] #calculating sum of nth features over m training sets
            if (x[j][i]>large): #largest
                large=x[j][i]
            if (x[j][i]<small): #smallest
                small=x[j][i]
        s=s/m
        avg.append(s) #mean
        dev.append(large-small) #deviation
        for j in range (0,m):
            if ((large-small)!=0):
                x[j][i]=(x[j][i]-s)/(large-small) #each feature now lies between -0.5 and +0.5

    
#gradient descent function:

def grad_desc(x,y,m,n,C,K):
    #K is the learning rate alpha
    #C is a n+1 dimensional vector (corresponding to theta), x is an m dimenstional array of n+1 dimensional vectors, y is an m dimensional array
    #C=[C0,C1,C2,...,Cn]
    #x=[{x0,x1,x2,...,xn}1,{x0,x1,x2,...,xn}2,...,{x0,x1,x2,...,xn}m]
    #y=[y1,y2,y3,...,yn]
    last=0.0    #variable to store last error
    flag=0      #if error is considerably less, flag = 1
    flag_first=0 #is it the first iteration?
    while (flag!=1): #flag=1 implies that cost error is quite less
        temp=[]
        for j in range(0,n+1):
            temp.append(C[j]-K*derivative(x,y,m,n,C,j)) #temporarily stored for simultaneous update later
        for j in range(0,n+1):
            v=C[j]-temp[j]
            C[j]=temp[j] #simultaneous update of C, corresponding to theta
        J=error(x,y,m,n,C) #error
        if (flag_first==1):
            if (J>last): #current error must be less than previous error for grad desc to work correctly
                K/=10 #reducing learning rate
                continue
        else:
            flag_first=1
        if (J<0.001 or J==last):
            flag=1
        if (last!=J):
            last=J
    return C


#Run-time code:

m=int(input("Enter number of training cases = "))
n=int(input("Enter number of variables per training case = "))
x=[] #array of training sets
y=[] #array outputs of training sets
C=[] #coefficient vector
A=[] #feature values array, whose output is to be predicted
avg=[] #array of means of n features
dev=[] #deviation of n features
avg.append(0)
dev.append(0)
for i in range (0,m):
    x.append([])  #creating list of lists
input_training(x,y,m,n) #calling function to input data
for i in range (0,n+1):
    C.append(0) #some random coefficient vector
L_Rate=0.1 #reasonably small learning rate
feat_scale(x,m,n,avg,dev)
C=grad_desc(x,y,m,n,C,L_Rate) #coefficient vector is calculated properly
print ("\n*****************************************************************************\n")
print ("\nEnter feature values of something whose output you want to predict:\n")
A.append(1)
for i in range(1,n+1):
    inp=float(input("#F%d value = " %i)) #input features by user
    A.append(inp)
    if (dev[i]!=0):
        A[i]=(A[i]-avg[i])/dev[i] #manipulating A vector as per feature scaling algorithm (I missed this step the last time, so I got weird results!)
ans = dot_product(C,A,n+1) #calculating output based on calculated coefficient vector
print ("Estimated output: %f" %ans)
