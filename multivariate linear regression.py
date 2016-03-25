#Second program on gradient descent
#Uses linear regression in multiple variables
#Rohit Kr. Bose
#Feature scaling not implemented

#function to input values from user

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
        deriv+=(dot_product(C,x[i],n+1)-y[i])*x[i][j]
    deriv/=m #from formula
    return deriv

#function to calculate error

def error(x,y,m,n,C):
    J=0.0
    for i in range (0,m):
        J+=(dot_product(x[i],C,n+1)-y[i])**2/(2*m)
    return J

#gradient descent function:

def grad_desc(x,y,m,n,C,K):
    #K is the learning rate alpha
    #C is a n+1 dimensional vector (corresponding to theta), x is an m dimenstional array of n+1 dimensional vectors, y is an m dimensional array
    #C=[C0,C1,C2,...,Cn]
    #x=[{x0,x1,x2,...,xn}1,{x0,x1,x2,...,xn}2,...,{x0,x1,x2,...,xn}m]
    #y=[y1,y2,y3,...,yn]
    flag=0
    while (flag!=1): #flag=1 implies that minima is reached
        temp=[]
        for j in range(0,n+1):
            temp.append(C[j]-K*derivative(x,y,m,n,C,j)) #temporarily stored for simultaneous update later
        for j in range(0,n+1):
            v=C[j]-temp[j]
            C[j]=temp[j] #simultaneous update of C, corresponding to theta
        J=error(x,y,m,n,C) #error
        if (J<0.001): #small error means convergence!
            flag=1
    return C #coefficient vector is returned

#Run-time code:

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
L_Rate=0.01 #reasonably small learning rate
C=grad_desc(x,y,m,n,C,L_Rate) #coefficient vector is calculated properly
print (C)
print ("\n*****************************************************************************\n")
print ("\nEnter feature values of something whose output you want to predict:\n")
A.append(1)
for i in range(1,n+1):
    inp=float(input("#F%d value = " %i)) #input features by user
    A.append(inp)
ans = dot_product(C,A,n+1) #calculating output based on calculated coefficient vector
print ("\nEstimated output: %f" %ans)
