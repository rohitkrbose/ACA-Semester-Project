#Clustering
#Implementing K-means algorithm
#Rohit Kr Bose

import math
import random

#function for input

def input_data (x,m,n):
    for i in range (0,m):
        print ("Data point #%d:" %(i+1))
        x.append([])
        for j in range (0,n):
            inp=float(input("F #%d = " %(j+1)))
            x[i].append(inp)

#function to add two vectors:

def vect_add (v1,v2,n):
    s=[]
    for i in range (0,n):
        s.append(v1[i]+v2[i])
    return s

#function to divide a vector by a scalar:

def vect_div (v1,scalar,n):
    for i in range (0,n):
        v1[i]=v1[i]/(float(scalar))
    return v1

#function to calculate square distance:

def sq_dist (v1,v2,n):
    s=0.0
    for i in range (0,n):
        s+=(v1[i]-v2[i])**2
    return s

#function to compare two vectors:

def vect_compare (v1,v2,n):
    for i in range (0,n):
        if (v1[i]!=v2[i]):
            return 0
    return 1

#function to compare two matrices:

def mat_compare (mat1, mat2, row, col):
    for i in range (0,row):
        for j in range (0,col):
            if (mat1[i][j]!=mat2[i][j]):
                return 0
    return 1

#function to compare two matrices:

def mat_copy (mat_to, mat_from, row, col):
    for i in range (0,row):
        mat_to.append([])
        for j in range (0,col):
            mat_to[i].append(mat_from[i][j])

#function to copy vectors(v_to,v_from,n)

def vect_copy (v_to,v_from,n):
    for i in range (0,n):
        v_to[i]=v_from[i]

#function to implement K-means algorithm:

def K_Means (x,m,n,K,C,label,last_label,dist,v):
    #K: no. of clusters
    #C: Array of position vectors of clusters
    #label: label given to a set of data points
    new=0
    C_copy=[]   
    mat_copy (C_copy,C,K,n)
    for i in range (0,m):
        for j in range (0,K):
            dist[j]=sq_dist(x[i],C[j],n) #calculating square distance of a i-th vector from all clusters, one by one
        min_ind=0
        for j in range (1,K):
            if (dist[j]<dist[min_ind]):
                min_ind=j #storing index of cluster from which min distance is acquired
        label[i]=min_ind #labelling that vector with the index thus obtained
    print label
    #all data points are now labelled
    vect_copy(last_label,label,m)
    for i in range (0,m):
        if (label[i]!=-1):
            count=0
            for j in range (0,n):
                v[j]=0
            for j in range (i,m):
                if (label[j]==label[i]):
                    label[j]= -1
                    count+=1 #no. of same labels
                    v=vect_add(v,x[j],n) #adding
            v=vect_div(v,count,n) #computing average
            vect_copy(C[label[i]],v,n) #storing vector
    if (mat_compare(C_copy,C,K,n)==1):
        return 1
    else:
        return 0
    

x=[]
A=[]
C=[]
v=[]
last_label=[]
dist=[]
n=(int)(input("Enter number of features = "))
m=(int)(input("Enter number of data points = "))
K=(int)(input("Enter number of clusters = "))
label=[]
last_label=[]
dist=[]
for i in range (0,m):
    last_label.append(0)
    label.append(0) #random label
for i in range (0,K):
    dist.append(0)
for i in range (0,K):
    C.append([])
    for j in range (0,n):
        C[i].append(random.random())
for i in range (0,n):
    v.append(0)
input_data(x,m,n)
print ("\n")
p=0
while (p==0):
    p=K_Means(x,m,n,K,C,label,last_label,dist,v)
print
for i in range (0,m):
    print 'Data: ' , x[i] , ' Label: ', last_label[i]
