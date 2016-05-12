import random
import math
import numpy as np
import pandas as pd

def user_input (m,n,V):
	for i in range (0,m):
		print ("Data point #%d" %(i+1))
		V.append([])
		for j in range (0,n):
			inp=float(input("F #%d = " %(j+1)))
			V[i].append(inp)

#V: 2D mXn feature matrix
#c: stores index of closest centroid (vector of size m)
#K: no. of clusters
#cntr: co-ordinates of centroid (size of array = K)

def k_means (m,n,K,V,c,cntr,cntr_prev):
	for i in range (0,m):
		dist=np.linalg.norm(V[i]-cntr[0])
		c[i]=0
		for j in range (0,K):
			if (np.linalg.norm(V[i]-cntr[j])<dist):
				dist=np.linalg.norm(V[i]-cntr[j]) #norm
				c[i]=j #assigning index of least distance
	for j in range (0,K):
		temp=np.zeros(n)
		count=0 #no. of points assigned to j-th centroid
		for i in range (0,m):
			if (c[i]==j):
				temp+=V[i]
				count+=1 #incrementing
		if (count!=0):
			temp/=count #taking mean
			cntr[j]=np.copy(temp) #co-ordinate of new centroid
	if (np.array_equal(cntr,cntr_prev)==True):
		flag=1 #k-means done!
	else:
		cntr_prev=np.copy(cntr) #creating copy
		flag=0
	return (flag,cntr,cntr_prev,c) #returning tuple

#function to calculate error (cost function)
#error=sum of normed distances from each data-point to its assigned cluster centroid point

def cost (m,n,K,cntr,V):
	error=0.0
	for i in range (0,m):
		error+=np.linalg.norm(V[i]-cntr[(int)(c[i])])
	return error 

m=(int)(input("Enter number of points = "))
n=(int)(input("Enter number of features = "))
V=[]
user_input(m,n,V)
V=np.asarray(V)
np.random.shuffle(V) #the co-rodinates are shuffled
K=(int)(input("Enter number of clusters = "))
best=np.zeros(m)
error=0
for i in range (0,100):
	cntr=np.zeros((K,n))
	cntr=random.sample(V,K) #initializing each centroid randomly as a given point
	cntr_prev=np.zeros((K,n))
	c=np.zeros(m)
	p=0 #flag variable
	while (p==0): #till 1 is returned, i.e. K-means is complete
		(p,cntr,cntr_prev,c)=k_means(m,n,K,V,c,cntr,cntr_prev)
	if (i==0): #for first iteration
		error=cost(m,n,K,cntr,V)
		best=np.copy(c)
	if (cost(m,n,K,cntr,V)<error): #when least error is detected
		error=cost(m,n,K,cntr,V)
		best=np.copy(c)
best=best+1 #for show purposes
for i in range (0,m):
	print 'Data: ' , V[i] , ' Label: ', (int)(best[i])
