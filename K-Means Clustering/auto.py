import random
import math
import numpy as np
import pandas as pd

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

#function to display grouping:

def display_f(X,ans,m,best,K):
	print ("Original grouping:\n") #original grouping
	for u in range (1,K+1):
		print "GROUP ", u
		print ""
		for i in range (0,m):
			if (ans[i]==u):
				print X[i][0]
		print ""
	print("XxXxXxXxXxXxXxXxXxXxXxXxXxX")
	print ("\nNew grouping:\n") #new grouping
	for u in range (1,K+1):
		print "GROUP ", u
		print ""
		for i in range (0,m):
			if (best[i]==u):
				print X[i][0]
		print ""

data=pd.read_csv("crime_data.csv", header=None, sep=",")
X=(data.ix[:,data.columns!=1]).values
ans=(data.ix[:,data.columns==1]).values.flatten() 
#dataset is such that the second column contains group no., which is not needed now
#np.random.shuffle(X) (could be done, for even better output)
m=len(data) #no. of data points
n=len(data.columns)-2 #no. of features
V=np.delete(X, 0, 1) #deleting first column, as it contains text which is unimportant
K=(int)(input("Enter number of clusters = ")) #K=4 for our dataset
print "\n\n"
best=np.zeros(m) #this vector stores the optimum label assigned to each data-point
error=0.0
#random initialization of centroids a 100 times, to get the least error
for i in range (0,100):
	cntr=np.zeros((K,n))
	cntr=random.sample(V,K) #initializing each centroid randomly as a given point
	cntr_prev=np.zeros((K,n))
	c=np.zeros(m)
	p=0 #flag variable
	while (p==0): #till 1 is returned, i.e. K-means is complete, loop continues
		(p,cntr,cntr_prev,c)=k_means(m,n,K,V,c,cntr,cntr_prev)
	if (i==0): #for first iteration
		error=cost(m,n,K,cntr,V)
		best=np.copy(c)
	if (cost(m,n,K,cntr,V)<error): #when least error is detected
		error=cost(m,n,K,cntr,V)
		best=np.copy(c)
best=best+1 #for show purposes
display_f(X,ans,m,best,K) #displaying groups
