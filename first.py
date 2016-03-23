#First program on gradient descent
#Uses linear regression in one variable

def input_training(x,y,m):
    print "Enter floor area of %d houses" %m
    for i in range(0,m):
        n=float(raw_input("#%d: " %(i+1)))
        x.append(n)
    print "Enter price of corresponding %d houses" %m
    for i in range (0,m):
        n=float(raw_input("#%d: " %(i+1)))
        y.append(n)


def derivative(x,y,m,C):
    deriv=0.0
    for i in range(0,m):
        deriv+=(C*x[i]-y[i])*x[i] #calculating derivative term
    deriv/=m
    return deriv;


def grad_desc(x,y,m,C,K):
    while (derivative(x,y,m,C)!=0):
        D=derivative(x,y,m,C)
        C = C-K*D
        #repeat until convergence
    return C;
                 
m=int(raw_input("Enter number of training examples = "))
x=[]
y=[]
input_training(x,y,m)
slope = grad_desc(x,y,m,1,0.1)
l=float(raw_input("\nEnter size of house = "))
print "Estimated price will be:"
print slope*l
