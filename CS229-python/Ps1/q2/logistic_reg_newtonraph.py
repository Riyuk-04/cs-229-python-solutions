import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

tau = 0.1
regulari_param = 0.0001
resolution = 30

def plot(X_train,Y_train,resolution):
	for i in range(resolution):
		for j in range(resolution):
			x = []
			x.append(2*i/(resolution+1) - 1)
			x.append(2*j/(resolution+1) - 1)
			x.append(1)
			if(lwlr(X_train,Y_train,x) == 1):
				plt.plot(x[0],x[1],'r.')
			else:
				plt.plot(x[0],x[1],'b.')
	plt.figure(1)	  

	plt.figure(2)
	for i in range(X_train.shape[0]):
		if Y_train[i] == 1:
			plt.plot(X_train[i][0],X_train[i][1],'rv')
		else:
			plt.plot(X_train[i][0],X_train[i][1],'bv')

	plt.show()


def weight(x,x_i):
	a = np.subtract(x,x_i)
	mag = np.dot(a,a)
	value = mag/(2*tau*tau)
	return np.exp(-value)		

def hypo(theta,x_i):
	a = np.dot(theta,x_i)
	sig_a = 1.0/(1+np.exp(-a))
	return sig_a

def lwlr(X_train,Y_train,x):
	theta = np.zeros(X_train.shape[1])
	
	Del_l = np.ones(X_train.shape[1])

	while(np.dot(Del_l,Del_l)>0.0001):
		weights = []
		for i in range(X_train.shape[0]):
			weights.append(weight(x,X_train[i]))
		
		hypot = []
		for j in range(X_train.shape[0]):
			hypot.append(hypo(theta,X_train[j]))
		

		Del_l = np.dot(np.transpose(X_train),np.multiply(weights,np.subtract(Y_train,hypot)))
		Diag_D = -1*np.multiply(weights,np.subtract(hypot,np.multiply(hypot,hypot)))
		D = np.diag(Diag_D)
		Lambda_I = np.diag(regulari_param*np.ones(X_train.shape[1]))
		Hessian = np.subtract(np.dot(np.transpose(X_train),np.dot(D,X_train)),Lambda_I)

		theta = theta - np.dot(np.linalg.inv(Hessian),Del_l)

	result = 0
	if(hypo(theta,x)>0.5):
		result = 1
	
	#print(result)
	return result


X_train = np.loadtxt('x.dat')
Y_train = np.loadtxt('y.dat')
X_train_bias = np.ones(X_train.shape[0])
X_train_bias = np.reshape(X_train_bias,(X_train.shape[0],1))
X_train = np.append(X_train, X_train_bias,axis = 1)
plot(X_train,Y_train,resolution)

