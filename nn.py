from tempfile import TemporaryFile
import numpy as np
import cv2
def relu(Z):

    A=np.log(1+np.exp(Z))
    return A

outfile = TemporaryFile()

x = [1,2,3,4]
#np.save('/home/jose/Desktop/nu.npy', x)
image=cv2.imread('file_64.jpg')
arr=np.reshape(image,image.shape[0]*image.shape[1]*3)

W1=np.load('/home/jose/Desktop/W1.npy')
W2=np.load('/home/jose/Desktop/W2.npy')
W3=np.load('/home/jose/Desktop/W3.npy')
b1=np.load('/home/jose/Desktop/b1.npy')
b2=np.load('/home/jose/Desktop/b2.npy')
b3=np.load('/home/jose/Desktop/b3.npy')

X_test=np.asmatrix([arr]).T
X_test=X_test/255.0

Z1 = np.dot(W1,X_test)+b1                                              # Z1 = np.dot(W1, X) + b1
#print Z1
A1 = relu(Z1)                                             # A1 = relu(Z1)
#print A1
Z2 =  np.dot(W2,A1)+b2                                              # Z2 = np.dot(W2, a1) + b2
#A2 = np.relu(Z2)                                              # A2 = relu(Z2)
Z3 = np.dot(W3,Z2)+b3                                      # Z3 = np.dot(W3,Z2) + b3
### END CODE HERE ###

pred=np.argmax(Z3)
if pred==0:
    print "Blink"
    #cv2.imshow('note',image)
    cv2.waitKey(0)
elif pred==1:
    print "Centre"
    #cv2.imshow('note',image1)
    cv2.waitKey(0)
elif pred==2:
    print "Left"
    #cv2.imshow('note',image2)
    cv2.waitKey(0)
elif pred==3:
    print "Right"
    #cv2.imshow('note',image3)
    cv2.waitKey(0)

cv2.waitKey(0)