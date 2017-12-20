import numpy as np
import cv2
import time
d=0
cap = cv2.VideoCapture(-1)
print cap.isOpened()
def relu(Z):

    A=np.log(1+np.exp(Z))
    return A

W1=np.load('/home/jose/Desktop/W1.npy')
W2=np.load('/home/jose/Desktop/W2.npy')
W3=np.load('/home/jose/Desktop/W3.npy')
b1=np.load('/home/jose/Desktop/b1.npy')
b2=np.load('/home/jose/Desktop/b2.npy')
b3=np.load('/home/jose/Desktop/b3.npy')
 
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #print d

    
    
    
    
    # Display the resulting frame
    

    cv2.imshow('frame',gray)

    if cv2.waitKey(1) & 0xFF != ord('q'):
    	arr=np.reshape(frame,frame.shape[0]*frame.shape[1]*3)
    	X_test=np.asmatrix([arr]).T
        X_test=X_test/255.0
        #time.sleep(0.2)

        Z1 = np.dot(W1,X_test)+b1                                              # Z1 = np.dot(W1, X) + b1
#print Z1
        A1 = relu(Z1)                                             # A1 = relu(Z1)
#print A1
        Z2 =  np.dot(W2,A1)+b2                                              # Z2 = np.dot(W2, a1) + b2
#A2 = np.relu(Z2)                                              # A2 = relu(Z2)
        Z3 = np.dot(W3,Z2)+b3 
        #cv2.imwrite(filename, gray)
        #d=d+1
        

        pred=np.argmax(Z3)		
        if pred==0:
			print "blink"			
                    
        elif pred==1:
   			print "Center"
   
            		
        elif pred==2:
    		print "Left"
    #cv2.imshow('note',image2)
    		#cv2.waitKey(0)
        elif pred==3:
    		print "Right"
    #cv2.imshow('note',image3)
    		#cv2.waitKey(0)
    #cv2.waitKey(0)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

