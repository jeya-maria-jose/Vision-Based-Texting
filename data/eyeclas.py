import math
import numpy as np
import cv2
import tensorflow as tf
import matplotlib as plt
from tensorflow.python.framework import ops
np.random.seed(1)
# GRADED FUNCTION: one_hot_matrix

def relu(Z):

    A=np.log(1+np.exp(Z))
    return A
def convert_to_activated(Z):
    m,n=Z.shape
    for i in range(n):
        sums=0
        for j in range(m):
            sums+=Z[j][i]
        for k in range(m):
            Z[j][i]/=sums
    return Z


def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                     will be 1.
    Arguments:
    labels -- vector containing the labels
    C -- number of classes, the depth of the one hot dimension
    Returns:
    one_hot -- one hot matrix
    """

    ### START CODE HERE ###

    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C,name="C")

    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(labels,C,axis=0)

    # Create the session (approx. 1 line)
    sess = tf.Session()

    # Run the session (approx. 1 line)
    one_hot = sess.run(one_hot_matrix)

    # Close the session (approx. 1 line). See method 1 above.
    sess.close()

    ### END CODE HERE ###

    return one_hot

# GRADED FUNCTION: create_placeholders

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    ### START CODE HERE ### (approx. 2 lines)
    X =tf.placeholder(tf.float32,shape=[n_x,None],name = 'X')
    Y = tf.placeholder(tf.float32,shape=[n_y,None], name = 'Y')
    ### END CODE HERE ###

    return X, Y

# GRADED FUNCTION: initialize_parameters

def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    tf.set_random_seed(1)                   # so that your "random" numbers match ours

    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.get_variable("W1", [50,921600], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [50,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [20,50], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [20,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [4,20], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [4,1], initializer = tf.zeros_initializer())
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters

# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1,X),b1)                                              # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                               # A1 = relu(Z1)
    Z2 =  tf.add(tf.matmul(W2,A1),b2)                                              # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3,Z2),b3)                                      # Z3 = np.dot(W3,Z2) + b3
    ### END CODE HERE ###

    return Z3

# GRADED FUNCTION: compute_cost

def compute_cost(Z3, Y):
    """
    Computes the cost
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    Returns:
    cost - Tensor of the cost function
    """
    #z = tf.placeholder(tf.float32, name = "z")
    #y = tf.placeholder(tf.float32, name = "y")

    # Use the loss function (approx. 1 line)
    #cost = tf.nn.sigmoid_cross_entropy_with_logits(logits =z,  labels =y)

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)


    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits =logits, labels = labels))

    ### END CODE HERE ###

    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate = 1,
          num_epochs = 500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_x, n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters()

    ### END CODE HERE ###
    #print(X_train)
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z3 = forward_propagation(X, parameters)
    ### END CODE HERE ###

    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z3, Y)

    ### END CODE HERE ###

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    ### END CODE HERE ###

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            seed = seed + 1
            #print m

            _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
                ### END CODE HERE ###
            #print minibatch_cost
            #epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch,minibatch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(minibatch_cost)

        # plot the cost

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters



image=cv2.imread('b1.jpg')
image1=cv2.imread('b2.jpg')
image2=cv2.imread('b3.jpg')
image3=cv2.imread('b4.jpg')
image4=cv2.imread('b5.jpg')
image5=cv2.imread('b6.jpg')
image6=cv2.imread('b7.jpg')
image7=cv2.imread('b8.jpg')
image8=cv2.imread('c1.jpg')
image9=cv2.imread('c2.jpg')
image10=cv2.imread('c3.jpg')
image11=cv2.imread('c4.jpg')
image12=cv2.imread('c5.jpg')
image13=cv2.imread('c6.jpg')
image14=cv2.imread('c7.jpg')
image15=cv2.imread('c8.jpg')
image16=cv2.imread('c9.jpg')
image17=cv2.imread('c10.jpg')
image18=cv2.imread('l1.jpg')
image19=cv2.imread('l2.jpg')
image20=cv2.imread('l3.jpg')
image21=cv2.imread('l4.jpg')
image22=cv2.imread('l5.jpg')
image23=cv2.imread('l6.jpg')
image24=cv2.imread('l7.jpg')
image25=cv2.imread('l8.jpg')
image26=cv2.imread('l9.jpg')
image27=cv2.imread('r1.jpg')
image28=cv2.imread('r2.jpg')
image29=cv2.imread('r3.jpg')
image30=cv2.imread('r4.jpg')
image31=cv2.imread('r5.jpg')
image32=cv2.imread('r6.jpg')
image33=cv2.imread('r10.jpg')
image34=cv2.imread('r8.jpg')
image35=cv2.imread('r9.jpg')



arr=np.reshape(image,image1.shape[0]*image1.shape[1]*3)
arr1=np.reshape(image1,image1.shape[0]*image1.shape[1]*3)
arr2=np.reshape(image2,image2.shape[0]*image2.shape[1]*3)
arr3=np.reshape(image3,image3.shape[0]*image3.shape[1]*3)
arr4=np.reshape(image4,image3.shape[0]*image3.shape[1]*3)
arr5=np.reshape(image5,image3.shape[0]*image3.shape[1]*3)
arr6=np.reshape(image6,image3.shape[0]*image3.shape[1]*3)

arr7=np.reshape(image7,image.shape[0]*image.shape[1]*3)
arr8=np.reshape(image8,image1.shape[0]*image1.shape[1]*3)
arr9=np.reshape(image9,image2.shape[0]*image2.shape[1]*3)
arr10=np.reshape(image10,image3.shape[0]*image3.shape[1]*3)
arr11=np.reshape(image11,image3.shape[0]*image3.shape[1]*3)
arr12=np.reshape(image12,image3.shape[0]*image3.shape[1]*3)
arr13=np.reshape(image13,image3.shape[0]*image3.shape[1]*3)

arr14=np.reshape(image14,image.shape[0]*image.shape[1]*3)
arr15=np.reshape(image15,image1.shape[0]*image1.shape[1]*3)
arr16=np.reshape(image16,image2.shape[0]*image2.shape[1]*3)

arr17=np.reshape(image17,921600)
arr18=np.reshape(image18,image3.shape[0]*image3.shape[1]*3)
arr19=np.reshape(image19,image3.shape[0]*image3.shape[1]*3)
arr20=np.reshape(image20,image3.shape[0]*image3.shape[1]*3)


arr21=np.reshape(image21,image.shape[0]*image.shape[1]*3)
arr22=np.reshape(image22,image1.shape[0]*image1.shape[1]*3)
arr23=np.reshape(image23,image2.shape[0]*image2.shape[1]*3)
arr24=np.reshape(image24,image3.shape[0]*image3.shape[1]*3)
arr25=np.reshape(image25,image3.shape[0]*image3.shape[1]*3)
arr26=np.reshape(image26,image3.shape[0]*image3.shape[1]*3)
arr27=np.reshape(image27,image3.shape[0]*image3.shape[1]*3)

arr28=np.reshape(image28,image.shape[0]*image.shape[1]*3)
arr29=np.reshape(image29,image1.shape[0]*image1.shape[1]*3)
arr30=np.reshape(image30,image2.shape[0]*image2.shape[1]*3)
arr31=np.reshape(image31,image3.shape[0]*image3.shape[1]*3)
arr32=np.reshape(image32,image3.shape[0]*image3.shape[1]*3)
arr33=np.reshape(image33,image3.shape[0]*image3.shape[1]*3)
arr34=np.reshape(image34,image3.shape[0]*image3.shape[1]*3)
arr35=np.reshape(image35,image3.shape[0]*image3.shape[1]*3)

X1=np.asmatrix([arr,arr1,arr2,arr3,arr4,arr5,arr6,arr7])
X2=np.asmatrix([arr8,arr9,arr10,arr11,arr12,arr13,arr14,arr15,arr16,arr17])
X3=np.asmatrix([arr18,arr19,arr20,arr21,arr22,arr23,arr24,arr25,arr26])
X4=np.asmatrix([arr27,arr28,arr29,arr30,arr31,arr32,arr33,arr34,arr35])


Xtrain= np.asmatrix([arr,arr1,arr2,arr8,arr9,arr10,arr18,arr19,arr20,arr27,arr28,arr29])

Xtrain=Xtrain.T
Xtrain = Xtrain/255.0
Y1=np.asmatrix([[1,0,0,0],[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,1],[0,0,0,1]]).T
############################################################################
#Give input here
X_test=np.asmatrix([arr,arr1,arr2,arr3,arr4,arr5,arr6,arr7,arr8,arr9,arr10,arr11,arr12,arr13,arr14,arr15,arr16,arr17,arr18,arr19,arr20,arr21,arr22,arr23,arr24,arr25,arr26,arr27,arr28,arr29,arr30,arr31,arr32,arr33,arr34,arr35]).T
X_test=X_test/255.0
#######################################################################[0,1,0,0],[0,1,0,0]#####


Y_test=np.asmatrix([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1],[0,0,0,1]]).T

parameters = model(Xtrain, Y1, X_test, Y_test,0.00001,500)
#print parameters

W1 = parameters['W1']
b1 = parameters['b1']
W2 = parameters['W2']
b2 = parameters['b2']
W3 = parameters['W3']
b3 = parameters['b3']

print "W1",W1
print "b1",b1
print "w2",W2
print "b2",b2
print "w3",W3
print "b3",b3


### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
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