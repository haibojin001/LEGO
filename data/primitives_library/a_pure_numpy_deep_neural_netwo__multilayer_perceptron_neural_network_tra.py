def backprop(X,y,model,z1,a1,z2,a2,z3,output,reg_lambda):
    delta3 = output
    delta3[range(X.shape[0]), y] -= 1  #yhat - y
    dW3 = (a2.T).dot(delta3)
    db3 = np.sum(delta3, axis=0, keepdims=True)
    delta2 = delta3.dot(model['W3'].T) * relu_derivative(a2) #if ReLU
    dW2 = np.dot(a1.T, delta2)
    db2 = np.sum(delta2, axis=0)
    #delta2 = delta3.dot(model['W2'].T) * (1 - np.power(a1, 2)) #if tanh
    delta1 = delta2.dot(model['W2'].T) * relu_derivative(a1) #if ReLU
    dW1 = np.dot(X.T, delta1)
    db1 = np.sum(delta1, axis=0)
    # Add regularization terms
    dW3 += reg_lambda * model['W3']
    dW2 += reg_lambda * model['W2']
    dW1 += reg_lambda * model['W1']
    return dW1, dW2, dW3, db1, db2, db3