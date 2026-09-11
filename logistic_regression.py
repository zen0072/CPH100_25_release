import numpy as np
import tqdm
import math
class LogisticRegression():
    """
        A logistic regression model trained with stochastic gradient descent.
    """

    def __init__(self, num_epochs=100, learning_rate=1e-4, batch_size=16, regularization_lambda=0,  verbose=False):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose
        self.regularization_lambda = regularization_lambda

    def fit(self, X, Y):
        """
            Train the logistic regression model using stochastic gradient descent.
        """
        rows, features = X.shape
        self.theta= np.zeros(features)
        self.bias = 0.0
        for i in range(self.num_epochs):
            for e in range(0,rows, self.batch_size):
                u = e + self.batch_size
                newX = X[e:u]
                newY=Y[e:u]
                l_theta, l_b = self.gradient(newX, newY)
                self.theta -= self.learning_rate * l_theta
                self.bias -=self.learning_rate * l_b


        p = self.predict_proba(X)
        eps = 1e-10
        p = np.clip(p, eps, 1 - eps)

        loss = -np.mean(
            Y * np.log(p) + (1 - Y) * np.log(1 - p)
    )

    # L2 regularization
        loss += self.regularization_lambda * np.sum(self.theta ** 2) / 2

        self.train_loss = loss


        #raise NotImplementedError("Not implemented yet")

    def gradient(self, X, Y):
        """
            Compute the gradient of the loss with respect to theta and bias with L2 Regularization.
            Hint: Pay special attention to the numerical stability of your implementation.
        """

        p = self.predict_proba(X)
        eps = 1e-10
        p = np.clip(p, eps, 1-eps)
        l_theta = (X.T @ (p-Y))/len(Y) +self.regularization_lambda *self.theta
        l_b = np.mean(p-Y)
        return l_theta, l_b
        raise NotImplementedError("Not implemented yet")


    def predict_proba(self, X):
        """
            Predict the probability of lung cancer for each sample in X.
        """
        #z= thetaT *X + b
        z= X @ self.theta + self.bias
        z = np.clip(z, -500, 500)
        proba = 1/(1+ np.exp(-z))
        return proba
        raise NotImplementedError("Not implemented yet")

    def predict(self, X, threshold=0.5):
        """
            Predict the if patient will develop lung cancer for each sample in X.
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)