import numpy as np


class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.w=None
        self.bias=None

    def fit(self, X,y):
        #init param
        n_samples,n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0

        for _ in range(self.n_iters):
            # Gradient Descent MSE
            y_predicted=np.dot(X,self.weights)+self.bias

            dw=(1/n_samples)*np.dot(X.T,y_predicted-y)*2
            db=(1/n_samples)*np.sum(y_predicted-y)*2

            self.weights-=self.lr*dw
            self.bias-=self.lr*db
    def predict(self,X):
        y_predicted=np.dot(X,self.weights)+self.bias
        return y_predicted

if __name__ == "__main__":
    #Import
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt
    
    #Dataset
    X,y=datasets.make_regression(n_samples=100,n_features=1, noise=20,random_state=4)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)

    # fig=plt.figure(figsize=(8,6))
    # plt.scatter(X[:,0],y,color='b',marker='o',s=30)
    # plt.show()

    rgs=LinearRegression(lr=0.001,n_iters=10000)
    rgs.fit(X_train,y_train)
    predicted=rgs.predict(X_test)

    y_pred_line=rgs.predict(X)
    plt.scatter(X[:,0],y,color='b',marker='o',s=30)
    plt.plot(X,y_pred_line,color='black',linewidth=2,label='prediction')
    plt.show()
