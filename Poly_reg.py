# Solutions to polynomial regression exercise

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
####
import numpy as np
import csv

# read input from HackerRank
def read_input():
    train_data = list()
    test_data = list()
    F, N = map(int,input().split(' '))
    [train_data.append(input().split(' ')) for _ in range(0,N)]
    T = int(input())
    [test_data.append(input().split(' ')) for _ in range(0,T)]
    train_data = np.array(train_data,dtype=np.float64)
    test_data = np.array(test_data,dtype=np.float64)
    X_train = train_data[:,0:F]
    Y_train = train_data[:,-1]
    X_test = test_data
    return X_train,Y_train,X_test

# read input from local source
def read_csv():
    results = []
    with open("idea.csv") as csvfile:
        reader = csv.reader(csvfile)#, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            results.append(row)
    N = int(results[0][1])
    X_train = np.zeros((N,2))
    Y_train = np.zeros((N,))
    for i in range(N):
        Y_train[i,] = float(results[i+1][2])
        for j in range(2):
            X_train[i,j] = float(results[i+1][j])
    T = int(results[N+1][0])
    X_test = np.zeros((T,2))
    for i in range(T):
        for j in range(2):
            X_test[i,j] = float(results[N+i+2][j])
    return X_train,Y_train,X_test

# train model
def fit_and_predict(X_train,Y_train,X_test):
    degree=2
    model = make_pipeline(PolynomialFeatures(degree), Ridge(normalize=False))
    model.fit(X_train,Y_train)
    #scores.append(model.score(X_train,Y_train))
    Y_test = model.predict(X_test)
    return Y_test

def main():
    X_train, Y_train, X_test = read_csv()
    result = fit_and_predict(X_train,Y_train,X_test)
    return '\n'.join(list(map(str,result)))

if __name__ == '__main__':
    prediction = main()
    print(prediction)
