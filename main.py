import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def LogisticReg():

    # importing the dataset
    dataset = pd.read_csv('Social_Network_Ads.csv')
    print(dataset.head())
    print(dataset.info())
    print(dataset.describe())
    # plt.scatter(dataset['Age'], dataset['Purchased'], color='orange')
    # plt.scatter(dataset['EstimatedSalary'], dataset['Purchased'], color='blue')
    # plt.show()
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)


    # Training the Logistic Regression model on the Training set
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(x_train, y_train)

    # Predicting a new result
    print(classifier.predict(sc.transform([[30, 87000]])))

    # Predicting the Test set results
    y_pred = classifier.predict(x_test)
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))

   # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    ac = accuracy_score(y_test, y_pred)
    print(ac)

    # Visualising the Training set results
    from matplotlib.colors import ListedColormap
    x_set, y_set = sc.inverse_transform(x_train), y_train
    X1, X2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 10, stop=x_set[:, 0].max() + 10, step=0.25),
                         np.arange(start=x_set[:, 1].min() - 1000, stop=x_set[:, 1].max() + 1000, step=0.25))
    plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                alpha=0.75, cmap=ListedColormap(('red', 'blue')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(('yellow', 'green'))(i), label=j)
    plt.title('Logistic Regression (training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

    # Visualising the Test set results
    from matplotlib.colors import ListedColormap
    x_set, y_set = sc.inverse_transform(x_test), y_test
    X1, X2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 10, stop=x_set[:, 0].max() + 10, step=0.25),
                         np.arange(start=x_set[:, 1].min() - 1000, stop=x_set[:, 1].max() + 1000, step=0.25))
    plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                alpha=0.75, cmap=ListedColormap(('green', 'orange')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(('orange', 'green'))(i), label=j)
    plt.title('Logistic Regression (test set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    LogisticReg()
