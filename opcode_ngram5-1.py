import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm
import xgboost as xgb
import numpy as np
from sklearn.svm import SVC
import pandas as pd
import os

def _2gram_make_dataset(t):
    with open("_2gram_antipatch_multiple_feature_selected", 'r') as reader:
        data = reader.read()
        lines = data.split(",\n")[:-1]
    for i in lines[:t]:
        tmp = i.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
        _2gram_feature_opcode.append(tmp)

    with open("_2gram_antidump_multiple_feature_selected", 'r') as reader:
        data = reader.read()
        lines = data.split(",\n")[:-1]
    for i in lines[:t]:
        tmp = i.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
        if tmp not in _2gram_feature_opcode:
            _2gram_feature_opcode.append(tmp)

    with open("_2gram_antivm_multiple_feature_selected", 'r') as reader:
        data = reader.read()
        lines = data.split(",\n")[:-1]
    for i in lines[:t]:
        tmp = i.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
        if tmp not in _2gram_feature_opcode:
            _2gram_feature_opcode.append(tmp)

def _3gram_make_dataset(t):
    with open("_3gram_antipatch_multiple_feature_selected", 'r') as reader:
        data = reader.read()
        lines = data.split(",\n")[:-1]
    for i in lines[:t]:
        tmp = i.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
        _3gram_feature_opcode.append(tmp)

    with open("_3gram_antidump_multiple_feature_selected", 'r') as reader:
        data = reader.read()
        lines = data.split(",\n")[:-1]
    for i in lines[:t]:
        tmp = i.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
        if tmp not in _3gram_feature_opcode:
            _3gram_feature_opcode.append(tmp)

    with open("_3gram_antivm_multiple_feature_selected", 'r') as reader:
        data = reader.read()
        lines = data.split(",\n")[:-1]
    for i in lines[:t]:
        tmp = i.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
        if tmp not in _3gram_feature_opcode:
            _3gram_feature_opcode.append(tmp)

def _4gram_make_dataset(t):
    with open("_4gram_antipatch_multiple_feature_selected", 'r') as reader:
        data = reader.read()
        lines = data.split(",\n")[:-1]
    for i in lines[:t]:
        tmp = i.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
        _4gram_feature_opcode.append(tmp)

    with open("_4gram_antidump_multiple_feature_selected", 'r') as reader:
        data = reader.read()
        lines = data.split(",\n")[:-1]
    for i in lines[:t]:
        tmp = i.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
        if tmp not in _4gram_feature_opcode:
            _4gram_feature_opcode.append(tmp)

    with open("_4gram_antivm_multiple_feature_selected", 'r') as reader:
        data = reader.read()
        lines = data.split(",\n")[:-1]
    for i in lines:
        tmp = i.replace('"', '').replace(' ', '').replace("]", '').replace("[", '')
        if tmp not in _4gram_feature_opcode:
            _4gram_feature_opcode.append(tmp)



def _2gram_t(t):
    _2gram_selected_df = pd.read_pickle("./2gram_selected.pkl")
    _2gram_selected_test_df = pd.read_pickle("./2gram_selected_test.pkl")
    total_accuracy_result = []
    for feature in tqdm(t, mininterval=1):
        accuracy_result = []
        X_train = _2gram_selected_df.iloc[:-1, :feature]
        X_test = _2gram_selected_test_df.iloc[:-1, :feature]
        y_train = _2gram_selected_df.iloc[:-1, -1]
        y_test = _2gram_selected_test_df.iloc[:-1, -1]

        params = {
            'n_estimators': [300, 400, 500],
            'max_depth': [6, 8, 10, 12],
            'min_samples_leaf': [8, 12, 16],
            'min_samples_split': [8, 12, 16]
        }

        # rf_clf=RandomForestClassifier(random_state=0, n_jobs=-1)
        # grid_cv=GridSearchCV(rf_clf, param_grid=params, cv=2, n_jobs=-1)
        # grid_cv.fit(X_train, y_train)

        # print("최적 하이퍼 파라미터:\n", grid_cv.best_params_)
        # print("최고 예측 정확도: {0:.4f}".format(grid_cv.best_score_))

        rf_clf = RandomForestClassifier(max_features='sqrt', n_estimators=300, max_depth=6, min_samples_leaf=16,
                                        min_samples_split=8, random_state=0)
        rf_clf.fit(X_train, y_train)
        pred = rf_clf.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        # print("Random Forest 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        gnb = GaussianNB()
        pred = gnb.fit(X_train, y_train).predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        # print("GaussianNB 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        """
        knn_result=[]
        for k in range(1,20):
            knn= KNeighborsClassifier(n_neighbors = k)
            knn.fit(X_train, y_train)
            pred=knn.predict(X_test)
            knn_result.append(accuracy_score(y_test,pred))
            #print("KNN 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))
        plt.plot(range(1,20), knn_result)
        plt.title("Best k for KNN")
        plt.xlabel("$k$")
        plt.ylabel("$accuracy$")
        plt.show()
        """

        knn = KNeighborsClassifier(n_neighbors=8)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        #knn_result.append(accuracy_score(y_test, pred))
        accuracy_result.append(accuracy_score(y_test, pred))
        # print("KNN 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        dtc = DecisionTreeClassifier(random_state=0, max_depth=50, max_features='sqrt')
        dtc.fit(X_train, y_train)
        pred = dtc.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        # print("Decision Tree Classifier 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        gb = GradientBoostingClassifier(random_state=0)
        gb.fit(X_train, y_train)
        pred = gb.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        # print("Gradient Boosting Classifier 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        svm = SVC(kernel='linear', C=1.0, random_state=0)
        svm.fit(X_train, y_train)
        pred = svm.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        #print("SVM 예측 정확도:{0:.4f}".format(accuracy_score(y_test, pred)))

        total_accuracy_result.append(accuracy_result)

    total_accuracy_result = np.array(total_accuracy_result)
    print("#########################################\n")
    print(total_accuracy_result)
    print(total_accuracy_result.mean(axis=0))
    print(total_accuracy_result.mean(axis=0).mean(axis=0))

    fig=plt.figure()
    plt.plot(t, total_accuracy_result[:, 0], color='k', label='RF')
    plt.plot(t, total_accuracy_result[:, 1], color='b', label='NB')
    plt.plot(t, total_accuracy_result[:, 2], color='g', label='KNN')
    plt.plot(t, total_accuracy_result[:, 3], color='r', label='DTC')
    plt.plot(t, total_accuracy_result[:, 4], color='c', label='GB')
    plt.plot(t, total_accuracy_result[:, 5], color='m', label='SVM')
    plt.title("2-gram Best t")
    plt.xlabel("$t$")
    plt.ylabel("$accuracy$")
    plt.legend(loc='upper left')
    fig.show()

def _3gram_t(t):
    _3gram_selected_df = pd.read_pickle("./3gram_selected.pkl")
    _3gram_selected_test_df = pd.read_pickle("./3gram_selected_test.pkl")

    total_accuracy_result = []
    for feature in tqdm(t, mininterval=1):
        accuracy_result = []
        X_train = _3gram_selected_df.iloc[:-1, :feature]
        X_test = _3gram_selected_test_df.iloc[:-1, :feature]
        y_train = _3gram_selected_df.iloc[:-1, -1]
        y_test = _3gram_selected_test_df.iloc[:-1, -1]

        params = {
            'n_estimators': [300, 400, 500],
            'max_depth': [6, 8, 10, 12],
            'min_samples_leaf': [8, 12, 16],
            'min_samples_split': [8, 12, 16]
        }

        # rf_clf=RandomForestClassifier(random_state=0, n_jobs=-1)
        # grid_cv=GridSearchCV(rf_clf, param_grid=params, cv=2, n_jobs=-1)
        # grid_cv.fit(X_train, y_train)

        # print("최적 하이퍼 파라미터:\n", grid_cv.best_params_)
        # print("최고 예측 정확도: {0:.4f}".format(grid_cv.best_score_))

        rf_clf = RandomForestClassifier(max_features='sqrt', n_estimators=300, max_depth=6, min_samples_leaf=16,
                                        min_samples_split=8, random_state=0)
        rf_clf.fit(X_train, y_train)
        pred = rf_clf.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        # print("Random Forest 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        gnb = GaussianNB()
        pred = gnb.fit(X_train, y_train).predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        # print("GaussianNB 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        """
        knn_result=[]
        for k in range(1,20):
            knn= KNeighborsClassifier(n_neighbors = k)
            knn.fit(X_train, y_train)
            pred=knn.predict(X_test)
            knn_result.append(accuracy_score(y_test,pred))
            #print("KNN 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))
        plt.plot(range(1,20), knn_result)
        plt.title("Best k for KNN")
        plt.xlabel("$k$")
        plt.ylabel("$accuracy$")
        plt.show()
        """

        knn = KNeighborsClassifier(n_neighbors=8)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        #knn_result.append(accuracy_score(y_test, pred))
        accuracy_result.append(accuracy_score(y_test, pred))
        # print("KNN 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        dtc = DecisionTreeClassifier(random_state=0, max_depth=50, max_features='sqrt')
        dtc.fit(X_train, y_train)
        pred = dtc.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        # print("Decision Tree Classifier 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        gb = GradientBoostingClassifier(random_state=0)
        gb.fit(X_train, y_train)
        pred = gb.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        # print("Gradient Boosting Classifier 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        svm = SVC(kernel='linear', C=1.0, random_state=0)
        svm.fit(X_train, y_train)
        pred = svm.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        #print("SVM 예측 정확도:{0:.4f}".format(accuracy_score(y_test, pred)))

        total_accuracy_result.append(accuracy_result)

    total_accuracy_result = np.array(total_accuracy_result)
    print("#########################################\n")
    print(total_accuracy_result)
    print(total_accuracy_result.mean(axis=0))
    print(total_accuracy_result.mean(axis=0).mean(axis=0))
    fig=plt.figure()
    plt.plot(t, total_accuracy_result[:, 0], color='k', label='RF')
    plt.plot(t, total_accuracy_result[:, 1], color='b', label='NB')
    plt.plot(t, total_accuracy_result[:, 2], color='g', label='KNN')
    plt.plot(t, total_accuracy_result[:, 3], color='r', label='DTC')
    plt.plot(t, total_accuracy_result[:, 4], color='c', label='GB')
    plt.plot(t, total_accuracy_result[:, 5], color='m', label='SVM')
    plt.title("3-gram Best t")
    plt.xlabel("$t$")
    plt.ylabel("$accuracy$")
    plt.legend(loc='upper left')
    fig.show()

def _4gram_t(t):
    _4gram_selected_df = pd.read_pickle("./4gram_selected.pkl")
    _4gram_selected_test_df = pd.read_pickle("./4gram_selected_test.pkl")

    total_accuracy_result = []
    for feature in tqdm(t, mininterval=1):
        accuracy_result = []
        X_train = _4gram_selected_df.iloc[:-1, :feature]
        X_test = _4gram_selected_test_df.iloc[:-1, :feature]
        y_train = _4gram_selected_df.iloc[:-1, -1]
        y_test = _4gram_selected_test_df.iloc[:-1, -1]

        params = {
            'n_estimators': [300, 400, 500],
            'max_depth': [6, 8, 10, 12],
            'min_samples_leaf': [8, 12, 16],
            'min_samples_split': [8, 12, 16]
        }

        # rf_clf=RandomForestClassifier(random_state=0, n_jobs=-1)
        # grid_cv=GridSearchCV(rf_clf, param_grid=params, cv=2, n_jobs=-1)
        # grid_cv.fit(X_train, y_train)

        # print("최적 하이퍼 파라미터:\n", grid_cv.best_params_)
        # print("최고 예측 정확도: {0:.4f}".format(grid_cv.best_score_))

        rf_clf = RandomForestClassifier(max_features='sqrt', n_estimators=300, max_depth=6, min_samples_leaf=16,
                                        min_samples_split=8, random_state=0)
        rf_clf.fit(X_train, y_train)
        pred = rf_clf.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        # print("Random Forest 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        gnb = GaussianNB()
        pred = gnb.fit(X_train, y_train).predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        # print("GaussianNB 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        """
        knn_result=[]
        for k in range(1,20):
            knn= KNeighborsClassifier(n_neighbors = k)
            knn.fit(X_train, y_train)
            pred=knn.predict(X_test)
            knn_result.append(accuracy_score(y_test,pred))
            #print("KNN 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))
        plt.plot(range(1,20), knn_result)
        plt.title("Best k for KNN")
        plt.xlabel("$k$")
        plt.ylabel("$accuracy$")
        plt.show()
        """

        knn = KNeighborsClassifier(n_neighbors=8)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        #knn_result.append(accuracy_score(y_test, pred))
        accuracy_result.append(accuracy_score(y_test, pred))
        # print("KNN 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        dtc = DecisionTreeClassifier(random_state=0, max_depth=50, max_features='sqrt')
        dtc.fit(X_train, y_train)
        pred = dtc.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        # print("Decision Tree Classifier 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        gb = GradientBoostingClassifier(random_state=0)
        gb.fit(X_train, y_train)
        pred = gb.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        # print("Gradient Boosting Classifier 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        svm = SVC(kernel='linear', C=1.0, random_state=0)
        svm.fit(X_train, y_train)
        pred = svm.predict(X_test)
        accuracy_result.append(accuracy_score(y_test, pred))
        #print("SVM 예측 정확도:{0:.4f}".format(accuracy_score(y_test, pred)))

        total_accuracy_result.append(accuracy_result)

    total_accuracy_result = np.array(total_accuracy_result)
    print("#########################################\n")
    print(total_accuracy_result)
    print(total_accuracy_result.mean(axis=0))
    print(total_accuracy_result.mean(axis=0).mean(axis=0))
    fig=plt.figure()
    plt.plot(t, total_accuracy_result[:, 0], color='k', label='RF')
    plt.plot(t, total_accuracy_result[:, 1], color='b', label='NB')
    plt.plot(t, total_accuracy_result[:, 2], color='g', label='KNN')
    plt.plot(t, total_accuracy_result[:, 3], color='r', label='DTC')
    plt.plot(t, total_accuracy_result[:, 4], color='c', label='GB')
    plt.plot(t, total_accuracy_result[:, 5], color='m', label='SVM')
    plt.title("4-gram Best t")
    plt.xlabel("$t$")
    plt.ylabel("$accuracy$")
    plt.legend(loc='upper left')
    fig.show()


def _4gram():
    _4gram_selected_df = pd.read_pickle("./4gram_selected.pkl")
    _4gram_selected_test_df = pd.read_pickle("./4gram_selected_test.pkl")

    accuracy_result = []
    X_train = _4gram_selected_df.iloc[:-1, :35000]
    X_test = _4gram_selected_test_df.iloc[:-1, :35000]
    y_train = _4gram_selected_df.iloc[:-1, -1]
    y_test = _4gram_selected_test_df.iloc[:-1, -1]

    params = {
        'n_estimators': [300, 400, 500],
        'max_depth': [6, 8, 10, 12],
        'min_samples_leaf': [8, 12, 16],
        'min_samples_split': [8, 12, 16]
    }
    """
    rf_clf=RandomForestClassifier(random_state=0, n_jobs=-1)
    grid_cv=GridSearchCV(rf_clf, param_grid=params, cv=2, n_jobs=-1)
    grid_cv.fit(X_train, y_train)

    print("최적 하이퍼 파라미터:\n", grid_cv.best_params_)
    print("최고 예측 정확도: {0:.4f}".format(grid_cv.best_score_))
    """
    rf_clf = RandomForestClassifier(max_features='sqrt', n_estimators=500, max_depth=6, min_samples_leaf=8,
                                    min_samples_split=8, random_state=0)
    rf_clf.fit(X_train, y_train)
    pred = rf_clf.predict(X_test)
    accuracy_result.append(accuracy_score(y_test, pred))
    print("Random Forest 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

    gnb = GaussianNB()
    pred = gnb.fit(X_train, y_train).predict(X_test)
    accuracy_result.append(accuracy_score(y_test, pred))
    print("GaussianNB 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

    """
    knn_result=[]
    for k in range(1,20):
        knn= KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_train, y_train)
        pred=knn.predict(X_test)
        knn_result.append(accuracy_score(y_test,pred))
        print("KNN 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))
    fig=plt.figure()
    plt.plot(range(1,20), knn_result)
    plt.title("Best k for KNN")
    plt.xlabel("$k$")
    plt.ylabel("$accuracy$")
    fig.show()

    """
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    accuracy_result.append(accuracy_score(y_test, pred))
    print("KNN 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

    dtc = DecisionTreeClassifier(random_state=0, max_depth=50, max_features='sqrt')
    dtc.fit(X_train, y_train)
    pred = dtc.predict(X_test)
    accuracy_result.append(accuracy_score(y_test, pred))
    print("Decision Tree Classifier 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

    gb = GradientBoostingClassifier(random_state=0)
    gb.fit(X_train, y_train)
    pred = gb.predict(X_test)
    accuracy_result.append(accuracy_score(y_test, pred))
    print("Gradient Boosting Classifier 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

    svm = SVC(kernel='linear', C=1.0, random_state=0)
    svm.fit(X_train, y_train)
    pred = svm.predict(X_test)
    accuracy_result.append(accuracy_score(y_test, pred))
    print("SVM 예측 정확도:{0:.4f}".format(accuracy_score(y_test, pred)))

def _4gram_round2(t):
    _4gram_selected_df = pd.read_pickle("./4gram_selected.pkl")
    _4gram_selected_test_df = pd.read_pickle("./4gram_selected_test.pkl")

    total_accuracy_result = []
    for feature in tqdm(t, mininterval=1):
        #accuracy_result = []
        X_train = _4gram_selected_df.iloc[:-1, :feature]
        X_test = _4gram_selected_test_df.iloc[:-1, :feature]
        y_train = _4gram_selected_df.iloc[:-1, -1]
        y_test = _4gram_selected_test_df.iloc[:-1, -1]

        gb = GradientBoostingClassifier(random_state=0)
        gb.fit(X_train, y_train)
        pred = gb.predict(X_test)
        #accuracy_result.append(accuracy_score(y_test, pred))
        # print("Gradient Boosting Classifier 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

        total_accuracy_result.append(accuracy_score(y_test, pred))

    total_accuracy_result = np.array(total_accuracy_result)
    print("#########################################\n")
    print(total_accuracy_result)
    fig=plt.figure()
    plt.plot(t, total_accuracy_result, color='k', label='RF')
    plt.title("4-gram Best t")
    plt.xlabel("$t$")
    plt.ylabel("$accuracy$")
    plt.legend(loc='upper left')
    fig.show()


def main():

    _2gram_t(range(100, 787, 100))
    _3gram_t(range(1000,6149,1000))
    _4gram_t(range(1000,23853, 5000))

    #_3gram_t(range(2000,3000,100))
    #_4gram_t(range(30000,35000,500))
    #_4gram_round2(range(34500,35000,50))
    #_4gram()
    plt.show()
if __name__=='__main__':
    _2gram_feature_opcode = []
    _3gram_feature_opcode = []
    _4gram_feature_opcode = []

    main()
