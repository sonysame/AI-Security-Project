import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm
import numpy as np

_2gram_selected_df = pd.read_pickle("./2gram_selected.pkl")
_2gram_selected_test_df = pd.read_pickle("./2gram_selected_test.pkl")

total_accuracy_result = []
for feature in tqdm(range(200, 825, 100), mininterval=1):
    accuracy_result = []
    X_train = _2gram_selected_df.iloc[:, :feature]
    X_test = _2gram_selected_test_df.iloc[:, :feature]
    y_train = _2gram_selected_df.iloc[:, -1]
    y_test = _2gram_selected_test_df.iloc[:, -1]

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

    dtr = DecisionTreeRegressor(random_state=0, max_depth=50, max_features='sqrt')
    dtr.fit(X_train, y_train)
    pred = dtr.predict(X_test)
    accuracy_result.append(accuracy_score(y_test, pred))
    # print("Decision Tree Regressor 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

    gb = GradientBoostingClassifier(random_state=0)
    gb.fit(X_train, y_train)
    pred = gb.predict(X_test)
    accuracy_result.append(accuracy_score(y_test, pred))
    # print("Gradient Boosting Classifier 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

    total_accuracy_result.append(accuracy_result)

total_accuracy_result = np.array(total_accuracy_result)

fig=plt.figure()
plt.plot(range(200, 825, 100), total_accuracy_result[:, 0], color='k', label='RF')
plt.plot(range(200, 825, 100), total_accuracy_result[:, 1], color='b', label='NB')
plt.plot(range(200, 825, 100), total_accuracy_result[:, 2], color='g', label='KNN')
plt.plot(range(200, 825, 100), total_accuracy_result[:, 3], color='r', label='DTC')
plt.plot(range(200, 825, 100), total_accuracy_result[:, 4], color='c', label='DTR')
plt.plot(range(200, 825, 100), total_accuracy_result[:, 5], color='m', label='GB')

plt.title("Best t")
plt.xlabel("$t$")
plt.ylabel("$accuracy$")
plt.legend(loc='upper left')
fig.show()

_3gram_selected_df = pd.read_pickle("./3gram_selected.pkl")
_3gram_selected_test_df = pd.read_pickle("./3gram_selected_test.pkl")

total_accuracy_result = []
for feature in tqdm(range(300, 6571, 1000), mininterval=1):
    accuracy_result = []
    X_train = _3gram_selected_df.iloc[:, :feature]
    X_test = _3gram_selected_test_df.iloc[:, :feature]
    y_train = _3gram_selected_df.iloc[:, -1]
    y_test = _3gram_selected_test_df.iloc[:, -1]

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

    dtr = DecisionTreeRegressor(random_state=0, max_depth=50, max_features='sqrt')
    dtr.fit(X_train, y_train)
    pred = dtr.predict(X_test)
    accuracy_result.append(accuracy_score(y_test, pred))
    # print("Decision Tree Regressor 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

    gb = GradientBoostingClassifier(random_state=0)
    gb.fit(X_train, y_train)
    pred = gb.predict(X_test)
    accuracy_result.append(accuracy_score(y_test, pred))
    # print("Gradient Boosting Classifier 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

    total_accuracy_result.append(accuracy_result)

total_accuracy_result = np.array(total_accuracy_result)

fig=plt.figure()
plt.plot(range(300, 6571, 1000), total_accuracy_result[:, 0], color='k', label='RF')
plt.plot(range(300, 6571, 1000), total_accuracy_result[:, 1], color='b', label='NB')
plt.plot(range(300, 6571, 1000), total_accuracy_result[:, 2], color='g', label='KNN')
plt.plot(range(300, 6571, 1000), total_accuracy_result[:, 3], color='r', label='DTC')
plt.plot(range(300, 6571, 1000), total_accuracy_result[:, 4], color='c', label='DTR')
plt.plot(range(300, 6571, 1000), total_accuracy_result[:, 5], color='m', label='GB')

plt.title("Best t")
plt.xlabel("$t$")
plt.ylabel("$accuracy$")
plt.legend(loc='upper left')
fig.show()
"""
_4gram_selected_df = pd.read_pickle("./4gram_selected.pkl")
_4gram_selected_test_df = pd.read_pickle("./4gram_selected_test.pkl")

total_accuracy_result = []
for feature in tqdm(range(20000, 25253, 1000), mininterval=1):
    accuracy_result = []
    X_train = _4gram_selected_df.iloc[:, :feature]
    X_test = _4gram_selected_test_df.iloc[:, :feature]
    y_train = _4gram_selected_df.iloc[:, -1]
    y_test = _4gram_selected_test_df.iloc[:, -1]

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

    dtr = DecisionTreeRegressor(random_state=0, max_depth=50, max_features='sqrt')
    dtr.fit(X_train, y_train)
    pred = dtr.predict(X_test)
    accuracy_result.append(accuracy_score(y_test, pred))
    # print("Decision Tree Regressor 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

    gb = GradientBoostingClassifier(random_state=0)
    gb.fit(X_train, y_train)
    pred = gb.predict(X_test)
    accuracy_result.append(accuracy_score(y_test, pred))
    # print("Gradient Boosting Classifier 예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))

    total_accuracy_result.append(accuracy_result)

total_accuracy_result = np.array(total_accuracy_result)
fig=plt.figure()
plt.plot(range(6000, 25253, 1000), total_accuracy_result[:, 0], color='k', label='RF')
plt.plot(range(6000, 25253, 1000), total_accuracy_result[:, 1], color='b', label='NB')
plt.plot(range(6000, 25253, 1000), total_accuracy_result[:, 2], color='g', label='KNN')
plt.plot(range(6000, 25253, 1000), total_accuracy_result[:, 3], color='r', label='DTC')
plt.plot(range(6000, 25253, 1000), total_accuracy_result[:, 4], color='c', label='DTR')
plt.plot(range(6000, 25253, 1000), total_accuracy_result[:, 5], color='m', label='GB')

plt.title("Best t")
plt.xlabel("$t$")
plt.ylabel("$accuracy$")
plt.legend(loc='upper left')
"""
plt.show()