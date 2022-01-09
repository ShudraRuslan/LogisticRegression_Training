import Imports as imp


def visualizationData_1(X, y_true):
    df = imp.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y_true))
    colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow'}
    fig, ax = imp.plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    imp.plt.show()


def visualizationData_2(X, y_true):
    df = imp.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y_true))
    colors = {1: 'red', 0: 'blue'}
    fig, ax = imp.plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    imp.plt.show()


def createModel_1(X_train, Y_train, penalty='l2'):
    model = imp.LogisticRegression(penalty=penalty, multi_class="multinomial", solver="lbfgs")

    # visualisation model
    model.fit(X_train, Y_train)
    x_net = imp.np.zeros((800, 1100))
    for i in range(800):
        for j in range(1100):
            if j % 2 == 0:
                x_net[i][j] = -4 + 0.01 * i
            else:
                x_net[i][j] = -1 + 0.01 * j
    x_net = x_net.reshape(440000, 2)
    y_net = model.predict(x_net)
    visualizationData_1(x_net, y_net)
    return model


def createModel_2(X_train, Y_train, penalty='l2'):
    model = imp.LogisticRegression(penalty=penalty)

    # visualisation model
    model.fit(X_train, Y_train)
    x_net = imp.np.zeros((600, 500))
    for i in range(600):
        for j in range(500):
            if j % 2 == 0:
                x_net[i][j] = -3 + 0.01 * i
            else:
                x_net[i][j] = -3 + 0.01 * j
    x_net = x_net.reshape(150000, 2)
    y_net = model.predict(x_net)
    visualizationData_2(x_net, y_net)
    return model


def plotROC(model, testX, testy):
    yhat = model.predict_proba(testX)
    # retrieve just the probabilities for the positive class
    pos_probs = yhat[:, 1]
    # plot no skill roc curve
    imp.plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    # calculate roc curve for model
    fpr, tpr, _ = imp.roc_curve(testy, pos_probs)
    # plot model roc curve
    imp.plt.plot(fpr, tpr, marker='.', label='Logistic')
    # axis labels
    imp.plt.xlabel('False Positive Rate')
    imp.plt.ylabel('True Positive Rate')
    # show the legend
    imp.plt.legend()
    # show the plot
    imp.plt.show()


def plotPR(model, testX, testy, y):
    yhat = model.predict_proba(testX)
    # retrieve just the probabilities for the positive class
    pos_probs = yhat[:, 1]
    # calculate the no skill line as the proportion of the positive class
    no_skill = len(y[y == 1]) / len(y)
    # plot the no skill precision-recall curve
    imp.plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    # calculate model precision-recall curve
    precision, recall, _ = imp.precision_recall_curve(testy, pos_probs)
    # plot the model precision-recall curve
    imp.plt.plot(recall, precision, marker='.', label='Logistic')
    # axis labels
    imp.plt.xlabel('Recall')
    imp.plt.ylabel('Precision')
    # show the legend
    imp.plt.legend()
    # show the plot
    imp.plt.show()


X_1, y_true_1 = imp.make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)
rng = imp.np.random.RandomState(13)
X_stretched = imp.np.dot(X_1, rng.randn(2, 2))
imp.np.random.seed(0)
X_2 = imp.np.random.randn(300, 2)
y_true_2 = imp.np.logical_xor(X_2[:, 0] > 0, X_2[:, 1] > 0)

# visualizationData_1(X_1, y_true_1)
# visualizationData_2(X_2, y_true_2)

X_train_1, X_test_1, y_train_1, y_test_1 = imp.train_test_split(X_1, y_true_1, test_size=0.1)
X_train_2, X_test_2, y_train_2, y_test_2 = imp.train_test_split(X_2, y_true_2, test_size=0.1)

modelWithRegulariz_1 = createModel_1(X_train_1, y_train_1)
modelWithoutRegulariz_1 = createModel_1(X_train_1, y_train_1, "none")
modelWithRegulariz_2 = createModel_2(X_train_2, y_train_2)
modelWithoutRegulariz_2 = createModel_2(X_train_2, y_train_2, "none")



# print(modelWithRegulariz_1.score(X_train_1, y_train_1))
# print(modelWithRegulariz_1.score(X_test_1, y_test_1))
# print(modelWithoutRegulariz_1.score(X_train_1, y_train_1))
# print(modelWithoutRegulariz_1.score(X_test_1, y_test_1))
# print(modelWithRegulariz_2.score(X_train_2, y_train_2))
# print(modelWithRegulariz_2.score(X_test_2, y_test_2))
# print(modelWithoutRegulariz_2.score(X_train_2, y_train_2))
# print(modelWithoutRegulariz_2.score(X_test_2, y_test_2))

# print(modelWithRegulariz_2.predict_proba(X_test_2))

# print(imp.recall_score(y_train_1, modelWithoutRegulariz_1.predict(X_train_1),average="macro"))
# print(imp.recall_score(y_train_1, modelWithRegulariz_1.predict(X_train_1),average="macro"))
# print(imp.recall_score(y_test_1, modelWithoutRegulariz_1.predict(X_test_1),average="macro"))
# print(imp.recall_score(y_test_1, modelWithRegulariz_1.predict(X_test_1),average="macro"))
# print(imp.recall_score(y_train_2, modelWithoutRegulariz_2.predict(X_train_2)))
# print(imp.recall_score(y_train_2, modelWithRegulariz_2.predict(X_train_2)))
# print(imp.recall_score(y_test_2, modelWithoutRegulariz_2.predict(X_test_2)))
# print(imp.recall_score(y_test_2, modelWithRegulariz_2.predict(X_test_2)))

# print(imp.precision_score(y_train_1, modelWithoutRegulariz_1.predict(X_train_1), average="macro"))
# print(imp.precision_score(y_train_1, modelWithRegulariz_1.predict(X_train_1), average="macro"))
# print(imp.precision_score(y_test_1, modelWithoutRegulariz_1.predict(X_test_1),average="macro"))
# print(imp.precision_score(y_test_1, modelWithRegulariz_1.predict(X_test_1),average="macro"))
# print(imp.precision_score(y_train_2, modelWithoutRegulariz_2.predict(X_train_2)))
# print(imp.precision_score(y_train_2, modelWithRegulariz_2.predict(X_train_2)))
# print(imp.precision_score(y_test_2, modelWithoutRegulariz_2.predict(X_test_2)))
# print(imp.precision_score(y_test_2, modelWithRegulariz_2.predict(X_test_2)))

# print(imp.f1_score(y_train_1, modelWithoutRegulariz_1.predict(X_train_1),average="macro"))
# print(imp.f1_score(y_train_1, modelWithRegulariz_1.predict(X_train_1),average="macro"))
# print(imp.f1_score(y_test_1, modelWithoutRegulariz_1.predict(X_test_1),average="macro"))
# print(imp.f1_score(y_test_1, modelWithRegulariz_1.predict(X_test_1),average="macro"))
# print(imp.f1_score(y_train_2, modelWithoutRegulariz_2.predict(X_train_2)))
# print(imp.f1_score(y_train_2, modelWithRegulariz_2.predict(X_train_2)))
# print(imp.f1_score(y_test_2, modelWithoutRegulariz_2.predict(X_test_2)))
# print(imp.f1_score(y_test_2, modelWithRegulariz_2.predict(X_test_2)))

# print(imp.confusion_matrix(y_test_1,modelWithRegulariz_1.predict(X_test_1)))
# print(imp.confusion_matrix(y_test_1,modelWithoutRegulariz_1.predict(X_test_1)))
# print(imp.confusion_matrix(y_test_2,modelWithRegulariz_2.predict(X_test_2)))
# print(imp.confusion_matrix(y_test_2,modelWithoutRegulariz_2.predict(X_test_2)))


# plotROC( modelWithoutRegulariz_2,X_test_2,y_test_2)
# plotPR(modelWithoutRegulariz_2,X_test_2,y_test_2,y_true_2)

# scores = imp.cross_val_score(modelWithRegulariz_1,X_train_1,y_train_1)
# print(scores)
# scores = imp.cross_val_score(modelWithoutRegulariz_1,X_train_1,y_train_1)
# print(scores)
# scores = imp.cross_val_score(modelWithRegulariz_2,X_train_2,y_train_2)
# print(scores)
# scores = imp.cross_val_score(modelWithoutRegulariz_2,X_train_2,y_train_2)
# print(scores)
