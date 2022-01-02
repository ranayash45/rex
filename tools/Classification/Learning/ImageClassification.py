import os.path

import cv2
import os.path
from .boundarydetector import ShapeDetector
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from mahotas.features import haralick
import boto3
from rex.settings import AWS_SECRET_ACCESS_KEY,AWS_STORAGE_BUCKET_NAME,AWS_ACCESS_KEY_ID
import io

def IdentifyMango(imgnp):
    Result = {}
    orignal = imgnp
    #orignal = cv2.resize(orignal,(4000,4000))
    img = orignal
    labformat = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,gray.mean(),255,cv2.THRESH_BINARY)
    thresh = 255 - thresh
    contours,heirachy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    maxIndex = -1
    if len(contours) > 0:
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) >= cv2.contourArea(contours[maxIndex]):
                maxIndex = i
                cv2.contourArea(contours[i])

    img_lab = labformat
    if maxIndex != -1:
        cv2.drawContours(img, contours, maxIndex, (255, 0, 0))
        MangoArea = cv2.contourArea(contours[maxIndex])
        x,y,w,h = cv2.boundingRect(contours[maxIndex])
        print("Detected Mango Area is : {}".format(MangoArea))

        orignal = orignal[y:y+h,x:x+w,:]
        red = orignal[:,:,2].mean()
        green = orignal[:, :, 1].mean()
        blue = orignal[:, :, 0].mean()


    print("Shape of Lab Format",labformat.shape)
    a = img_lab[:,:,1]
    b = img_lab[:,:,2]
    shape = ShapeDetector(img)
    chaincode = shape.detectBoundary()

    channel_a_mean = a.mean()
    channel_b_mean = b.mean()
    print("Channel_A Mean : {}".format(channel_a_mean))
    print("Channel_B Mean : {}".format(channel_b_mean))
    print("Chain Code : {}".format(chaincode))
    Result['channela'] = channel_a_mean
    Result['channelb'] = channel_b_mean
    Result['chaincode'] = chaincode



    import pandas
    import numpy as np
    import matplotlib.pyplot as plt
    s3 = boto3.client('s3',
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    key = 'static/QualityFeatures.csv'
    if not os.path.exists(key):
        obj = s3.get_object(Bucket=AWS_STORAGE_BUCKET_NAME,Key=key)
        fp = open(key,'wb')
        fp.write(obj['Body'].read())
        fp.close()
        del fp
    dataframe = pandas.read_csv(key)
    dataframe = dataframe.sort_values(by='Partially Ripe')
    ripe = dataframe.loc[dataframe['Partially Ripe'] == 'Ripe']
    unripe = dataframe.loc[dataframe['Partially Ripe'] == 'Unripe']
    partially_ripe = dataframe.loc[dataframe['Partially Ripe'] == 'Partially Ripe']
    ripe = ripe.iloc[:110,:].values
    unripe = unripe.iloc[:110,:].values
    partially_ripe = partially_ripe.iloc[:110,:].values

    data = []
    for i in range(ripe.shape[0]):
        data.append(ripe[i,:].tolist())
    for i in range(unripe.shape[0]):
        data.append(unripe[i,:].tolist())
    for i in range(partially_ripe.shape[0]):
        data.append(partially_ripe[i,:].tolist())

    data = np.asarray(data)


    Y = data[:, 0]
    X = data[:, 2:]
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    Y = labelencoder.fit_transform(Y)

    #Y[Y!=0] = 1
    #print(Y)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.6, random_state=1)

    from sklearn.preprocessing import StandardScaler, LabelEncoder

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.svm import SVC

    classifier = SVC(gamma='scale',kernel='rbf', decision_function_shape='ovo')
    #classifier = SVC(kernel='linear', random_state=0, decision_function_shape='ovo')
    classifier.fit(X_train, y_train)

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train,y_train)

    perceptron_classifier = MLPClassifier()
    perceptron_classifier.fit(X_train,y_train)



    from sklearn.metrics import confusion_matrix


    y_pred = classifier.predict(X_test)
    svm_cm = confusion_matrix(y_test, y_pred)
    score = classifier.score(X_test, y_test)

    y_pred = decision_tree.predict(X_test)
    decision_tree_cm = confusion_matrix(y_test, y_pred)
    score2 = decision_tree.score(X_test,y_test)

    y_pred = perceptron_classifier.predict(X_test)
    perceptron_cm = confusion_matrix(y_test, y_pred)
    score3 = perceptron_classifier.score(X_test,y_test)
    print("Score of SVM For ripe and unripe : ",score)
    print("Score of Decision Tree For ripe and unripe : ",score2)
    print("Score of Percenptron Classifier for ripe and unripe : ",score3)
    Result['svm'] = score
    Result['des'] = score2
    Result['pes'] = score3
    from matplotlib.colors import ListedColormap

    # X_set, y_set = X_train, y_train
    # X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    #                      np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    # plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    #              alpha=0.75, cmap=ListedColormap(('red', 'green')))
    # plt.xlim(X1.min(), X1.max())
    # plt.ylim(X2.min(), X2.max())
    # for i, j in enumerate(np.unique(y_set)):
    #     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    #                 c=ListedColormap(('red', 'green'))(i), label=j)
    # plt.title('SVM (Training set)')
    # plt.xlabel('Age')
    # plt.ylabel('Estimated Salary')
    # plt.legend()
    # plt.show()
    #
    # # Visualising the Test set results
    # from matplotlib.colors import ListedColormap
    #
    # X_set, y_set = X_test, y_test
    # X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    #                      np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    # plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    #              alpha=0.75, cmap=ListedColormap(('red', 'green')))
    # plt.xlim(X1.min(), X1.max())
    # plt.ylim(X2.min(), X2.max())
    # for i, j in enumerate(np.unique(y_set)):
    #     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    #                 c=ListedColormap(('red', 'green'))(i), label=j)
    # plt.title('SVM (Test set)')
    # plt.xlabel('Age')
    # plt.ylabel('Estimated Salary')
    # plt.legend()
    # plt.show()



    features = [red,green,blue]
    features.extend(haralick(orignal).mean(0))
    features = np.asarray(features).reshape(1,-1)
    features = sc.transform(features)




    svm_val = classifier.predict(features)
    decision_tree_val = decision_tree.predict(features)
    preceptron_val =perceptron_classifier.predict(features)


    #print("SVM classfied Mango as : ",labelencoder.inverse_transform(svm_val))
    #print("Decision classfied Mango as : ",labelencoder.inverse_transform(decision_tree_val))
    #print("Perceptron classfied Mango as : ",labelencoder.inverse_transform(preceptron_val))
    Result['svmp'] = svm_val
    Result['desp'] = decision_tree_val
    Result['pesp'] = preceptron_val
    Result['svml'] = labelencoder.inverse_transform(svm_val)
    Result['desl'] = labelencoder.inverse_transform(decision_tree_val)
    Result['pesl'] = labelencoder.inverse_transform(preceptron_val)

    img = cv2.resize(img,(512,512))
    Result['img'] = img


    key = 'static/QualityFeaturesMango.csv'
    if not os.path.exists(key):

        obj = s3.get_object(Bucket=AWS_STORAGE_BUCKET_NAME,Key=key)
        fp = open(key,
                  'wb')
        fp.write(obj['Body'].read())
        fp.close()
        del fp

    dataframe = pandas.read_csv(key)
    Y = dataframe.iloc[:,0].values
    X = dataframe.iloc[:,2:].values
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    Y = labelencoder.fit_transform(Y.reshape(-1,1))

    # Y[Y!=0] = 1
    # print(Y)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.6, random_state=1)

    from sklearn.preprocessing import StandardScaler, LabelEncoder

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.svm import SVC

    classifier = SVC(gamma='scale', kernel='rbf', decision_function_shape='ovo')
    # classifier = SVC(kernel='linear', random_state=0, decision_function_shape='ovo')
    classifier.fit(X_train, y_train)

    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)

    perceptron_classifier = MLPClassifier()
    perceptron_classifier.fit(X_train, y_train)

    from sklearn.metrics import confusion_matrix

    y_pred = classifier.predict(X_test)
    svm_cm = confusion_matrix(y_test, y_pred)
    score = classifier.score(X_test, y_test)

    y_pred = decision_tree.predict(X_test)
    decision_tree_cm = confusion_matrix(y_test, y_pred)
    score2 = decision_tree.score(X_test, y_test)

    y_pred = perceptron_classifier.predict(X_test)
    perceptron_cm = confusion_matrix(y_test, y_pred)
    score3 = perceptron_classifier.score(X_test, y_test)
    print("Score of SVM For ripe and unripe : ", score)
    print("Score of Decision Tree For ripe and unripe : ", score2)
    print("Score of Percenptron Classifier for ripe and unripe : ", score3)
    Result['tsvm'] = score
    Result['tdes'] = score2
    Result['tpes'] = score3
    from matplotlib.colors import ListedColormap

    # X_set, y_set = X_train, y_train
    # X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    #                      np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    # plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    #              alpha=0.75, cmap=ListedColormap(('red', 'green')))
    # plt.xlim(X1.min(), X1.max())
    # plt.ylim(X2.min(), X2.max())
    # for i, j in enumerate(np.unique(y_set)):
    #     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    #                 c=ListedColormap(('red', 'green'))(i), label=j)
    # plt.title('SVM (Training set)')
    # plt.xlabel('Age')
    # plt.ylabel('Estimated Salary')
    # plt.legend()
    # plt.show()
    #
    # # Visualising the Test set results
    # from matplotlib.colors import ListedColormap
    #
    # X_set, y_set = X_test, y_test
    # X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    #                      np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    # plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    #              alpha=0.75, cmap=ListedColormap(('red', 'green')))
    # plt.xlim(X1.min(), X1.max())
    # plt.ylim(X2.min(), X2.max())
    # for i, j in enumerate(np.unique(y_set)):
    #     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    #                 c=ListedColormap(('red', 'green'))(i), label=j)
    # plt.title('SVM (Test set)')
    # plt.xlabel('Age')
    # plt.ylabel('Estimated Salary')
    # plt.legend()
    # plt.show()

    features = [red, green, blue]
    features.extend(haralick(orignal).mean(0))
    features = np.asarray(features).reshape(1, -1)
    features = sc.transform(features)

    svm_val = classifier.predict(features)
    decision_tree_val = decision_tree.predict(features)
    preceptron_val = perceptron_classifier.predict(features)

    # print("SVM classfied Mango as : ",labelencoder.inverse_transform(svm_val))
    # print("Decision classfied Mango as : ",labelencoder.inverse_transform(decision_tree_val))
    # print("Perceptron classfied Mango as : ",labelencoder.inverse_transform(preceptron_val))
    Result['tsvmp'] = svm_val
    Result['tdesp'] = decision_tree_val
    Result['tpesp'] = preceptron_val
    Result['tsvml'] = labelencoder.inverse_transform(svm_val)
    Result['tdesl'] = labelencoder.inverse_transform(decision_tree_val)
    Result['tpesl'] = labelencoder.inverse_transform(preceptron_val)


    return Result
