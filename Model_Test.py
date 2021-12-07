import pickle
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def run_sentiment_model():
    data = pd.read_csv('fer2013.csv')
    X = []
    y = []
    data = data.drop(["Usage"], axis=1)
    for index, row in data.iterrows():
        val = row['pixels'].split(" ")
        X.append(np.array(val, 'float32'))
        y.append(row['emotion'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    X_test = np.array(X_test, 'float32')

    X_test -= np.mean(X_test, axis=0)
    X_test /= np.std(X_test, axis=0)
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
    image = X_test
    knn_from_joblib = joblib.load('mlruns/0/a67be6d8fa6b49879792bf0c24e33f2b/artifacts/model/model.pkl')
    result = knn_from_joblib.predict(image)
    mark = -9999999;
    flag = 0
    for i in range(len(result[0])):
        if result[0][i] > mark:
            mark = result[0][i]
            flag = i
    print(flag)
    if flag == 2:
        return True
    else:
        return False


def run_occlusion():
    new_model = tf.keras.models.load_model('mask_model/')
    data = []
    image = load_img("test.jpg", target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    data.append(image)
    data = np.array(data, dtype='float32')
    result = new_model.predict(data)
    print(result)
    flag = 0
    if result[0][0] > result[0][1]:
        print("Occlusion Detected")
        flag = 0
    else:
        print("Occlusion not Detected")
        flag = 1
    if flag == 1:
        return True
    else:
        return False


def result_return():
    flag1 = run_sentiment_model()
    flag2 = run_occlusion()
    if flag2:
        print("Image Fine for Use")
    else:
        print("Image not for Use")
