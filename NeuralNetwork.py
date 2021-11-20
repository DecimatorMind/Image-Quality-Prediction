from urllib.parse import urlparse

import mlflow
from sklearn.model_selection import train_test_split


def neuralnetwork():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from keras.optimizers import rmsprop_v2
    from keras.losses import categorical_crossentropy
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.utils import np_utils


    data = pd.read_csv('fer2013.csv')
    X = []
    y = []
    data = data.drop(["Usage"], axis=1)
    for index, row in data.iterrows():
        val = row['pixels'].split(" ")
        X.append(np.array(val, 'float32'))
        y.append(row['emotion'])

    num_features = 64
    num_labels = 7
    batch_size = 64
    epochs = 300
    width, height = 48, 48

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    X_train = np.array(X_train, 'float32')
    y_train = np.array(y_train, 'float32')
    X_test = np.array(X_test, 'float32')
    y_test = np.array(y_test, 'float32')

    y_train = np_utils.to_categorical(y_train, num_classes=num_labels)
    y_test = np_utils.to_categorical(y_test, num_classes=num_labels)

    X_train -= np.mean(X_train, axis=0)
    X_train /= np.std(X_train, axis=0)
    X_test -= np.mean(X_test, axis=0)
    X_test /= np.std(X_test, axis=0)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    with mlflow.start_run():

        first_layer_size = 32
        model = Sequential()
        model.add(Dense(first_layer_size, activation='sigmoid', input_shape=(2304,)))
        model.add(Dense(32, activation='sigmoid'))
        model.add(Dense(32, activation='sigmoid'))
        model.add(Dense(7, activation='softmax'))
        model.summary()

        model.compile(loss=categorical_crossentropy,optimizer=rmsprop_v2.RMSprop(), metrics=['accuracy'])
        flag = model.fit(X_train, y_train, batch_size=128, epochs=epochs, verbose=1)

        score = model.evaluate(X_test, y_test, verbose=0)
        print('Test loss:', score[0])
        test_loss = score[0]
        print('Test accuracy:', score[1])
        test_accuracy = score[1]

        prediction = model.predict(X_test)
        prediction = prediction[0]
        print('Prediction\n', prediction)

        flag = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=128)

        plt.plot(flag.history['accuracy'])
        plt.plot(flag.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        mlflow.log_param("Epochs", epochs)
        mlflow.log_param("Batch Size", 28)
        mlflow.log_metric("Test Loss", test_loss)
        mlflow.log_metric("Test Accuracy", test_accuracy)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(model, "Neural Network", registered_model_name="Neural_Network")
        else:
            mlflow.sklearn.log_model(model, "model")