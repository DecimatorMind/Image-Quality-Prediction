from urllib.parse import urlparse

import mlflow
from matplotlib import pyplot as plt


def sentiment_analysis():
    print("Sentiment Analysis")
    import pandas as pd
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras.losses import categorical_crossentropy
    from keras.utils import np_utils

    data = pd.read_csv('fer2013.csv')
    X_train, y_train, X_test, y_test = [], [], [], []

    for index, row in data.iterrows():
        val = row['pixels'].split(" ")
        if 'Training' in row['Usage']:
            X_train.append(np.array(val, 'float32'))
            y_train.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
            X_test.append(np.array(val, 'float32'))
            y_test.append(row['emotion'])

    num_features = 64
    num_labels = 7
    batch_size = 64
    epochs = 20
    width, height = 48, 48

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

    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

    with mlflow.start_run():
        model = Sequential()

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())

        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(num_labels, activation='softmax'))

        model.compile(loss=categorical_crossentropy,metrics=['accuracy'])

        model.fit(X_train, y_train,batch_size=batch_size,epochs=epochs,
                  verbose=1,validation_data=(X_test, y_test),shuffle=True)

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
        mlflow.log_param("Batch Size", 128)
        mlflow.log_metric("Test Loss", test_loss)
        mlflow.log_metric("Test Accuracy", test_accuracy)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(model, "Convolutional Neural Network", registered_model_name="Convolutional_Neural_Network")
        else:
            mlflow.sklearn.log_model(model, "model")