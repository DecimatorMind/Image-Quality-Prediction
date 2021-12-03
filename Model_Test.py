import pickle
import cv2


def run_sentiment_model():
    b = open('mlruns/0/a67be6d8fa6b49879792bf0c24e33f2b/artifacts/model/model.pkl', 'rb')
    lr = pickle.load(b)
    image = cv2.imread("test.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (48, 48))
    image = image.flatten()
    print(image)
    print(image.shape)
    result = lr.predict(image)
    print(result)
