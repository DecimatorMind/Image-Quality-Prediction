def randomforestclassifier():
    print("Running Random Forest Classifier Model")
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import pandas as pd
    import warnings
    from urllib.parse import urlparse
    import mlflow.sklearn
    import logging
    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger(__name__)
    warnings.filterwarnings('ignore')
    from sklearn.metrics import precision_score, recall_score, f1_score, plot_confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    data = pd.read_csv("fer2013.csv")
    data = data.drop(["Usage"], axis=1)
    X = []
    mark = data.iloc[:, -1].values
    for i in range(mark.shape[0]):
        nums = [int(n) for n in mark[0].split()]
        X.append(nums)
    flag = data.iloc[:, :-1].values
    y = np.empty([35887, ])
    for i in range(len(flag)):
        y[i] = flag[i][0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    with mlflow.start_run():
        random = RandomForestClassifier(max_depth=2, random_state=0)
        random.fit(X_train, y_train)
        y_pred = random.predict(X_test)
        accuracy = random.score(X_test, y_test) * 100
        print("Accuracy on Testing Set: ", accuracy)
        precision = precision_score(y_test, y_pred, average='macro')
        print("Precision Score: ", precision)
        recall = recall_score(y_test, y_pred, average='micro')
        print("Recall Score: ", recall)
        fmeasure = f1_score(y_test, y_pred, average='macro')
        print("F-Measure Score: ", fmeasure, "\n")
        plot_confusion_matrix(random, X_test, y_test)
        (rmse, mae, r2) = eval_metrics(y_test, y_pred)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("Fmeasure", fmeasure)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(random, "Randoom_Forest_Classifier",
                                     registered_model_name="Random_Forest_Classifier")
        else:
            mlflow.sklearn.log_model(random, "model")