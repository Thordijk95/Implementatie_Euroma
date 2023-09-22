from sklearn.ensemble import RandomForestRegressor
from ModelImportExport import export_fit
from datetime import datetime
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def random_forest_regressor(train_features, train_target, test_features, test_target):
    hyper_param = {"n_estimators": 15, "min_samples_split": 3, "min_samples_leaf": 5, "max_features": "sqrt",
                    "max_depth": 25, "criterion": "friedman_mse"}

    # convert the features that are strings to numerical codes
    np_train_features = np.array(train_features)
    np_train_target = np.array(train_target)
    np_train_target = np.abs(np_train_target)
    transposed = np_train_target.ravel()

    np_test_features = np.array(test_features)
    np_test_target = np.array(test_target)
    np_test_target = np.abs(np_test_target)

    reg_model = RandomForestRegressor(n_estimators=15, min_samples_split=3, min_samples_leaf=5, max_features='sqrt',
                                      max_depth=25, criterion='friedman_mse')
    print('Start Random Forrest model fit, randomized search')
    starttime = datetime.now()
    print('start model fit: ' + str(starttime))
    # Random search cv
    reg_model.fit(np_train_features, transposed)
    endtime = datetime.now()
    print('End model fit: ' + str(endtime))
    prediction = reg_model.predict(np_test_features)
    print(prediction)
    mse = mean_squared_error(np_test_target, prediction)
    r2 = r2_score(np_test_target, prediction)
    print("MSE: " + str(mse))
    print("R2: " + str(r2))
    export_fit(reg_model, "RandomForest")

    return reg_model




