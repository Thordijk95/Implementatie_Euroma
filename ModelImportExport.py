import datetime
import os
from joblib import dump



def export_fit(best_model, algo_name):
    # store the fit of a model to be used later on new data

    now = datetime.datetime.now()
    day = now.strftime("%d")
    month = now.strftime("%m")
    hour = now.strftime("%H")
    minute = now.strftime("%M")
    my_t_stamp = day + "_" + month + "_" + hour + "_" + minute

    model_name = os.getcwd() + '/Model/' + algo_name + ".joblib"
    dump(best_model, model_name)
