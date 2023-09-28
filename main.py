# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import pandas as pd
import numpy as np
import joblib
from Data import get_new_data, merge_data, clean_data, convert_type, replace_character, get_correction, \
    restructure_ln_date_time, insert_tx, extrapolate_features, split_df, integer_encode_multi_column, apply_encoding, \
    get_discrepancy, decoding
from Regression import random_forest_regressor
from testing import start_testing
from sklearn.model_selection import train_test_split
from datetime import datetime

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    testing = False
    # Run the entire sequence
    # a seperate file containts very similar code but broken up into smaller segments for testing
    if testing:
        start_testing()
    # run the full algorithm
    else:
        # get the esa & ln tx data
        esa_df, ln_df = get_new_data()

        # Check the datatypes and if they are not correct, change them
        esa_df['artikel_nr'] = convert_type(column=esa_df['artikel_nr'], dtype='str')
        ln_df['Item_Code'] = convert_type(column=ln_df['Item_Code'], dtype='str')
        ln_df['Transaction_Type_Code'] = convert_type(column=ln_df['Transaction_Type_Code'], dtype='str')
        # replace the comma with a point to indicate decimals
        # Remove the point that indicates thousands
        esa_df['Totaal_gewicht'] = replace_character(esa_df['Totaal_gewicht'], ".", "")
        esa_df['Totaal_gewicht'] = replace_character(esa_df['Totaal_gewicht'], ",", ".")
        esa_df['Totaal_gewicht'] = convert_type(column=esa_df['Totaal_gewicht'], dtype='float')
        esa_df['Stuk_gewicht'] = replace_character(esa_df['Stuk_gewicht'], ".", "")
        esa_df['Stuk_gewicht'] = replace_character(esa_df['Stuk_gewicht'], ",", ".")
        esa_df['Stuk_gewicht'] = convert_type(column=esa_df['Stuk_gewicht'], dtype='float')

        # Remove the articles that are not included in the research
        # all included articles and their warehouse are noted in the following list
        sku_warehouse_df = pd.read_csv(os.getcwd() + '/Conversions/sku_loc_list.csv')
        esa_df = clean_data(esa_df, sku_warehouse_df, sort_col='buchungs_nr', sku_nr_col='artikel_nr',
                            warehouse_col='Magazijn_LN')

        ln_df = clean_data(ln_df, sku_warehouse_df, sort_col='Transaction_Date', sku_nr_col='Item_Code',
                           warehouse_col='Warehouse_Code')
        esa_df = esa_df.reset_index()
        esa_df = esa_df.drop(columns='index')
        ln_df = ln_df.reset_index()
        ln_df = ln_df.drop(columns='index')

        # merge exported csv datasetset
        merged_df = merge_data(esa_df, ln_df)
        # create a column of zeros to later fill with the inventory at the time of transaction
        inv_at_tx = np.zeros(len(merged_df))
        merged_df['Voorraad na mutatie'] = inv_at_tx
        corrections = get_correction(ln_df, 'Transaction_Type_Code', '1')
        corrections = corrections.sort_values(by='Date_Time')
        merged_df = insert_tx(merged_df, corrections)
        merged_df = merged_df.drop(columns=['Tijd', 'Lager', 'Werkplek'])
        merged_df['artikel_nr'] = convert_type(merged_df['artikel_nr'], dtype=int)
        merged_df.to_csv(os.getcwd() + '/tmp.csv')
        full_df = extrapolate_features(merged_df)

        # reorganize the data so that it is the same structure as when training
        sorted_df = full_df[['buchungs_nr', 'Date_Time', 'artikel_nr', 'bewegungsart', 'Totaal_gewicht',
                             'Stuk_gewicht', 'Magazijn_LN', 'Voorraad na mutatie', 'Beweging',
                             'Transactions since correction']]
        sorted_df = sorted_df.reset_index().drop(columns='index')

        # Split the transactions that happen after the last correction for each sku and warehouse combination
        uncorrected_df, corrected_df = split_df(sorted_df, sku_warehouse_df)

        corrected_df = corrected_df.drop(columns=['buchungs_nr', 'Date_Time'])
        enc_corrected_df = integer_encode_multi_column(corrected_df, columns=['bewegungsart', 'Magazijn_LN', 'Beweging'])

        # Split the data in a train and test set
        train_df, test_df = train_test_split(enc_corrected_df)

        train_features = train_df.drop(columns='Discrepancy')
        train_target = train_df['Discrepancy']

        test_features = test_df.drop(columns='Discrepancy')
        test_target = test_df['Discrepancy']

        # train the model
        print('Train the estimator')
        model = random_forest_regressor(train_features=train_features, train_target=train_target,
                                        test_features=test_features, test_target=test_target)
        # create a string for the time stamp
        now = datetime.now()
        day = now.strftime("%d")
        month = now.strftime("%m")
        hour = now.strftime("%H")
        minute = now.strftime("%M")
        my_t_stamp = day + "_" + month + "_" + hour + "_" + minute
        # store the model
        joblib.dump(model, os.getcwd() + '/Model/RandomForest_' + my_t_stamp + '.joblib')
        # remove unneccessary features
        uncorrected_df = uncorrected_df.drop(columns=['buchungs_nr', 'Date_Time'])
        # Apply integer encoding
        enc_uncorrected_df = apply_encoding(uncorrected_df, ['bewegungsart', 'Magazijn_LN', 'Beweging'])
        # Initiate a dataframe to store all predictions
        prediction_df = pd.DataFrame(columns=enc_uncorrected_df.columns)
        # drop the discrepancy column as this is what will be predicted. Shape needs to match that of the training set
        enc_uncorrected_df = enc_uncorrected_df.drop(columns='Discrepancy')
        # Loop over all the uncorrected transactions and predict the discrepancy
        print('Perform predictions')
        for i in range(0, len(enc_uncorrected_df)):
            # Create a copy that can be parsed
            tx = enc_uncorrected_df.iloc[i].copy(deep=True)
            # Filter the dataset to see if the article in the current transaction has already been parsed
            filter_df = prediction_df[
                (prediction_df['artikel_nr'] == tx['artikel_nr']) & (prediction_df['Magazijn_LN'] == tx['Magazijn_LN'])]
            # Match on the article and location, copy the discrepancy from the previous state to the current for N-1
            if len(filter_df) > 0:
                tx['Discrepancy n-1'] = filter_df['Discrepancy'].iloc[-1]
            # no previous transactions, so no available error
            else:
                tx['Discrepancy n-1'] = 0
            # perform the prediction
            prediction = model.predict([tx])
            # Store the prediction in the transaction
            tx['Discrepancy'] = prediction[0]
            # Store the transaction in the prediction dataframe
            prediction_df.loc[len(prediction_df.index)] = tx
            # Progress indicator, gives a tick every 10% of the uncorrected dataframe
            if i % round(len(enc_uncorrected_df)/10) == 0:
                print(i + '/' + len(enc_uncorrected_df))
        # Store the prediction for inspection if needed
        prediction_df.to_csv(os.getcwd() + '/Output/predictions_' + my_t_stamp + '.csv')
        # decode the prediction dataframe
        decoded_df = decoding(prediction_df, ['bewegungsart', 'Magazijn_LN', 'Beweging'])
        # Convert the type of the column to an integer
        decoded_df['artikel_nr'] = convert_type(column=decoded_df['artikel_nr'], dtype=int)
        # Store the current state of the inventory and the predictions on it
        current_prediction = get_discrepancy(decoded_df, sku_warehouse_df)
        current_prediction.to_csv(os.getcwd() + '/Output/current_state_' + my_t_stamp + '.csv')
