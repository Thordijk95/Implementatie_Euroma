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
        sku_warehouse_df = pd.read_csv(os.getcwd() + '/Data/sku_loc_list.csv')
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
        train_df, test_df = train_test_split(enc_corrected_df)

        train_features = train_df.drop(columns='Discrepancy')
        train_target = train_df['Discrepancy']

        test_features = test_df.drop(columns='Discrepancy')
        test_target = test_df['Discrepancy']

        model = random_forest_regressor(train_features=train_features, train_target=train_target,
                                        test_features=test_features, test_target=test_target)

        # model = joblib.load(os.getcwd() + '/Model/RandomForest_18_09_12_59.joblib')
        # uncorrected_df = pd.read_csv(os.getcwd() + '/data_w_o_corrections.csv')
        uncorrected_df = uncorrected_df.drop(columns=['buchungs_nr', 'Date_Time'])
        enc_uncorrected_df = apply_encoding(uncorrected_df, ['bewegungsart', 'Magazijn_LN', 'Beweging'])
        prediction_df = pd.DataFrame(columns=enc_uncorrected_df.columns)
        enc_uncorrected_df = enc_uncorrected_df.drop(columns='Discrepancy')
        for i in range(0, len(enc_uncorrected_df)):
            tx = enc_uncorrected_df.iloc[i].copy(deep=True)

            filter_df = prediction_df[
                (prediction_df['artikel_nr'] == tx['artikel_nr']) & (prediction_df['Magazijn_LN'] == tx['Magazijn_LN'])]
            if len(filter_df) > 0:
                tx['Discrepancy n-1'] = filter_df['Discrepancy'].iloc[-1]
            else:
                tx['Discrepancy n-1'] = 0
            prediction = model.predict([tx])
            tx['Discrepancy'] = prediction[0]

            prediction_df.loc[len(prediction_df.index)] = tx

        prediction_df.to_csv(os.getcwd() + '/Output/predictions.csv')

        decoded_df = decoding(prediction_df, ['bewegungsart', 'Magazijn_LN', 'Beweging'])

        decoded_df['artikel_nr'] = convert_type(column=decoded_df['artikel_nr'], dtype=int)

        current_status = get_discrepancy(decoded_df, sku_warehouse_df)
        current_status.to_csv(os.getcwd() + '/Output/current_state.csv')
