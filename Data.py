import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


def split_df(df, sku_loc):
    loc_df = df.copy(deep=True)

    col_names = []
    for i in range(len(loc_df.columns)):
        col_names.append(loc_df.columns[i])
    col_names.append('Discrepancy')
    col_names.append('Discrepancy n-1')

    complete_uncorrected_df = pd.DataFrame(columns=col_names)
    complete_corrected_df = pd.DataFrame(columns=col_names)


    for i in range(len(sku_loc)):
        tmp_df = loc_df[(loc_df['artikel_nr'] == sku_loc['Artikel'][i]) & (loc_df['Magazijn_LN'] == sku_loc['Opslaglocatie'][i])]
        if len(tmp_df) > 0:
            tmp_df = tmp_df.reset_index().drop(columns='index')
            # Filter returned rows
            # convert the mutation type to numpy array for fast search
            tx_types = tmp_df['bewegungsart'].to_numpy()
            # find the transactions that are corrections
            corrections = np.where(tx_types == 'V')[0]
            if len(corrections) == 0:
                corrections = [0]
            uncorrected_df = tmp_df.loc[corrections[-1]:]
            corrected_df = tmp_df.loc[:corrections[-1]]

            if len(corrected_df) > 0:
                corrected_df = calc_discrepancy(corrected_df, corrections)
                complete_corrected_df = pd.concat([complete_corrected_df, corrected_df])
            if len(uncorrected_df) > 0:
                complete_uncorrected_df = pd.concat([complete_uncorrected_df, uncorrected_df])
    uncorrected_disc = np.zeros(len(complete_uncorrected_df))
    uncorrected_disc_n_1 = np.zeros(len(complete_uncorrected_df))
    complete_uncorrected_df['Discrepancy'] = uncorrected_disc
    complete_uncorrected_df['Discrepancy n-1'] = uncorrected_disc_n_1
    complete_uncorrected_df = complete_uncorrected_df.sort_values(by='Date_Time')
    complete_corrected_df = complete_corrected_df.sort_values(by='Date_Time')
    return complete_uncorrected_df, complete_corrected_df


def get_new_data():
    # initialize the locations of the data
    esa_path = os.getcwd() + '/Data/ESA.csv'  # FIXME change to actual location
    ln_path = os.getcwd() + '/Data/LN.csv'    # FIXME change to actual location

    # import datasets
    esa_df = pd.read_csv(esa_path, sep=";")
    ln_df = pd.read_csv(ln_path, sep=";")

    # remove empty columns and add column with LN warehouses
    esa_empty_cols = ['Artikel_type', 'Lotcode', 'Drager_Krat', 'Magazijnoverboeking', 'Productieorder']
    esa_df = esa_df.drop(columns=esa_empty_cols)
    esa_df['Magazijn_LN'] = convert_location(esa_df['Lager'], esa_df['Opmerkingen'])
    esa_df['Date_Time'] = restructure_esa_date_time(esa_df['Tijd'])

    # Remove empty columns and reformate datetime to be in 24hour mode similar to ESA data
    ln_empty_cols = ['Order_Number', 'Order_Line', 'Line_Sequence_Number', 'Warehouse_From_Code',
                     'Warehouse_From_Description', 'Warehouse_From', 'Warehouse_From_Address_Code',
                     'Warehouse_From_Address_Name', 'Warehouse_From_Address_Country', 'Warehouse_To_Code',
                     'Warehouse_To', 'Warehouse_To_Description', 'Warehouse_To_Address_Code',
                     'Warehouse_To_Address_Name', 'Warehouse_To_Address_Country', 'Lot_Code', 'Order_Type_Code',
                     'Order_Type_Description', 'Order_Type']
    ln_df = ln_df.drop(columns=ln_empty_cols)
    ln_df[['Date', 'Time', 'Month', 'Day', 'Year', 'Hour', 'Minutes', 'Seconds', 'Date_Time']] \
        = restructure_ln_date_time(ln_df['Transaction_Date'])
    # return the data
    return esa_df, ln_df


def get_parsed_data():
    parsed_df = pd.read_csv("some location")  # FIXME change to actual location
    return parsed_df


def clean_data(df, sku_loc_list, sort_col, sku_nr_col, warehouse_col):

    concat_df = pd.DataFrame()
    # loop over the sku_loc_list and filter the df on this data
    for i in range(0, len(sku_loc_list)):
        filtered_df = df[df[sku_nr_col] == str(sku_loc_list['Artikel'].iloc[i])]
        filtered_df = filtered_df[filtered_df[warehouse_col] == str(sku_loc_list['Opslaglocatie'].iloc[i])]

        concat_df = pd.concat([concat_df, filtered_df])
    # when all the data is filtered reorden the transactions
    new_df = concat_df.sort_values(by=sort_col)
    return new_df


def restructure_esa_date_time(date_column):
    print('restructure ESA time data')
    loc_column = date_column.copy(deep=True)
    # date columns are configure date time am/pm, split the columns
    loc_column = loc_column.str.split(' ', expand=True)
    # rename the columns to corresponding names
    loc_column = loc_column.rename(columns={0: 'Date', 1: 'Time'})
    loc_column[['Day', 'Month', 'Year']] = loc_column['Date'].str.split("-", expand=True)

    loc_column['Date'] = loc_column['Year'] + "-" + loc_column['Month'] + "-" + loc_column['Day']
    loc_column['Date_Time'] = loc_column['Date'] + " " + loc_column['Time']

    return loc_column['Date_Time']


def restructure_ln_date_time(date_column):
    print('restructure LN time data')
    # Create local copy
    loc_column = date_column.copy(deep=True)
    # date columns are configure date time am/pm, split the columns
    loc_column = loc_column.str.split(' ', expand=True)
    # rename the columns to corresponding names
    loc_column = loc_column.rename(columns={0: 'Date', 1: 'Time', 2: 'AM/PM'})
    loc_column[['Month', 'Day', 'Year']] = loc_column['Date'].str.split("/", expand=True)
    months = loc_column['Month'].tolist()
    for i in range(len(months)):
        if int(months[i]) < 10:
            months[i] = "0" + str(months[i])
    loc_column['Month'] = months
    loc_column['Date'] = loc_column['Year']+"-"+loc_column['Month']+"-"+loc_column['Day']
    # split the time column
    loc_column[['Hour', 'Minutes', 'Seconds']] = loc_column['Time'].str.split(':', expand=True)
    # Convert to numpy array for faster processing
    hour_array = loc_column['Hour'].to_numpy()
    minute_array = loc_column['Minutes'].to_numpy()
    seconds_array = loc_column['Seconds'].to_numpy()
    am_pm_array = loc_column['AM/PM'].to_numpy()
    # Loop over array and find PM data, add twelve hours
    for i in range(len(hour_array)):
        if am_pm_array[i] == 'PM':
            if int(hour_array[i]) < 12:
                hour_array[i] = str(int(hour_array[i])+12)
            else:
                hour_array[i] = '00'
        if 0 < int(hour_array[i]) < 10:
            hour_array[i] = '0' + hour_array[i]
        if len(minute_array[i]) == 1:
            minute_array[i] = '0' + minute_array[i]
        if seconds_array[i] is None:
            seconds_array[i] = '00'

    # Restore the data to the dataframe
    loc_column['Hour'] = hour_array
    loc_column['Seconds'] = seconds_array
    loc_column['Time'] = loc_column['Hour'] + ":" + loc_column['Minutes'] + ":" + loc_column['Seconds']
    loc_column = loc_column.drop(columns='AM/PM')
    loc_column['Date_Time'] = loc_column['Date'] + " " + loc_column['Time']
    return loc_column


def convert_type(column, dtype):
    print('Convert the datatypes to match')
    # Check the datatypes of the columns
    if column.dtype.name != dtype:
        # if wrong datatype, convert datatype
        column = column.astype(dtype, copy=True, errors='raise')
    return column


def replace_character(column, old_character, new_character):

    column = column.map(lambda x: x.replace(old_character, new_character))

    return column


def convert_location(locations, descriptions):
    print('convert_location')
    esa_ln_locations = pd.read_csv(os.getcwd() + '/ESA_LN_Locations.csv')
    # Create local copy
    missing_found = 0
    missing_locations = []
    missing_descriptions = []
    np_descriptions = descriptions.to_numpy()
    np_location = locations.to_numpy()
    esa_locations = esa_ln_locations['ESA locatie'].to_numpy()
    ln_warehouses = esa_ln_locations['LN warehouse'].to_numpy()
    # create empty list
    warehouses = []
    for i in range(len(np_location)):
        warehouse_index = -1
        if i % (round(len(np_location)/10)) == 0:
            print(str(i) + '/' + str(len(np_location)))
        tmp_loc = np_location[i]
        warehouse_index = np.where(esa_locations == tmp_loc)
        warehouse = ln_warehouses[warehouse_index[0]]
        # conversion = esa_ln_locations[esa_ln_locations['ESA locatie'].str.contains(tmp_loc)]

        if len(warehouse_index[0]) == 0:
            missing_locations.append(locations[i])
            missing_descriptions.append(descriptions[i])
            missing_found += 1
        elif warehouse_index != -1:
            warehouses.append(warehouse[0])
        else:
            warehouses.append(tmp_loc)

    missing_df = pd.DataFrame(columns=['Location', 'Description'])
    missing_df['Location'] = missing_locations
    missing_df['Description'] = missing_descriptions
    missing_df.to_csv(os.getcwd() + "/missing fields.csv")
    # After converting al notations to LN warehouses set data back into df
    ln_warehouses = pd.DataFrame({"Magazijn_LN": warehouses})
    return ln_warehouses


def get_correction(df, col_name, code):

    corrections = df[df[col_name] == code]

    return corrections


def rename_columns(df, old_col, new_col):

    for i in range(0, len(old_col)):
        df = df.rename(columns={old_col[i]: new_col[i]})
    return df


def insert_tx(df, transactions):
    # insertion is best done by comparison on the time
    tmp_df = df.copy(deep=True)
    # initialize two arrays with the time values of both dataframes
    dt_format = '%Y-%m-%d %H:%M:%S'
    new_tx_time_strings = pd.to_datetime(transactions['Date_Time'], format=dt_format)
    tx_time_strings = pd.to_datetime(df['Date_Time'])
    # convert to datetime type so a comparison can be made
    new_tx_times = np.array([np.datetime64(dt_str) for dt_str in new_tx_time_strings])
    tx_time_df = pd.DataFrame(new_tx_times)
    tx_time_df.to_csv(os.getcwd() + '/txtimes.csv')
    tx_times = np.array([np.datetime64(dt_str) for dt_str in tx_time_strings])

    transactions = transactions.drop(columns=['Transaction_Date', 'Warehouse_Description', 'Warehouse',
                                              'Item_Description', 'Item', 'Transaction_Type', 'Transaction_Direction',
                                              'Date', 'Time', 'Month', 'Day', 'Year', 'Hour', 'Minutes', 'Seconds'])
    tmp_df = tmp_df.drop(columns=['Opmerkingen', 'Gebruiker'])

    old_col = ['Warehouse_Code', 'Item_Code', 'Quantity_Original', 'Quantity', 'Transaction_Type_Code',
               'Transaction_Type_Description']

    new_col = ['Magazijn_LN', 'artikel_nr', 'Totaal_gewicht', 'Stuk_gewicht', 'Mutatiesoort', 'Beweging']

    transactions = rename_columns(transactions, old_col, new_col)
    j = 0
    indx = -1
    for i in range(0, len(new_tx_times)):
        while tx_times[j+1] <= new_tx_times[i]:
            j += 1
        # when while exits the correction timestamp is larger than the transaction time stamp
        # insert the correction at this point of the dataset
        # if j is zero, the correction comes before all transactions
        if j == 0:
            corr_tx = transactions.iloc[i:i+1].copy(deep=True)
            boekings_nr = tmp_df['buchungs_nr'].iloc[j] + 1
            corr_tx['buchungs_nr'] = boekings_nr
            corr_tx['bewegungsart'] = 'V'
            corr_tx['Voorraad na mutatie'] = 0
            tmp_df = pd.concat([corr_tx, tmp_df.iloc[:-1]])
            tmp_df = tmp_df.reset_index().drop(columns='index')
            tx_times = np.insert(tx_times, j+1, new_tx_times[i])
        # if j is not the last index of tx times, the correction is placed in the middle
        elif j < len(tx_times):
            corr_tx = transactions.iloc[i:i+1].copy(deep=True)
            boekings_nr = tmp_df['buchungs_nr'].iloc[j] + 1
            corr_tx['buchungs_nr'] = boekings_nr
            corr_tx['bewegungsart'] = 'V'
            corr_tx['Voorraad na mutatie'] = 0
            tmp_df = pd.concat([tmp_df.iloc[:j], corr_tx, tmp_df.iloc[j+1:]])
            tmp_df = tmp_df.reset_index().drop(columns='index')
            tx_times = np.insert(tx_times, j + 1, new_tx_times[i])
        # if end of the tx dataset is reached, the correction goes in the back
        else:
            corr_tx = transactions.iloc[i:i+1].copy(deep=True)
            boekings_nr = tmp_df['buchungs_nr'].iloc[j] + 1
            corr_tx['buchungs_nr'] = boekings_nr
            corr_tx['bewegungsart'] = 'V'
            corr_tx['Voorraad na mutatie'] = 0
            tmp_df = pd.concat([tmp_df.iloc[:-1], corr_tx])
            tmp_df = tmp_df.reset_index().drop(columns='index')
            tx_times = np.insert(tx_times, j + 1, new_tx_times[i])
    return tmp_df


def merge_data(esa_df, ln_df):
    print("merge data")
    # Copy the ESA df as the base for the merged df
    fail_count = 0
    matches = 0
    merged_df = esa_df.copy(deep=True)
    # add arrays to store values in
    matched_inventory_at_tx = np.zeros(len(merged_df), dtype='float')
    matched_quantity = np.zeros(len(merged_df), dtype='float')
    matched_realized_quantity = np.zeros(len(merged_df), dtype='float')
    matched_tx_type = np.zeros(len(merged_df), dtype='str')
    matched_tx_code = np.zeros(len(merged_df), dtype='long')
    # remove am/pm notation from date columns
    esa_df[['Date', 'Time']] = esa_df['Date_Time'].str.split(" ", expand=True)
    esa_df[['Year', 'Month', 'Day']] = esa_df['Date'].str.split("-", expand=True)
    esa_df[['Hour', 'Minute', 'Second']] = esa_df['Time'].str.split(":", expand=True)
    #  compare the transactions of the different databases and merge the data where possible
    #  loop over all rows of the ESA_Df
    for i in range(0, len(esa_df)):
        fail_count += 1
    #  First try matching on date and size
        try:
            # Filter the tx data based on article, warehouse and tx size
            filtered_ln_df = ln_df[ln_df['Item_Code'] == esa_df["artikel_nr"].iloc[i]]
            filtered_ln_df = filtered_ln_df[filtered_ln_df['Warehouse_Code'] == esa_df["Magazijn_LN"].iloc[i]]
            # check for exact match on size
            if len(filtered_ln_df[filtered_ln_df['Quantity'] == esa_df["Totaal_gewicht"].iloc[i]]) == 1:
                filtered_ln_df = filtered_ln_df[filtered_ln_df['Quantity'] == esa_df["Totaal_gewicht"].iloc[i]]
                matched_tx_type[i] = filtered_ln_df['Transaction_Type_Code'].values[0]
                matches += 1
                fail_count -= 1
                continue
            elif len(filtered_ln_df[filtered_ln_df['Quantity'] == esa_df["Totaal_gewicht"].iloc[i]]) > 1:
                filtered_ln_df = filtered_ln_df[filtered_ln_df['Quantity'] == esa_df["Totaal_gewicht"].iloc[i]]
            # Exact match on date/time
            if len(filtered_ln_df[filtered_ln_df['Date'] == esa_df['Date'].iloc[i]]) == 1:
                filtered_ln_df = filtered_ln_df[filtered_ln_df['Date'] == esa_df['Date'].iloc[i]].reset_index()
                # matched_inventory_at_tx[i] = filtered_ln_df['Voorraad_na_mutatie'][0]
                matched_tx_type[i] = filtered_ln_df['Transaction_Type_Code'].values[0]
                matches += 1
                fail_count -= 1
                continue
            else:  # no exact match, create a bound of 5 minutes around the time stamp and check again
                if len(filtered_ln_df[filtered_ln_df['Date'] == esa_df['Date'].iloc[i]]) > 0:
                    filtered_ln_df = filtered_ln_df[filtered_ln_df['Date'] == esa_df['Date'].iloc[i]]

                # split the datetime stamps to parseable objects
                esa_tx_datetime = datetime.strptime(esa_df['Date_Time'].iloc[i], '%Y-%m-%d %H:%M:%S')
                filtered_ln_df = filtered_ln_df.reset_index()
                for j in range(0, len(filtered_ln_df)):
                    # print(str(j) + "/" + str(len(filtered_ln_df)))
                    if " " in filtered_ln_df.Mutatiedatum[0]:
                        ln_tx_datetime = datetime.strptime(filtered_ln_df['Date_Time'].iloc[j][:-8],
                                                           '%Y-%m-%d %H:%M:%S')
                    else:
                        ln_tx_datetime = datetime.strptime(filtered_ln_df['Date_Time'][j], '%Y-%m-%d') 
                    if ((esa_tx_datetime.year == ln_tx_datetime.year) and
                            (esa_tx_datetime.month == ln_tx_datetime.month) and
                            (esa_tx_datetime.day == ln_tx_datetime.day) and
                            (esa_tx_datetime.hour == ln_tx_datetime.hour) and
                            (esa_tx_datetime.minute-5) <= ln_tx_datetime.minute <= (esa_tx_datetime.minute+5)):
                        # matched_inventory_at_tx[i] = filtered_ln_df['Voorraad_na_mutatie'][j]
                        matched_tx_type[i] = filtered_ln_df['Transaction_Type'][j]
                        matched_tx_code[i] = filtered_ln_df['Transaction_Type_Code'][j]
                        matches += 1
                        fail_count -= 1
                        break
            if (i % 1000) == 0:
                print(i)
        except:
            print('failed on:' + str(i))
    merged_df['Mutatiesoort'] = matched_tx_type
    print("failed merges :" + str(fail_count))
    return merged_df


def extrapolate_features(df):
    # Main data preparation loop to structure the data as needed
    print("prep data")
    tmp_df = df.copy(deep=True)
    # Split the data by article
    a_df, nr_articles = split_by_article(tmp_df)
    df_list = []
    for k in range(nr_articles):
        # Split data by location, tx on silo does not affect miniload (case specific example)
        f_l_a_df, nr_locations = split_by_location(a_df[k])
        for j in range(nr_locations):
            if len(f_l_a_df[j].index) > 20:  # only parse sets larger than 20 transactions
                # For the rows not containing a current inventory, calculate what it shoud be based on corrections and transactions
                f_l_a_i_df = calc_expected_inv(f_l_a_df[j])
                # combine the data into a list of dataframes
                df_list.append(f_l_a_i_df)
    # Combine the list back to a dataframe, remove the index and sort on booking nr
    print("Concat full df")
    full_df = pd.concat(df_list)
    full_df = full_df.sort_values(by='Date_Time')
    return full_df


def split_by_article(df):  # Split the transaction data based on the location
    articles = df["artikel_nr"].unique().tolist()  # Find the number of locations being used in the dataset
    tmp_df = df.copy(deep=True)
    db_split = [[]]
    for i in range(len(articles)):
        db_split.append(articles[i])   # Add a location to the list
        db_split[i] = tmp_df.loc[df['artikel_nr'] == articles[i]]  # Locate all rows were the marking equals one of the stored locations, create a new list and reset the index
    return db_split, len(articles)


def split_by_location(df):  # Split the transaction data based on the location
    #  Make a local copy
    tmp_df = df.copy(deep=True)
    #  Convert ESA locations to LN warehouses
    db_split = []
    locations = tmp_df["Magazijn_LN"].unique().tolist()  # Find the number of locations being used in the dataset
    for i in range(len(locations)):
        db_split.append(locations[i])   # Add a location to the list
        db_split[i] = tmp_df[df['Magazijn_LN']==locations[i]].reset_index()  # Locate all rows were the marking equals one of batches, create a new list and reset the index
        db_split[i] = db_split[i].drop(labels='index', axis=1)
    return db_split, len(locations)


def calc_expected_inv(df):
    print("calc_exp_inv")
    # loop over the dataframe and calculate the expected inventory values
    tmp_df = df.copy(deep=True)
    tmp_df = tmp_df.sort_values(by='Date_Time')
    tx_size = np.array(tmp_df['Stuk_gewicht'])
    inv_at_tx = np.array(tmp_df['Voorraad na mutatie'])
    tx_type = np.array(tmp_df['bewegungsart'])
    tx_to_corr = np.zeros(len(tmp_df))
    # loop over the inventory level at time of transaction
    for i in range(0, len(inv_at_tx)):
        if tx_type[i] == 'V':
            tx_to_corr[i] = 0
        else:
            if i == 0:
                tx_to_corr[i] = 0
            else:
                tx_to_corr[i] = tx_to_corr[i-1]+1
        inv_at_tx[i] = inv_at_tx[i-1]+tx_size[i]

    df['Voorraad na mutatie'] = inv_at_tx
    df['Transactions since correction'] = tx_to_corr

    return df


def calc_discrepancy(df, corrections):
    # create local copy of the dataframe to prevent making changes in the original
    loc_df = df.copy(deep=True)
    discrepancy = np.zeros(len(df))
    discrepancy_n_1 = np.zeros(len(df))
    if len(corrections) > 0:
        # create an array to store the discrepanc
        # execute the discrepancy loop for as many corrections as there are present
        for i in range(0, len(corrections)-1):
            if ((corrections[i+1]-1) - (corrections[i])) > 0:
                tmp_df = loc_df.loc[corrections[i]+1:corrections[i+1]-1]
                tx_size = tmp_df['Stuk_gewicht'].to_numpy()
                total_tx = np.sum(np.abs(tx_size))
                change = loc_df['Stuk_gewicht'].iloc[corrections[i+1]]/(corrections[i+1]-1 - corrections[i]+1)
                for j in range(corrections[i]+1, corrections[i+1]):
                    if j == 0:
                        discrepancy[j] = change
                        discrepancy_n_1 = 0
                    else:
                        discrepancy[j] = discrepancy[j-1] + change
                        discrepancy_n_1[j] = discrepancy[j-1]

    loc_df['Discrepancy'] = discrepancy
    loc_df['Discrepancy n-1'] = discrepancy_n_1

    return loc_df


def integer_encode_single_column(column, column_name):
    local = column.copy(deep=True)
    unique = local.unique()
    unique_df = pd.DataFrame([unique])
    # export the encoding for referencing lateron
    path = os.getcwd() + "/Encoding/" + column_name + ".csv"
    unique_df.to_csv(path)
    np_total = local
    for i in range(len(np_total)):
        # if np.where(np_total[i] == unique) == nan:
        #     indx[0][0] = 0
        indx = np.where(np_total[i] == unique)
        np_total[i] = indx[0][0]
    return np_total


def integer_encode_multi_column(df, columns):
    local = df.copy(deep=True)
    for i in columns:
        if i in local.columns:
            local[i] = integer_encode_single_column(local[i], i)
        else:
            print('column: ' + i + ' not in dataframe')
    return local
