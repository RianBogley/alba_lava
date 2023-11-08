# %% LAVA Data Merging Functions by Rian Bogley ###############################
###############################################################################
# %% PACKAGES #################################################################
# Import Packages
import pandas as pd
import numpy as np
###############################################################################
# %% GLOBAL VARIABLES #########################################################
###############################################################################
# %% FUNCTIONS ################################################################
# Merge LAVA data with input list based on PIDN and DCDate
def merge_lava(df_input, df_lava, tolerance = '365 Days'):
    """
    Merge Data - Main Function
    """
    # Copy df_input as df_merged dataframe
    df_merged = df_input.copy()

    # Check for any cases in df_merged

    # For any cases with a missing DCDate in the input dataset,
    # save them in a temporary different dataframe and delete them from the main one (for now)
    if 'DCDate' in df_merged.columns:
        df_missing_dates = df_merged[df_merged['DCDate'].isnull()].copy()
        df_merged = df_merged[df_merged['DCDate'].notnull()].copy()

    if 'DCDate' in df_merged.columns:
            # Convert DCDate column into Date Values leaving only YYYY-MM-DD format and sort values by DCDate:
            df_merged['DCDate'] = pd.to_datetime(df_merged['DCDate'])
            df_merged.sort_values('DCDate', inplace=True)
    # Set tolerance based on tolerance input
    tolerance = pd.Timedelta(tolerance)

    for df in df_lava:
        # Create a copy of the df_lava[df] dataframe input as df
        df_temp = df_lava[df].copy()
        # If DCDate exists in df, convert to datetime and sort by DCDate
        if 'DCDate' in df_temp.columns:
            # Convert DCDate column into Date Values leaving only YYYY-MM-DD format and sort values by DCDate:
            df_temp['DCDate'] = pd.to_datetime(df_temp['DCDate'])
            df_temp.sort_values('DCDate', inplace=True)
            # Create a copy of the DCDate column to the right of the PIDN column and append the df name to the column name:
            df_temp.insert(1, 'DCDate' + '_' + df, df_temp['DCDate'])
            # Merge df_lava[df] to df_merged based on PIDN and DCDate:
            df_merged = pd.merge_asof(left=df_merged, right=df_temp, on='DCDate', by='PIDN', direction='nearest', tolerance=tolerance, suffixes=['','_' + df])
        elif 'DCDate' not in df_lava[df].columns:
            # Merge df_lava[df] to df_merged based on PIDN only:
            df_merged = pd.merge(left=df_merged, right=df_temp, on='PIDN', how='left', suffixes=['','_'+df])

    # # Re-add the cases with missing DCDate back to the main dataframe if any cases existed without a DCDate
    # if 'DCDate' in df_merged.columns and df_missing_dates.empty == False:
    #     df_merged = df_merged.append(df_missing_dates)

    return df_merged
###############################################################################
# Merge REDCap data with input list based on PIDN only
def merge_redcap(df_input, df_redcap, tolerance = '365 Days', suffix = '_redcap'):
    """
    Merge Data - Main Function
    """
    # Copy df_input as df_merged dataframe
    if 'PIDN' in df_input.columns:
        df_merged = df_input.copy()
    elif 'pidn' in df_input.columns:
        df_merged = df_input.copy()
        df_merged.rename(columns={'pidn':'PIDN'}, inplace=True)
    elif 'PIDN' not in df_input.columns:
        raise ValueError('PIDN column does not exist in input dataframe. Try again.')
    if 'DCDate' in df_merged.columns:
            # Convert DCDate column into Date Values leaving only YYYY-MM-DD format and sort values by DCDate:
            df_merged['DCDate'] = pd.to_datetime(df_merged['DCDate'])
            df_merged.sort_values('DCDate', inplace=True)
    df_temp = df_redcap.copy()
    if 'pidn' in df_temp.columns:
        df_temp.rename(columns={'pidn':'PIDN'}, inplace=True)
    elif 'PIDN' in df_temp.columns:
        pass
    elif 'pidn' and 'PIDN' not in df_temp.columns:
        raise ValueError('PIDN column does not exist in REDCap dataframe. Try again.')
    # If 'PIDN' column in both df_merged and df_temp, merge df_redcap to df_merged (aka df_input) based on PIDN only:
    if 'PIDN' in df_merged.columns and 'PIDN' in df_temp.columns:
        df_merged = pd.merge(left=df_merged, right=df_temp, on='PIDN', how='left', suffixes=['',suffix])
    return df_merged
###############################################################################
# Check for duplicate PIDNs in dataframe
def duplicate_pidn_check(df):
    """
    Check for duplicate PIDNs in dataframe.
    """
    # Check if there are duplicate pidns in cases with no dates
    if df['PIDN'].nunique() != len(df['PIDN']):
        raise ValueError('Duplicate PIDN values exist in dataframe. Cannot run without specifying a date column.')
    else:
        print('No duplicate PIDNs found in dataframe.')
    return
###############################################################################