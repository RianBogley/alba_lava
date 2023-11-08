# %% LAVA Data Importing Functions by Rian Bogley #############################
###############################################################################
# %% PACKAGES #################################################################
# Import Packages
import os
import pandas as pd
###############################################################################
# %% GLOBAL VARIABLES #########################################################
###############################################################################
# %% FUNCTIONS ################################################################

# ---------------- IMPORT INPUT DATASET FROM CSV OR XLSX FILE ---------------- #
def import_input_file(input_file):
    """
    Import Input CSV or XLSX File
    """
    print("Importing input dataset...")

    # Check if the file is a CSV or XLSX file:
    if input_file.endswith('.csv'):
        print('Input file is a CSV file.')
        df = pd.read_csv(input_file, encoding='UTF-8')
    elif input_file.endswith('.xlsx'):
        print('Input file is an XLSX file.')
        df = pd.read_excel(input_file, sheet_name=0)
    else:
        raise ValueError('Input file is not a CSV or XLSX file. Please fix and try again.')

    # Check if df has PIDN column, otherwise return error and stop script:
    if 'PIDN' not in df.columns:
        raise ValueError('Input file must have PIDN column. Please fix and try again.')
    
    # Check if there are any invalid values in the PIDN column (i.e. not an int)
    if df['PIDN'].apply(lambda x: isinstance(x, int)).all() == False:
        raise ValueError('Input file PIDN column has invalid values. Please fix and try again.')

    # Check if df has DCDate column, if not continue with just PIDN:
    if 'PIDN' in df.columns and 'DCDate' not in df.columns:
        print('Input file has PIDN column but does not have a DCDate column. Continuing with just PIDN.')

    # Check if df has PIDN and DCDate columns, if so continue with both and check for missing values, if so return error:
    if 'PIDN' and 'DCDate' in df.columns:
        print('Input file has PIDN and DCDate columns. Continuing with both.')
        
        # Check if PIDN or DCDate columns have any missing values, if so return error:
        if df['PIDN'].isnull().values.any() or df['DCDate'].isnull().values.any():
            raise ValueError('PIDN and/or DCDate columns have missing values. Please fix and try again.')
        
        # Convert DCDate column into Date Values leaving only YYYY-MM-DD format and convert NaN to NaT:
        df['DCDate'] = pd.to_datetime(df['DCDate'])
        # Sort df by DCDate
        df.sort_values('DCDate', inplace=True)
        # Create a copy of df['DCDate'] column and name it DCDate_input
        df['DCDate_input'] = df['DCDate']

    return df



# TODO:
# ---------------- IMPORT INPUT DATASET BASED ON INPUT METHOD ---------------- #



# -------------------------- IMPORT LAVA QUERY FILE -------------------------- #
def import_lava_query(lava_queries_dir, type):
    """
    Import LAVA Query Data
    """
    df = pd.read_excel(find_files(lava_queries_dir,type), sheet_name=0)
    # If no file is found, return error:
    if df.empty:
        print('No file found for type: ' + type)
    return df

# --------------------- IMPORT LAVA DATA DICTIONARY FILE --------------------- #
def import_lava_dict(lava_queries_dir):
    """
    Import LAVA Data Dictionary
    """
    df = pd.read_excel(find_files(lava_queries_dir,'lava_data_dictionary'), sheet_name=0)
    # If no file is found, return error:
    if df.empty:
        print('No LAVA Data Dictionary file found. Please download the most up-to-date version from LAVA Query and put it in the lava_queries directory.')
    return df

# --------------------- FIND SPECIFIC FILES IN DIRECTORY --------------------- #
def find_files(dir_name, string):
    """
    Find files in directory based on specicific string in filename.
    """
    # Find how many files in directory exist based on specific string in filename. If multiple exist, return error. If only one exists, return that file.
    files = [f for f in os.listdir(dir_name) if string in f]
    if len(files) > 1:
        print('Multiple files found for ' + string + '. Please fix and try again.')
        # Print the name and filepath of each duplicate file:
        for f in files:
            print(os.path.join(dir_name, f))
    elif len(files) == 0:
        print('No file found for ' + string + '. Please fix and try again.')
    else:
        return os.path.join(dir_name, files[0])

# ---------------------------------------------------------------------------- #