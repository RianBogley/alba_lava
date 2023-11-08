# %% IMPORT MODULES ############################################################
# ---------------------------------------------------------------------------- #
#                               IMPORT LIBRARIES                               #
# ---------------------------------------------------------------------------- #
print("Importing standard libraries...")

# Standard libraries:
import getpass
import glob
import math
import os
import pickle
import datetime
import shutil
import sys
import re
import time

# Third party libraries:
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import nibabel as nib
import nilearn.datasets
import nilearn.decoding as decoding
import nilearn.image as image
from nilearn.image import resample_to_img, get_data
from nilearn.input_data import NiftiLabelsMasker
import nilearn.maskers as maskers
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_stat_map, show
import nilearn.plotting as plotting
import numpy as np
import pandas as pd
import pingouin as pg
import plotly as py
import ptitprince as pt
import scipy.stats as stats
import seaborn as sns
import sklearn
from sklearn import model_selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, RidgeCV, LogisticRegressionCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, mean_squared_log_error, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.api import abline_plot, interaction_plot
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.contingency_tables import StratifiedTable
# ---------------------------------------------------------------------------- #
#                             IMPORT ALBA LIBRARIES                            #
# ---------------------------------------------------------------------------- #
print("Importing ALBA libraries...")

# ----------------------------- IMPORT LAVA TOOLS ---------------------------- #
from alba_lava.filtering import (filter_demographics,
                              filter_diagnosis,
                              filter_lava,
                              filter_ni_all,
                              filter_other_cog,
                              filter_t1,
                              filter_timepoints,
                              lava_max_scores)
from alba_lava.importing import (import_input_file,
                              import_lava_dict,
                              import_lava_query,
                              find_files)
from alba_lava.merging import (merge_lava,
                             merge_redcap)

from alba_imaging.imaging_core import (get_lava_wmap_filepaths)

# %% MAIN SETTINGS #############################################################
# ---------------------------------------------------------------------------- #
#                                 MAIN SETTINGS                                #
# ---------------------------------------------------------------------------- #

# ----------------------------- CHOOSE OUTPUT DIR ---------------------------- #
output_dir = '/Volumes/language/language/rbogley/wmaps_decoding/lava_neuropsych/'
# output_dir = '/Volumes/language/language/rbogley/wmaps_decoding/edevhx_project/'

# ---------------------------- CHOOSE INPUT METHOD --------------------------- #
# input_method = 'PIDN' # Method 1: PIDN Only
# input_method = 'PIDN-DCDate' # Method 2: PIDN + DCDate
input_method = 'LAVA' # Method 3: No PIDN or DCDate List, LAVA data only

# -------------------------- IF USING METHOD 1 OR 2 -------------------------- #
# Specify the input CSV/XLSX file path:
# NOTE: Must contain either a "PIDN" or both a "PIDN" and a "DCDate" column.
# input_file = os.path.join(output_dir,'EDevHx_NOFORMULAS_6-16-2023.xlsx')

# -------------------------- IF USING METHOD 1 OR 3 -------------------------- #
# Specify which LAVA dataset to use as the BASE dataset
# Method 1: Will match DCDates to specified PIDNs using the specified
    # LAVA dataset and timepoint.
# Method 3: Will use specified LAVA dataset and timepoint as the BASE dataset.

# NOTE: Specified LAVA Dataset must contain a "DCDate" column (i.e. no demog).
base_dataset = 'neuropsychbedside' # Selects Neuropsych: Bedside instrument.
# base_dataset = 'ni_all' # Selects Neuroimaging instrument.

# Specify which timepoint to use as the BASE timepoint
timepoint = 'first' # Earliest avaialble timepoint.
# timepoint = 'latest' # Latest available timepoint.
# timepoint = 'fullest' # Timepoint with the most data for each participant.
# timepoint = 'all' # All available timepoints.

# -------------------------- SPECIFY LAVA QUERY DATA ------------------------- #
# Specify directory where LAVA query files are downloaded:
lava_queries_dir = '/volumes/language/language/rbogley/LAVA_Queries/'
# Specify which LAVA query datasets to use:
lava_datasets = [
    'demographics',
    'diagnosis',
    'ni_all',
    'neuropsychbedside',
    'neuropsychmmse',
    'cdr',
    'languagebattery',
    'neuropsychcvlt',
    'neuropsychdkefs',
    'moca',
    'catssummary',
    'neuropsychother_cog',
    'ftdppgneuropsych',
    'extracog',
    'udsftldneuropsych',
    'rsmsinformant',
    'udsftldrsms',
    'spatialcogneuropsych'
]

# -------------------- SPECIFY W-MAPS FILEPATH ON R-DRIVE -------------------- #
# Specify main directory on R-Drive for where imaging core generated W-Maps are:
wmaps_dir = '/volumes/macdata/projects/knect/images/wmaps/spmvbm12/'

# %% IMPORT LAVA DATASETS ######################################################
# ---------------------------------------------------------------------------- #
#                             IMPORT LAVA DATASETS                             #
# ---------------------------------------------------------------------------- #
print("Importing LAVA Query datasets...")
# Import all LAVA datasets specified in lava_datasets list:
df_lava = {}
lava_datasets_standard = [
    'neuropsychmmse',
    'cdr',
    'languagebattery',
    'neuropsychcvlt',
    'neuropsychdkefs',
    'moca',
    'catssummary',
    'ftdppgneuropsych',
    'udsftldneuropsych',
    'rsmsinformant',
    'udsftldrsms',
    'spatialcogneuropsych'
    ]

# Check if there are any duplicate queries in the specified LAVA Queries dir using find_files:
# If there are duplicate queries, return error:
for dataset in lava_datasets:
    # If any of the specified datasets have multiple files in the LAVA Queries dir, return error:
    files = []
    # Add any files with the dataset name in the filename to the files list:
    files = [f for f in os.listdir(lava_queries_dir) if dataset in f]
    # If files list has no files, print a warning:
    if len(files) == 0:
        print(f'WARNING: No LAVA Queries found for {dataset}.')
    # If files list has more than one file, exit:
    if len(files) > 1:
        print(f'ERROR: Multiple LAVA Queries found for {dataset}. Please fix and try again.')
        sys.exit()

# Make a list to store all raw PIDN and row count variables:
lava_pidn_counts = []
lava_row_counts = []

for df in lava_datasets:
    df_lava[df] = import_lava_query(lava_queries_dir, df)
    # Save the number of unique PIDNs in each dataset as a variable called pidn_count_raw_{name_of_df}:
    exec(f'pidn_count_raw_{df} = len(df_lava["{df}"]["PIDN"].unique())')
    # Save the number of rows in each dataset as a variable called row_count_raw_{name_of_df}:
    exec(f'row_count_raw_{df} = len(df_lava["{df}"])')
    # Add these metrics to the raw_pidn_count and raw_row_count lists:
    lava_pidn_counts.append(f'pidn_count_raw_{df}')
    lava_row_counts.append(f'row_count_raw_{df}')

    print('Importing and filtering ' + df + ' dataset...')
    # Filter LAVA data:
    if df == 'ni_all':
        # Filter NI_All:
        df_lava[df] = filter_ni_all(df_lava[df])
        # Copy NI_ALL and Filter for only T1 scans:
        df_lava['t1'] = df_lava[df].copy()
        # Filter, sort, and trim the T1 df:
        df_lava['t1'] = filter_t1(df_lava['t1'])
        # Remove the NI_ALL df:
        df_lava.pop('ni_all')
    elif df == 'demographics':
        df_lava[df] = filter_demographics(df_lava[df])
    elif df == 'diagnosis':
         df_lava[df] = filter_diagnosis(df_lava[df])
    elif df == 'neuropsychother_cog' or 'extracog':
        df_lava[df] = filter_other_cog(df_lava[df])
    elif df == 'neuropsychbedside':
        df_lava[df] = filter_lava(df_lava[df])
        # # Calculate PPVT total score as a new variable in bedside:
        # df_lava[df]['PPVT'] = df_lava[df].apply(lambda x: x['LngPVrb'] + x['LngPDes'] + x['LngPAni'] + x['LngPIna'] if pd.notnull(x['LngPVrb']) and pd.notnull(x['LngPDes']) and pd.notnull(x['LngPAni']) and pd.notnull(x['LngPIna']) else np.nan, axis=1)
        # # Calculate MTCorr/MTTime, with fixes for invalid time values:
        # df_lava[df]['MTCorr_MTTime'] = df_lava[df].apply(lambda x: x['MTCorr']/x['MTTime'] if pd.notnull(x['MTCorr']) and pd.notnull(x['MTTime']) and x['MTTime'] > 0 and x['MTTime'] < 121 else np.nan, axis=1)
    # For any other dataset in the standard list above,
    # use the standard filter_lava function:
    elif df in lava_datasets_standard:
        df_lava[df] = filter_lava(df_lava[df])
    # Otherwise, if a specified df does not match any of the above criteria,
    # print an error:
    else:
        print('ERROR: LAVA dataset "' + df + '" not recognized.')

# For each df in df_lava, save the number of unique PIDNs and rows as variables after filtering:
for df in df_lava:
    # After filtering, make a new count of how many PIDN and rows there are and append them to the lava_pidn_counts and lava_row_counts lists:
    exec(f'pidn_count_filtered_{df} = len(df_lava["{df}"]["PIDN"].unique())')
    exec(f'row_count_filtered_{df} = len(df_lava["{df}"])')
    lava_pidn_counts.append(f'pidn_count_filtered_{df}')
    lava_row_counts.append(f'row_count_filtered_{df}')

# Make a backup copy of lava_df for reseting the dataframes later if need be:
df_lava_backup = df_lava.copy()

# Make another backup called df_lava_original without any variable fixes or additions:
df_lava_original = df_lava.copy()

# %% VARIABLE FIXES AND ADDITIONS ##############################################
# Check if the PPVT or MT variables exist in any df_lava dataset, if so add their total scores as columns:
for df in df_lava:
    if 'LngPVrb' in df_lava[df].columns and 'LngPDes' in df_lava[df].columns and 'LngPAni' in df_lava[df].columns and 'LngPIna' in df_lava[df].columns:
        print('Adding PPVT total score to ' + df + ' dataframe...')
        # Calculate PPVT total score as a new variable in bedside:
        df_lava[df]['PPVT'] = df_lava[df].apply(lambda x: x['LngPVrb'] + x['LngPDes'] + x['LngPAni'] + x['LngPIna'] if pd.notnull(x['LngPVrb']) and pd.notnull(x['LngPDes']) and pd.notnull(x['LngPAni']) and pd.notnull(x['LngPIna']) else np.nan, axis=1)
    if 'MTCorr' in df_lava[df].columns and 'MTTime' in df_lava[df].columns:
        print('Adding MTCorr/MTTime to ' + df + ' dataframe...')
        # Calculate MTCorr/MTTime, with fixes for invalid time values:
        df_lava[df]['MTCorr_MTTime'] = df_lava[df].apply(lambda x: x['MTCorr']/x['MTTime'] if pd.notnull(x['MTCorr']) and pd.notnull(x['MTTime']) and x['MTTime'] > 0 and x['MTTime'] < 121 else np.nan, axis=1)
        # # Move the MTCorr_MTTime column to the right of the MTTime column:
        # mttime_col = df_lava[df].columns.get_loc('MTTime')
        # df_lava[df].insert(mttime_col+1, 'MTCorr_MTTime', df_lava[df]['MTCorr_MTTime'])
    if 'Educ' in df_lava[df].columns:
        print('Replacing invalid values in Educ column with NaN in ' + df + ' dataframe...')
        # Replace any values of 99 in 'Educ' with NaN:
        df_lava[df]['Educ'] = df_lava[df]['Educ'].replace(99, np.nan)

# Update the backup copy of lava_df:
df_lava_backup = df_lava.copy()

# Print how many columns remain in each lava_df:
for df in df_lava:
    print('There are ' + str(len(df_lava[df].columns)) + ' columns in the fixed initial ' + df + ' dataframe.')

# %% ADD LAVA W-MAP FILEPATHS TO T1 DATAFRAME ##################################
# If t1 dataframe exist, add W-Map filepaths to it using the get_lava_wmaps_filepaths function:
if 't1' in df_lava:
    # Print how many unique rows there are in the t1 dataframe and how many unique pidns
    print(f'There are {len(df_lava["t1"])} T1 scans, with {len(df_lava["t1"]["PIDN"].unique())} unique PIDNs.')
    print(f'Looking for matching MAC Imaging-Core generated W-Maps for each T1 in {wmaps_dir}...')
    print('Please wait, this may take a while...')
    get_lava_wmap_filepaths(df=df_lava['t1'],pidn_col='PIDN',dcdate_col='DCDate',wmaps_dir=wmaps_dir)
    print(f'Found {df_lava["t1"]["wmap_lava"].notna().sum()} matching W-Maps in {wmaps_dir}.')
    print(f'Could not find {df_lava["t1"]["wmap_lava"].isna().sum()} matching W-Maps in {wmaps_dir}.')
    print(f'Filepaths for these W-Maps were added to the T1 dataframe under the "wmap_lava" column.')

# %% TEMP: FILTER LAVA DATASETS TO ONLY INCLUDE THE VARIABLES SPECIFIED IN THE THE SPECIFEID DICTIONARY BELOW:
# main_var = ['PIDN','DCDate','AgeAtDC','DOB','DOD','Educ']
# additional_var = ['ClinSynBestEst','ClinSynSecEst','ClinSynTerEst','ResDxA','ResDxB','ResDxC','ResDxD','ResDxE',
#                   'ScannerID','SourceID','ScanType','ImageLinkedImgFormat','ScannerNotes','ImgQuality','QualityNotes',
#                   'ScannerID_t1','SourceID_t1','ScanType_t1','ImageLinkedImgFormat_t1','ScannerNotes_t1','ImgQuality_t1','QualityNotes_t1',
#                   'T1Corr','T1Intr','T2Corr','T2Intr','T3Corr','T3Intr','T4Corr','T4Intr','TrCoTot','Corr30','Intr30','Corr10','Intr10','CuedCor','CueIntr','Recog','RecogNP','RecogNU','RecogFP',
#                   'Year','Season','Month','Date','Day','Place','Floor','City','County','State','RepBall','RepFlag','RepTree','Trials','World','RemBall','RemFlag','RemTree','Watch','Pencil','NoIfs','HandPap','FoldPap','OnFloor','Eyes','Sent','Pentgon',
#                   ]
# # Neuropsych Variables by Cognitive Domain:
# # General Cognition:
# generalcog_var = ['MMSETot','CDR Box','CDR Global','MocaTotWithoutEduc']
# # Episodic Memory:
# episodicmemory_var = ['TrCoTot','Corr30','Corr10','Recog','Rey10m']
# # Language:
# language_var = ['BNTCorr','BNTTot','BNTNumS','BNTStim','BNTPhon','BNTMult',
#                 'PPTP14Cor','Verbal','Syntax','Repeat5',
#                 'DogLion','Table','Anger','LoudTie','OldOx','Shallow','Beard','AbsTot',
#                 'WRATWrd','WRATLet','WRATTot','ReadReg','ReadIrr','PPVT']
# # Visuospatial:
# visuospatial_var = ['ModRey','ReyRecg','BryTot','NumbLoc','Calc',
#                     'Pentgon','CATSFMTot','Jolo','FaceRec']
# # Attention Speed & Executive Function:
# executive_var = ['DigitFW','DigitBW','MTTime','MTCorr','MTCorr_MTTime',
#                  'ANCorr','ANReps','ANRuleV','DCorr','DReps','DRuleV',
#                  'DFCorr','DFCoRep','DFRuleV','DFRVRep',
#                  'StrpCNCor','StrpCNErr','StrpCor','StrpErr','StrpSCE']
# # Emotions:
# emotions_var = ['CATSAMTot','GDSTot','RSMS_TOTI']

# # Combine All LAVA Variables:
# lava_var_to_keep = generalcog_var + episodicmemory_var + language_var + visuospatial_var + executive_var + emotions_var
# all_lava_var_to_keep = main_var + additional_var + lava_var_to_keep

# # ---------------------------------------------------------------------------- #
# # Loop through each lava_df and keep only the variables in lava_var_to_keep,
# # if a variable is not in lava_var_to_keep, drop it.
# # If a variable is in lava_var_to_keep, but not in the df, skip it.
# for df in df_lava:
#     # Loop through each column in the df:
#     for col in df_lava[df].columns:
#         # Count how many columns in all_lava_var_to_keep are in the df columns:
#         col_count = sum(x in df_lava[df].columns for x in all_lava_var_to_keep)
#         if col_count == 0:
#             print('There are no variables to keep in the ' + df + ' dataframe. Skipping...')
#             # Delete the df from df_lava:
#             df_lava.pop(df)

#         elif col_count > 0:
#             # If the column is not in lava_var_to_keep or main_var, drop it:
#             if col not in all_lava_var_to_keep:
#                 df_lava[df].drop(columns=col, inplace=True)

# # Print how many columns remain in each lava_df:
# for df in df_lava:
#     print('There are ' + str(len(df_lava[df].columns)) + ' columns left in the ' + df + ' dataframe.')

# %% SET BASE DATASET TO MERGE TO AND INPUT DATA IF USING METHOD 1 OR 2 ########
# ---------------------------------------------------------------------------- #
#                          IMPORT AND FILTER INPUT DATA                        #
# ---------------------------------------------------------------------------- #
# Import input dataset based on input_method:
print("Importing input dataset...")
# Specify which data to import as df_input based on input_method:
if input_method == 'PIDN-DCDate':
    # Import the specified input CSV with PIDNs & DCDates as df_input:
    df_input = import_input_file(input_file)
elif input_method == 'PIDN':
    # Import the specified input CSV with PIDNs only as df_input:
    df_input = import_input_file(input_file)
    # Find the PIDN column number and add a new empty DCDate column after it:
    pidn_col = df_input.columns.get_loc('PIDN')
    df_input.insert(pidn_col+1, 'DCDate', '')
    # Make a temporary dataframe importing data from the specified LAVA dataset and timepoint filter:
    df_temp = filter_timepoints(import_lava_query(lava_queries_dir, base_dataset), timepoint)
    # Add a DCDate column to the right of the PIDN column in df_input and put the DCDates from the temporary dataframe into the df_input dataframe matching by PIDN:
    df_input['DCDate'] = df_input['PIDN'].map(df_temp.set_index('PIDN')['DCDate'])
    # Drop any rows with missing DCDates and print the number of rows dropped:
    # df_input = df_input.dropna(subset=['DCDate'])
elif input_method == 'LAVA':
    # If using LAVA dataset method, import the specified LAVA dataset as df_input:
    df_input = filter_timepoints(df_lava[base_dataset], timepoint)
    # Drop all columns other than PIDN and DCDate:
    df_input = df_input[['PIDN','DCDate']]

# %% MERGE INPUT DATA WITH LAVA DATASETS #######################################
# Merge df_input with df_lava data based on PIDN if no DCDate,
# or PIDN & DCDate if DCDate exists in input:
df_merged = merge_lava(df_input, df_lava, tolerance='365 Days')

# Backup the df_merged dataframe:
df_merged_backup = df_merged.copy()

# %% EXPORT MERGED DATA TO OUTPUT DIRECTORY ####################################
current_date = datetime.datetime.now().strftime('%Y-%m-%d')
# Export the merged data to an excel file in the output directory:
# Specify the output file name:
output_file = f"{output_dir}/LAVA_Merged_Data_{timepoint}_{base_dataset}_{current_date}.csv"
# Export the df_merged dataframe to the output file:
df_merged.to_csv(output_file, index=False, encoding='utf-8-sig')

# %% SAVE LOGFILE ##############################################################

def lava_pipeline_logfile(output_dir):
    """
    Creates a logfile for the LAVA Pipeline.
    """
    # Create a logfile in the output directory:
    logfile = open(f"{output_dir}/LAVA_Pipeline_Logfile_{timepoint}_{base_dataset}_{current_date}.txt", "w")
    
    # Write basic info about LAVA Pipeline:
    logfile.write(f"LAVA Pipeline Datetime: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    # Write the output directory to the logfile:
    logfile.write(f"Output Directory: {output_dir}\n")
    # Write the LAVA datasets to the logfile:
    logfile.write(f"All LAVA Datasets Used: {lava_datasets}\n")
    
    # Specify input method and information:
    logfile.write(f"\n")
    if input_method == 'PIDN-DCDate' or 'PIDN':
        logfile.write(f'Input Method: {input_method}\n')
        # If input_file exists, write the input_file path to the logfile:
        if 'input_file' in globals():
            logfile.write(f"Input File: {input_file}\n")
        logfile.write(f"Base LAVA Dataset Used: {base_dataset}\n")
        if input_method == 'PIDN-DCDate':
            logfile.write(f"Timepoint Specified for Base LAVA Dataset: {timepoint}\n")
        logfile.write(f"Number of Unique PIDNs in df_input: {len(df_input['PIDN'].unique())}\n")
        logfile.write(f"Number of Rows in df_input: {len(df_input)}\n")
    elif input_method == 'LAVA':
        logfile.write(f'Input Method: {input_method}\n')
        logfile.write(f"Base LAVA Dataset Used: {base_dataset}\n")
        logfile.write(f"Timepoint Specified for Base LAVA Dataset: {timepoint}\n")
        logfile.write(f"Number of Unique PIDNs in df_input: {len(df_input['PIDN'].unique())}\n")
        logfile.write(f"Number of Rows in df_input: {len(df_input)}\n")

    # Write the counts for each LAVA dataset before and after filtering, and after merging:
    logfile.write(f"\n")
    for i in range(len(lava_pidn_counts)):
        logfile.write(f"{lava_pidn_counts[i]}: {eval(lava_pidn_counts[i])}\n")
    for i in range(len(lava_row_counts)):
        logfile.write(f"{lava_row_counts[i]}: {eval(lava_row_counts[i])}\n")
    logfile.write(f"\n")
    logfile.write(f"Number of Unique PIDNs in final df_merged: {len(df_merged['PIDN'].unique())}\n")
    logfile.write(f"Number of Rows in final df_merged: {len(df_merged)}\n")

    # Print information about T1 and W-Maps:
    if 't1' in df_lava:
        logfile.write(f"\n")
        logfile.write(f"Number of Unique PIDNs with T1 MRI scan: {len(df_lava['t1']['PIDN'].unique())}\n")
        logfile.write(f"Total number of scans in T1 dataset: {len(df_lava['t1'])}\n")
        # Print the amount of cases with W-Maps:
        logfile.write(f"\n")
        logfile.write(f"W-Maps Directory: {wmaps_dir}\n")
        logfile.write(f"Number of Unique PIDNs with Imaging-Core created W-Maps in T1 dataset: {len(df_lava['t1']['PIDN'].unique())}\n")
        logfile.write(f"Total number of Imaging-Core created W-Maps in T1 dataset: {len(df_lava['t1'][df_lava['t1']['wmap_lava'].notna()])}\n")
        # Print how many T1 and W-Maps remained in df_merged
        logfile.write(f"\n")
        # Count how many cases have a value 
        logfile.write(f"Number of Unique PIDNs with T1 MRI scan in final df_merged: {len(df_merged[df_merged['DCDate_t1'].notna()]['PIDN'].unique())}\n")
        logfile.write(f"Number of Unique PIDNs with Imaging-Core created W-Maps in final df_merged: {len(df_merged[df_merged['wmap_lava'].notna()]['PIDN'].unique())}\n")
    
    # logfile.write(f"\n")
    # # Using df_lava_backup, write the number of unique PIDNs with the base dataset:
    # logfile.write(f"Number of Unique PIDNs in df_lava['{base_dataset}']: {len(df_lava_backup[base_dataset]['PIDN'].unique())}\n")
    # # Number of unique cases in the base dataset after filtering for the specified timepoint:
    # logfile.write(f"Number of Unique Cases in df_lava['{base_dataset}'] after filtering for {timepoint}: {len(df_lava[base_dataset]['PIDN'].unique())}\n")
    
# Create the logfile:
lava_pipeline_logfile(output_dir)

# %%
################################################################################
################################################################################
# ------------------------------- TESTING BELOW ------------------------------ #
################################################################################
################################################################################
# %% TABCAT IMPORT TEST ########################################################
# # IMPORT TABCAT DATA
# df_tabcat = {}
# df_tabcat['TabCATStudyData'] = import_lava_query(lava_queries_dir, 'TabCATStudyData')
# # Filter timepoints by 'fullest':
# df_tabcat['TabCATStudyData'] = filter_timepoints(df_tabcat['TabCATStudyData'], 'fullest')
# # MERGE TABCAT TO DF_MERGED:
# df_merged = merge_lava(df_input, df_tabcat)

# %% DUPE TEST #################################################################
# print("Checking for duplicate HELPERID values in each LAVA dataset...")
# # for each df in df_lava, check if HELPERID column exists - if so check if there are duplicate values within it
# df_dupe = {}
# for df in df_lava:
#     if 'HELPERID' in df_lava[df].columns:
#         if df_lava[df].duplicated(subset=['HELPERID']).any():
#             print('HELPERID column has duplicate values in ' + df)
#             # create a copy of the df appending "_dupe" with only the rows of duplicate HELPERID values 
#             #df_dupe[df + '_dupe'] = df_lava[df][df_lava[df].duplicated(subset=['HELPERID'])].reset_index(drop=True)
#             # create a copy of the df appending "_dupe" with all duplicated HELPERID values
#             df_dupe[df + '_dupe'] = df_lava[df][df_lava[df].duplicated(subset=['HELPERID'], keep=False)].reset_index(drop=True)
#             # print how many unique HELPERIDs there are in the df_dupe
#             print('There are ' + str(len(df_dupe[df + '_dupe']['HELPERID'].unique())) + ' dupe HELPERIDs in ' + df)

# # export each df in df_dupe to a new sheet in the same excel file:
# with pd.ExcelWriter(lava_queries_dir + 'df_dupes.xlsx') as writer:
#     for df in df_dupe:
#         df_dupe[df].to_excel(writer, sheet_name=df, index=False)
# %%
