# %% LAVA Data Filtering Functions by Rian Bogley #############################

# TODO: FIX GROUPBY ERROR:
# # e.g. FutureWarning: Not prepending group keys to the result index of transform-like apply. In the future, the group keys will be included in the index, regardless of whether the applied function returns a like-indexed object.
# To preserve the previous behavior, use

# 	>>> .groupby(..., group_keys=False)

# To adopt the future behavior and silence this warning, use 

# 	>>> .groupby(..., group_keys=True)
#   df = df.groupby('HELPERID').apply(lambda x: x.sort_values('DCDate', ascending=False).ffill(limit=1).bfill(limit=1))

###############################################################################
# %% PACKAGES #################################################################
# Import Packages
import pandas as pd
import numpy as np
###############################################################################
# %% GLOBAL VARIABLES #########################################################
# Set filter variables:
column_filter = ['Version','InstrType','VType','ResearchStatus','QualityIssue','QualityIssue2','QualityIssue3','InstrID','MemImp','ExecImp','LangImp','VisImp','BehImp','MotImp','PsyImp','OthImp','OtherDesc','ProjName','FundingProj','ProjPercent','FundingProj2','Proj2Percent','FundingProj3','Proj3Percent']
dcstatus_filter = ['Complete']
imagelinked_filter = ['NONE LINKED']
# LAVA Diagnoses Simplification Dictionary:
lava_dx_dict = {
    'AD' : ["AD (OLD)","AD possible (PRE-LAVA)","AD probable (PRE-LAVA)","AD probable (PRE-LAVA) (OLD)","AD/VASCULAR MIXED: POSSIBLE (OLD)","AD/VASCULAR MIXED: PROBABLE (OLD)","AD: POSSIBLE (OLD)","AD: PROBABLE (OLD)","ALZHEIMER'S DISEASE","ALZHEIMER'S DISEASE (FRONTAL)","ALZHEIMER'S DISEASE (LANGUAGE)","ALZHEIMER'S DISEASE + ALS","ALZHEIMER'S DISEASE + DLB","ALZHEIMER'S DISEASE + PSP","ALZHEIMER'S DISEASE + VASCULAR DISEASE",],
    'MCI' : ["MCI - AMNESTIC (MULTI DOMAIN)","MCI - AMNESTIC (SINGLE DOMAIN)","MCI - FRONTAL/EXECUTIVE","MCI - LANGUAGE","MCI - MEMORY","MCI - MIXED/UNSPECIFIED","MCI - NONAMNESTIC (MULTI DOMAIN)","MCI - NONAMNESTIC (SINGLE DOMAIN)","MCI - PSYCHIATRIC","MCI - VISUOSPATIAL","MCI (OLD)","MCI (PRE-LAVA)","MCI/CIND: POSSIBLE (OLD)","MCI/CIND: PROBABLE (OLD)","MCI: POSSIBLE (OLD)","MCI: PROBABLE (OLD)",],
    'svPPA' : ["FTD: Semantic Dementia (PRE-LAVA)","SEMANTIC DEMENTIA - PREDOMINANTLY LEFT","SEMANTIC DEMENTIA - UNSPECIFIED","SEMANTIC DEMENTIA + ALS","SEMANTIC VARIANT PPA - PREDOMINANTLY LEFT","SEMANTIC VARIANT PPA - PREDOMINANTLY UNSPECIFED",],
    'sbvFTD' : ["SEMANTIC BEHAVIORAL VARIANT FTD","SEMANTIC DEMENTIA - PREDOMINANTLY RIGHT","SEMANTIC VARIANT PPA - PREDOMINANTLY RIGHT",],
    'lvPPA' : ["LOGOPENIC VARIANT PPA","PROGRESSIVE LOGOPENIC APHASIA SYNDROME",],
    'nfvPPA' : ["NONFLUENT VARIANT PPA","PROGRESSIVE NONFLUENT APHASIA SYNDROME",],
    'bvFTD' : ["BEHAVIORAL VARIANT FTD",],
    'FTD' : ["FRONTOTEMPORAL DEMENTIA (FRONTAL)","FRONTOTEMPORAL DEMENTIA + ALS","FTD (PRE-LAVA)","FTD RIGHT TEMPORAL: POSSIBLE (OLD)","FTD RIGHT TEMPORAL: PROBABLE (OLD)","FTD SD (OLD)","FTD/ALS (PRE-LAVA)","FTD/ALS PROBABLE (OLD)","FTLD FTD: POSSIBLE (OLD)","FTLD FTD: PROBABLE (OLD)","FTLD PA: POSSIBLE (OLD)","FTLD PA: PROBABLE (OLD)","FTLD SD: POSSIBLE (OLD)","FTLD SD: PROBABLE (OLD)",],
    'HC' : ["CLINICALLY NORMAL","CLINICALLY NORMAL (OLD)","Clinically Normal (PRE-LAVA)","FTD Family - Unaffected (PRE-LAVA)","FTD Family - unaffected (PRE-LAVA)","FTD FAMILY UNAFFECTED (OLD)","NORMAL (OLD)","Normal (PRE-LAVA)",],
    'PSP' : ["PROGRESSIVE SUPRANUCLEAR PALSY","Progressive Supranuclear Palsy (PRE-LAVA)","Progressive Supranuclear Palsy (PRE-LAVA) (OLD)","PSP + ALZHEIMER'S DISEASE","PSP/AD MIXED: PROBABLE (OLD)","PSP: PARKINSONISM","PSP: POSSIBLE (OLD)","PSP: PROBABLE (OLD)","PSP: PURE AKINESIA AND GAIT FREEZING","PSP: RICHARDSON'S SYNDROME",],
    'PCA' : ["POSTERIOR CORTICAL ATROPHY SYNDROME",],
    'CBS' : ["CORTICOBASAL SYNDROME",],
    'CBD' : ["CBD (OLD)","CBD: POSSIBLE (OLD)","CBD: PROBABLE (OLD)","CORTICOBASAL DEGENERATION","Corticobasal Degeneration (PRE-LAVA)",],
    'PD' : ["PARKINSON'S (OLD)","PARKINSON'S DISEASE","Parkinson's Disease (PRE-LAVA)","PARKINSON'S DISEASE DEMENTIA","PARKINSON'S DISEASE: POSSIBLE (OLD)","PARKINSON'S DISEASE: PROBABLE (OLD)",],
    'PPAnos' : ["PRIMARY PROGRESSIVE APHASIA - UNSPECIFIED","PROGRESSIVE APHASIA - UNSPECIFIED","Progressive Aphasia (PRE-LAVA)",],
    'DLB' : ["DEMENTIA WITH LEWY BODIES","Diffuse Lewy Body Disease (PRE-LAVA)","DLB (OLD)","DLB + ALZHEIMER'S DISEASE","DLB + VASCULAR DISEASE","DLB/AD MIXED: POSSIBLE (OLD)","DLB/AD MIXED: PROBABLE (OLD)","DLB: POSSIBLE (OLD)","DLB: PROBABLE (OLD)",],
    'ALS' : ["ALS: POSSIBLE (OLD)","ALS: PROBABLE (OLD)","AMYOTROPHIC LATERAL SCLEROSIS",],
    'HD' : ["HUNTINGTON'S DISEASE","Huntington's Disease-pp (PRE-LAVA)","HUNTINGTON'S PP (OLD)","HUNTINGTON'S UU (OLD)","HUNTINGTON'S: POSSIBLE (OLD)","HUNTINGTON'S: PROBABLE (OLD)","Huntington's-uu (PRE-LAVA)",],
    'Other' : ["0 (PRE-LAVA)","ALCOHOL ABUSE (OLD)","ALCOHOL ABUSE/DEPENDENCE","ALCOHOL ABUSE: POSSIBLE (OLD)","ALCOHOL ABUSE: PROBABLE (OLD)","ANXIETY SYMPTOMS","ANXIETY SYMPTOMS: POSSIBLE (OLD)","ANXIETY SYMPTOMS: PROBABLE (OLD)","AUTOIMMUNE DISORDER","AUTOIMMUNE DISORDER (NOT OTHERWISE LISTED)","BIPOLAR DISORDER","BIPOLAR DISORDER (OLD)","BRAIN TUMOR","BRAIN TUMOR: POSSIBLE (OLD)","BRAIN TUMOR: PROBABLE (OLD)","CADASIL","CEREBRAL AMYLOID ANGIOPATHY: POSSIBLE (OLD)","CEREBRAL AMYLOID ANGIOPATHY: PROBABLE (OLD)","CEREBROVASCULAR DISEASE","CHRONIC TRAUMATIC ENCEPHALOPATHY","CJD (PRE-LAVA)","CJD: POSSIBLE (OLD)","CJD: PROBABLE (OLD)","CLINICALLY ASYMPTOMATIC","CLINICALLY NORMAL w/ SYMPTOMS","CNS INFECTION (NOT HIV)","CNS INFECTION: POSSIBLE (OLD)","FTD Family - Affected (PRE-LAVA)","FTD FAMILY AFFECTED (OLD)","HIV ENCEPHALOPATHY","CNS INFECTION: PROBABLE (OLD)","DEPRESSION: PROBABLE (OLD)","DEPRESSIVE SYMPTOMS","DEPRESSIVE SYMPTOMS: POSSIBLE (OLD)","DEPRESSIVE SYMPTOMS: PROBABLE (OLD)","DIABETES: PROBABLE (OLD)","HASHIMOTO'S ENCEPHALOPATHY","HASHIMOTO'S ENCEPHALOPATHY: POSSIBLE (OLD)","HASHIMOTO'S ENCEPHALOPATHY: PROBABLE (OLD)","HEMORRHAGE: POSSIBLE (OLD)","HIPPOCAMPAL SCLEROSIS","HYDROCEPHALUS (NOT NPH)","HYPOXIC INSULT","HYPOXIC INSULT: POSSIBLE (OLD)","HYPOXIC INSULT: PROBABLE (OLD)","INCOMPLETE (OLD)","ISCHEMIC STROKES - SUBCORTICAL (OLD)","ISCHEMIC VASCULAR DISEASE","KNOWN OTHER (Describe in Comments) (OLD)","LACUNES (OLD)","LBD (OLD)","MEDICATION INDUCED","MEDICATION INDUCED: POSSIBLE (OLD)","MEDICATION INDUCED: PROBABLE (OLD)","METABOLIC DISORDER","METABOLIC DISORDER: POSSIBLE (OLD)","METABOLIC DISORDER: PROBABLE (OLD)","MITOCHONDRIAL DISEASE","Mixed Vascular and AD (PRE-LAVA)","MULTIPLE SYSTEM ATROPHY","MULTISYSTEM ATROPHY (OLD)","MULTISYSTEM ATROPHY: POSSIBLE (OLD)","MULTISYSTEM ATROPHY: PROBABLE (OLD)","NON-ISCHEMIC WHITE MATTER DISEASE","NORMAL PRESSURE HYDROCEPHALUS","NPH (OLD)","NPH: POSSIBLE (OLD)","NPH: PROBABLE (OLD)","OTHER (Describe in Comments)","OTHER (Describe in Comments) (OLD)","Other (PRE-LAVA)","PARANEOPLASTIC PROCESS: POSSIBLE (OLD)","PARANEOPLASTIC PROCESS: PROBABLE (OLD)","PARANEOPLASTIC SYNDROME","PENDING (OLD)","PERSONALITY DISORDER: PROBABLE (OLD)","PERSONALITY TRAITS","PERSONALITY TRAITS: POSSIBLE (OLD)","PERSONALITY TRAITS: PROBABLE (OLD)","PRIMARY LATERAL SCLEROSIS","PRION DISEASE (NOT CJD): POSSIBLE (OLD)","PRION DISEASE (NOT CJD): PROBABLE (OLD)","PRION DISEASE/CJD","PROGRESSIVE MUSCULAR ATROPHY","PSYCHOTIC DISORDER","PSYCHOTIC DISORDER: POSSIBLE (OLD)","PSYCHOTIC DISORDER: PROBABLE (OLD)","RAPID DEMENTIA UNKNOWN","RAPID DEMENTIA UNKNOWN: POSSIBLE (OLD)","RAPID DEMENTIA UNKNOWN: PROBABLE (OLD)","RC-abnormal (PRE-LAVA)","RECRUITED CONTROL EXCLUDED","RECRUITED CONTROL EXCLUDED (OLD)","SEIZURE DISORDER","SEIZURES: POSSIBLE (OLD)","SEIZURES: PROBABLE (OLD)","SLEEP APNEA","SLEEP APNEA: POSSIBLE (OLD)","SPINOCEREBELLAR ATAXIA","STROKE: POSSIBLE (OLD)","STROKE: PROBABLE (OLD)","SUBJECTIVE COG IMPAIR","SUBSTANCE ABUSE/DEPENDENCE","SUBSTANCE ABUSE: POSSIBLE (OLD)","SUBSTANCE ABUSE: PROBABLE (OLD)","TOXIN","TOXIN: POSSIBLE (OLD)","TOXIN: PROBABLE (OLD)","TRANSIENT EPILEPTIC AMNESIA","TRANSIENT GLOBAL AMNESIA","TRANSIENT GLOBAL AMNESIA: POSSIBLE (OLD)","TRANSIENT GLOBAL AMNESIA: PROBABLE (OLD)","TRAUMATIC BRAIN INJURY","TRAUMATIC BRAIN INJURY: POSSIBLE (OLD)","TRAUMATIC BRAIN INJURY: PROBABLE (OLD)","Undetermined (PRE-LAVA)","UNKNOWN (OLD)","VASCULAR DEMENTIA (OLD)","Vascular Dementia (PRE-LAVA)","Vascular Dementia (PRE-LAVA) (OLD)","VASCULAR DEMENTIA: POSSIBLE (OLD)","VASCULAR DEMENTIA: PROBABLE (OLD)","VASCULAR DISEASE - DEMENTED: POSSIBLE (OLD)","VASCULAR DISEASE - DEMENTED: PROBABLE (OLD)","VASCULAR DISEASE - NON DEMENTED (OLD)","VASCULAR DISEASE - NON-DEMENTED (OLD)","VASCULAR DISEASE - NON-DEMENTED: POSSIBLE (OLD)","VASCULAR DISEASE - NON-DEMENTED: PROBABLE (OLD)","VASCULAR DISEASE: POSSIBLE (OLD)","VASCULAR DISEASE: PROBABLE (OLD)","VASCULAR MCI: PROBABLE (OLD)","VASCULITIS","WHITE MATTER DISEASE (OLD)","WHITE MATTER DISEASE NON-ISCHEMIC: POSSIBLE (OLD)","WHITE MATTER DISEASE NON-ISCHEMIC: PROBABLE (OLD)",],
}
# LAVA Max Scores for Data Cleaning:
lava_max_scores = {
    'MMSETot':30,
    'MocaTotWithoutEduc':30,
    'Corr10':9,
    'Recog':9,
    'Rey10m':17,
    'BNTCorr':15,
    'PPTP14Cor':14,
    'Verbal':6,
    'Syntax':5,
    'Repeat5':5,
    'WRATTot':70,
    'ReadReg':4,
    'ReadIrr':6,
    'ModRey':17,
    'BryTot':16,
    'NumbLoc':10,
    'Calc':5,
    'Pentgon':1,
    'CATSFMTot':12,
    'DigitFW':9,
    'DigitBW':9,
    'MTCorr':14,
    'MTTime':120,
    'DFCorr':30,
    'StrpCNCor':150,
    'StrpCor':150,
    'CATSAMTot':16,
    }

###############################################################################
# %% FUNCTIONS ################################################################
# Filter NI_ALL:
def filter_ni_all(df,
                 scannerid_filter = ['SFVA 1.5T MRI','NIC 3T MRI','NIC 3T MRI PRISMA','SFVA 4T MRI']):
    """
    Filter Raw LAVA_NI_ALL Data
    """
    # Main Filters:
    # Columns we like: ['PIDN','DCDate','DCStatus','AgeAtDC','ScannerID','SourceID','ScanType','ImageLinked','ImgPath','ImgFormat','ScannerNotes','ImgQuality','QualityNotes']

    # Filter out any columns not in column_filter list:
    for column in column_filter:
        if column in df.columns:
            df = df.drop(columns=column)
    # Filter out any cases without "Complete" visit DCStatus and drop DCStatus column::
    df = df[df['DCStatus'].isin(dcstatus_filter)]
    df = df.drop(columns=['DCStatus'], axis=1)
    # Filter out any cases that aren't in ScannerID filter list:
    df = df[df['ScannerID'].isin(scannerid_filter)]
    # Filter out any cases that are listed as "NONE LINKED" in ImageLinked list:
    df = df[-df['ImageLinked'].isin(imagelinked_filter)]
    # Remove Timestamps from DCDate, leaving only YYYY-MM-DD format and convert NaN to NaT:
    df['DCDate'] = pd.to_datetime(df.DCDate, errors='coerce').dt.date
    # Drop all rows with NaT values in DCDate column:
    df.dropna(subset=['DCDate'], inplace=True)
    # Create HELPERID values and insert as first column (PIDN_SourceID_DCDate):
    df.insert(loc=0 , column = 'HELPERID' , value = (df['PIDN'].map(str) + "_" + df['SourceID'].map(str) + "_" + df['DCDate'].map(str)))
    df
    return (df)
###############################################################################
# Filter, Sort, and Trim for T1 Scans:
def filter_t1(df, t1_filter = ['T1-long','T1-long-3DC','T1-short','T1-short-3DC']):
    """
    Filter Neuroimaging Data to T1 Scans only
    Sort by PIDN (Smallest->Biggest),
    then by DCDate (Oldest->Newest), 
    then by ScanType (A->Z), priotizing: T1-long > T1-long-3DC > T1-short > T1-short-3DC
    then by ImgFormat (Z->A), prioritizing: NifTI > DICOM > Analyze
    Trim duplicates by HELPERID
    Reset index
    """
    # Filter out any cases that aren't in T1 ScanType list:
    df = df[df['ScanType'].isin(t1_filter)]
    # Sort by: PIDN (Smallest->Biggest), then by DCDate (Oldest->Newest), then by ScanType (A->Z), priotizing: T1-long > T1-long-3DC > T1-short > T1-short-3DC, then by ImgFormat (Z->A), prioritizing: NifTI > DICOM > Analyze
    df = df.sort_values(by = ['PIDN','DCDate','ScanType','ImgFormat'], ascending=[True,True,True,False])
    # Trim duplicates by HELPERID
    df = df.drop_duplicates(subset=['HELPERID'],keep='first')
    # Reset index
    df.reset_index(drop=True)
    return (df)
###############################################################################
# Filter Demographics:
def filter_demographics(df):
    """
    Filter Raw LAVA_Demographics Data
    """
    # Main Filters:
    # COLUMNS WE LIKE: ['PIDN','DOB','Deceased','DOD','Hand','Gender','Educ','Onset','VerifiedOnset']
    # If any columns exist in df in column_filter list, drop them:
    for column in column_filter:
        if column in df.columns:
            df = df.drop(columns=column)
    # # Remove Timestamp from DOB, leaving only YYYY-MM-DD format and convert NaN to NaT:
    df['DOB'] = pd.to_datetime(df.DOB, errors='coerce').dt.date
    # Remove Timestamp from DOD, leaving only YYYY-MM-DD format and convert NaN to NaT:
    df['DOD'] = pd.to_datetime(df.DOD, errors='coerce').dt.date
    # Replace any values of 99 in 'Educ' with NaN:
    df['Educ'] = df['Educ'].replace(99, np.nan)


    return (df)
###############################################################################
# Filter LAVA Diagnosis Data:
def filter_diagnosis(df):
    """
    Filter Raw LAVA Diagnosis Data
    """
    # If any columns exist in df in column_filter list, drop them:
    for column in column_filter:
        if column in df.columns:
            df = df.drop(columns=column)
    # Remove Timestamp from DCDate, leaving only YYYY-MM-DD format:
    df['DCDate'] = pd.to_datetime(df.DCDate, errors='coerce').dt.date
    # Drop all rows with NaT values in DCDate column:
    df.dropna(subset=['DCDate'], inplace=True)
    # Replace all negative values in dataset with NaN ignoring strings and datetime64 objects:
    df = df.apply(lambda x: x.mask(x < 0, np.nan) if x.dtype.kind in 'biufc' else x)
    # Create HELPERID values and insert as first column (PIDN_DCDate):
    df.insert(loc=0 , column = 'HELPERID' , value = (df['PIDN'].map(str) + "_" + df['DCDate'].map(str)))
    # Simplify LAVA Diagnosis:
    df = lava_dx_simplify(df, lava_dx_dict)
    return (df)
###############################################################################
# Simplify LAVA Diagnosis (Dx):
def lava_dx_simplify(df, lava_dx_dict):
    """
    Simplify LAVA Diagnosis Data by using the 'lava_dx-dict' dictionary to 
    replace all values in the columns:
    'ClinSynBestEst', 'ClinSynSecEst', and 'ClinSynTerEst'
    with their corresponding values in the dictionary in new columns:
    'dx_1', 'dx_2', and 'dx_3', respectively. 
    """
    # Flip 'lava_dx_dict' such that every value is a key
    # with the original key as the new value:
    lava_dx_dict_flipped = {}
    for key, value in lava_dx_dict.items():
        for i in range(len(value)):
            lava_dx_dict_flipped[value[i]] = key
    # Replace any remaining '-8' values with NaN:
    df = df.replace({'-8': np.nan})
    # Copy original diagnosis columns to new columns and replace values
    # with corresponding values in 'lava_dx_dict_flipped':
    df['dx_1'] = df['ClinSynBestEst'].replace(lava_dx_dict_flipped)
    df['dx_2'] = df['ClinSynSecEst'].replace(lava_dx_dict_flipped)
    df['dx_3'] = df['ClinSynTerEst'].replace(lava_dx_dict_flipped)
    return (df)
###############################################################################
# Filter General LAVA Data:
def filter_lava(df):
    """
    Filter Raw LAVA Data
    This works on most LAVA datasets, but may need to be modified for specific LAVA datasets.
    """
    # If any columns exist in df in column_filter list, drop them:
    for column in column_filter:
        if column in df.columns:
            df = df.drop(columns=column)
    # Remove any rows that aren't in dcstatus_filter list if DCStatus column exists:
    if 'DCStatus' in df.columns:
        df = df[df['DCStatus'].isin(dcstatus_filter)]
    # Remove Timestamp from DCDate, leaving only YYYY-MM-DD format:
    df['DCDate'] = pd.to_datetime(df.DCDate, errors='coerce').dt.date
    # Drop all rows with NaT values in DCDate column:
    df.dropna(subset=['DCDate'], inplace=True)
    # Replace all negative values in dataset with NaN ignoring strings and datetime64 objects:
    df = df.apply(lambda x: x.mask(x < 0, np.nan) if x.dtype.kind in 'biufc' else x)
    # Create HELPERID values and insert as first column (PIDN_DCDate):
    df.insert(loc=0 , column = 'HELPERID' , value = (df['PIDN'].map(str) + "_" + df['DCDate'].map(str)))
    # Using lava_max_scores dictionary, replace any values in the df
    # that are greater than the possible max with NaN:
    for column in df.columns:
        if column in lava_max_scores:
            df[column] = df[column].mask(df[column] > lava_max_scores[column])

    return (df)
###############################################################################
# Filter Other_Cog LAVA Data:
def filter_other_cog(df):
    """
    Filter Raw LAVA OtherCog Data
    """
    # If any columns exist in df in column_filter list, drop them:
    for column in column_filter:
        if column in df.columns:
            df = df.drop(columns=column)
    # Remove Timestamp from DCDate, leaving only YYYY-MM-DD format:
    df['DCDate'] = pd.to_datetime(df.DCDate, errors='coerce').dt.date
    # Drop all rows with NaT values in DCDate column:
    df.dropna(subset=['DCDate'], inplace=True)
    # Replace all negative values in dataset with NaN ignoring strings and datetime64 objects:
    df = df.apply(lambda x: x.mask(x < 0, np.nan) if x.dtype.kind in 'biufc' else x)
    # Create HELPERID values and insert as first column (PIDN_DCDate):
    df.insert(loc=0 , column = 'HELPERID' , value = (df['PIDN'].map(str) + "_" + df['DCDate'].map(str)))
    # Using lava_max_scores dictionary, replace any values in the df
    # that are greater than the possible max with NaN:
    for column in df.columns:
        if column in lava_max_scores:
            df[column] = df[column].mask(df[column] > lava_max_scores[column])
    # Merge rows in df with the same HELPERID values and closest DCDate values within 30 days, prioritizing cells with data over cells with NaN values:
    df = df.groupby('HELPERID').apply(lambda x: x.sort_values('DCDate', ascending=False).ffill(limit=1).bfill(limit=1))
    return (df)
###############################################################################
# Filter Timepoints:
def filter_timepoints(df, timepoint='first'):
    """
    Filter DCDate Timepoints based on timepoint argument:

    Timepoint options:
    'first' = Earliest valid DCDate for each PIDN
    'latest' = Latest valid DCDate for each PIDN
    'fullest' = Fullest dataset for each PIDN
    """
    # Sort by DCDate (Oldest->Newest):
    df = df.sort_values(by = ['PIDN','DCDate'], ascending=[True,True])
    # If timepoint is "first", filter for earliest DCDate for each PIDN:
    if timepoint == 'first':
        df = df.groupby('PIDN').first().reset_index()
        print('First Timepoint Filtered')
    # If timepoint is "latest", filter df for latest DCDate for each PIDN:
    elif timepoint == 'latest':
        df = df.groupby('PIDN').last().reset_index()
        print('Latest Timepoint Filtered')
    # If timepoint is "fullest", filter df for max count for each PIDN:
    elif timepoint == 'fullest':
        # Count how many values there are per row:
        df = df
        df['count'] = df.count(axis=1)
        # Sort by max count:
        df = df.sort_values(by = ['PIDN','count'], ascending=[True,False])
        # Filter for max count:
        df = df.groupby('PIDN').first().reset_index()
        # Remove count column:
        df = df.drop(['count'], axis=1)
        print('Fullest Timepoint Filtered')
    else:
        print('Invalid Timepoint Selection')
    return (df)
###############################################################################