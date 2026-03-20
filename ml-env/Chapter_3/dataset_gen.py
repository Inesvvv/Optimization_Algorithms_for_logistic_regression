"""
File created 19/03/2026
This file aims to extract the data on marriage matching and create the features. 
SISTA with newton method is applied in files "personality_traits_testing.py" and "Choo_Siow_age_testing"
to determine the relevant features/parameters in marriage matching

Data sources:
  - Choo & Siow (2006 JPE), via Galichon's mec_optim: aggregate marriage counts by age group (60x60)
  - Galichon & Dupuy (2014 JPE) "Personality traits and the marriage market": individual-level Big Five + demographics
  - Chiappori, Oreffice, Quintana-Domeque (2012 JPE) "Fatter Attraction": individual-level anthropometric + socioeconomic
  
  https://github.com/math-econ-code/mec_optim
  https://github.com/TraME-Project/TraME-Datasets
"""

import pandas as pd
import numpy as np

# =============================================================================
# 1. Choo-Siow aggregate data (age-group level, 60 groups per side)
# =============================================================================
choo_siow_path = 'https://raw.githubusercontent.com/math-econ-code/mec_optim_2021-01/master/data_mec_optim/marriage-ChooSiow/'

n_singles = pd.read_csv(choo_siow_path + 'n_singles.txt', sep='\t', header=None).values
marr      = pd.read_csv(choo_siow_path + 'marr.txt', sep='\t', header=None).values
n_avail   = pd.read_csv(choo_siow_path + 'n_avail.txt', sep='\t', header=None).values

ages = np.arange(16, 76)
nTypes = len(ages)

p_choo = n_avail[:, 0].astype(float)
q_choo = n_avail[:, 1].astype(float)
hat_pi_choo = marr.astype(float)

# Choo Siow dataset info 
print("=== Choo-Siow aggregate data ===")
print(f"  Types (age groups): {nTypes}")
print(f"  hat_pi shape: {hat_pi_choo.shape}")
print(f"  Total marriages: {hat_pi_choo.sum():,.0f}")

# =============================================================================
# 2. Personality Traits data (Galichon & Dupuy 2014)
#    Each row = one married couple with both partners' characteristics
#    m = husband, v = wife 
#    Big Five: conscientiousness, extraversion, agreeableness, emotional stability, autonomy + risk
# =============================================================================
trame_path = 'https://raw.githubusercontent.com/TraME-Project/TraME-Datasets/master/'

personality = pd.read_csv(trame_path + 'personality_traits/raw/data.csv', sep=';')
personality.columns = personality.columns.str.strip()

husband_cols = [c for c in personality.columns if c.endswith('m')]
wife_cols    = [c for c in personality.columns if c.endswith('v')]

print("\n=== Personality Traits (Galichon & Dupuy 2014) ===")
print(f"  Couples: {len(personality)}")
print(f"  Husband features: {husband_cols}")
print(f"  Wife features:    {wife_cols}")

# =============================================================================
# 3. Fatter Attraction data (Chiappori et al. 2012)
#    Individual-level: matched husbands and wives with anthropometric data
# =============================================================================
fa_female = pd.read_csv(trame_path + 'fatter_attraction/formatted/female.csv')
fa_male   = pd.read_csv(trame_path + 'fatter_attraction/formatted/male.csv')

fatter = fa_male.merge(fa_female, on=['pid', 'pid_spouse'], suffixes=('', '_dup'))
fatter = fatter[[c for c in fatter.columns if not c.endswith('_dup')]]

print("\n=== Fatter Attraction (Chiappori et al. 2012) ===")
print(f"  Matched couples: {len(fatter)}")
print(f"  Columns: {list(fatter.columns)}")

# =============================================================================
# 4. Build basis functions D (K x N x N) for Choo-Siow age-group model
#    Each D[k, i, j] measures how feature k interacts between man-type i 
#    and woman-type j. SISTA will select which features matter.
# =============================================================================
age_m = ages[:, None] * np.ones((1, nTypes))
age_w = np.ones((nTypes, 1)) * ages[None, :]

basis_functions = {
    'age_diff':        age_m - age_w, # do men tend to be older than their wives?
    'age_diff_sq':     (age_m - age_w)**2, # is large age gap penalized quadratically? 
    'abs_age_diff':    np.abs(age_m - age_w),  # is large age gap penalized lineearly? 
    'age_sum':         age_m + age_w, 
    'age_product':     age_m * age_w,
    'same_age':        (age_m == age_w).astype(float), # bonus for having same age ? 
    'man_older_5plus': (age_m - age_w >= 5).astype(float), # bonus for men being more than 5 years older?
    'close_age_3':     (np.abs(age_m - age_w) <= 3).astype(float), # bonus for having roughly the same age? 
}

D_choo = np.array(list(basis_functions.values()))
basis_names = list(basis_functions.keys())

for k in range(D_choo.shape[0]):
    s = D_choo[k].std()
    if s > 0:
        D_choo[k] /= s

print(f"\n=== Basis functions for Choo-Siow ===")
print(f"  D shape: {D_choo.shape}  (K={D_choo.shape[0]} features, {nTypes}x{nTypes} types)")
print(f"  Features: {basis_names}")

# =============================================================================
# 5. Build basis functions from Personality Traits data
#    Discretize individuals into types by education level, then construct
#    matching matrix and feature interactions.
# =============================================================================
personality['educ_bin_m'] = pd.Categorical(personality['educm']).codes
personality['educ_bin_v'] = pd.Categorical(personality['educv']).codes

n_types_m = personality['educ_bin_m'].nunique()
n_types_w = personality['educ_bin_v'].nunique()

hat_pi_pers = np.zeros((n_types_m, n_types_w))
for _, row in personality.iterrows():
    hat_pi_pers[int(row['educ_bin_m']), int(row['educ_bin_v'])] += 1

p_pers = hat_pi_pers.sum(axis=1)
q_pers = hat_pi_pers.sum(axis=0)

continuous_features_m = ['heightm', 'BMIm', 'consm', 'extram', 'agreem', 'emom', 'autom', 'riskym']
continuous_features_v = ['heightv', 'BMIv', 'consv', 'extrav', 'agreev', 'emov', 'autov', 'riskyv']

mean_m = personality.groupby('educ_bin_m')[continuous_features_m].mean()
mean_v = personality.groupby('educ_bin_v')[continuous_features_v].mean()

D_pers_list = []
pers_basis_names = []
for fm, fv in zip(continuous_features_m, continuous_features_v):
    feat_name = fm.replace('m', '')
    xm = mean_m[fm].values[:, None]
    xv = mean_v[fv].values[None, :]

    interaction = xm * xv
    s = interaction.std()
    if s > 0:
        interaction /= s
    D_pers_list.append(interaction)
    pers_basis_names.append(f'{feat_name}_interact')

    diff = (xm - xv)**2
    s = diff.std()
    if s > 0:
        diff /= s
    D_pers_list.append(diff)
    pers_basis_names.append(f'{feat_name}_diff_sq')

D_pers = np.array(D_pers_list)

print(f"\n=== Personality Traits: discretized matching model ===")
print(f"  Types: {n_types_m} x {n_types_w} (education bins)")
print(f"  hat_pi shape: {hat_pi_pers.shape}")
print(f"  D shape: {D_pers.shape}  (K={D_pers.shape[0]} features)")
print(f"  Features: {pers_basis_names}")

# Dataset size info:
#print(f"Choo-Siow (age groups):   p({nTypes},), q({nTypes},), hat_pi({nTypes},{nTypes}), D{D_choo.shape}")
#print(f" Personality (educ bins):   p({n_types_m},), q({n_types_w},), hat_pi{hat_pi_pers.shape}, D{D_pers.shape}")
