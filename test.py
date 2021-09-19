import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Title for the App###########################################
st.title("Simplified Approach for ECL Estimation")

# Loading Data
data_load_state = st.text('Loading Provision Matrix')
data = pd.read_csv("./data/test_data.csv")
data_load_state = st.text('Finished Loading Data')



# Inspecting the Data #########################################
st.subheader("Exposure Data")
st.write(data)

# GDP Percentage Data
# # Preprocessing macro data and calculating Z-scores
# ## Cleaning up 'Year' in macro_data
st.subheader("GDP Data")
GDP_data = pd.read_csv("./data/macro_data.csv", header = None)
@st.cache(allow_output_mutation = True)
def macro_preprocess(macro_data):
    macro_data = macro_data.T
    new_header = macro_data.iloc[0]
    macro_data = macro_data[1:]
    macro_data.columns = new_header
    macro_data_drop_year = macro_data.drop(columns = 'Year')
    cond = macro_data_drop_year != 'no data'

    # Function to take care of 'no data' entries in all columns
    def preprocessing(df):
        tmp = df.replace(to_replace = 'no data', value = np.nan)
        tmp = tmp.apply(pd.to_numeric)
        return tmp

    macro_data_drop_year = macro_data_drop_year.apply(preprocessing)

    # Preprocessing 'Year' column separately
    # tmp = map(lambda x:float(x), macro_data['Year']) # Converting Year to int
    # macro_data['Year'] = str(tmp)

    # Preprocesse GDP Dataframe
    tmp = macro_data_drop_year
    tmp_df = tmp.copy()
    tmp['Year'] = macro_data['Year'].apply(int)
    macro = tmp.copy()
    macro = macro.reset_index(drop = True)

    return (macro)

GDP = macro_preprocess(GDP_data)
col = GDP_data[0]
GDP = GDP[col]
GDP.to_csv("./data/gdp.csv")
gdp_data = pd.read_csv("./data/gdp.csv", header = [0], index_col=[0])
st.dataframe(gdp_data)


# Z- scores ########################################################
st.subheader("Converting GDP data to Z-scores for the Vasicek Model by normalizing")
tmp_df = gdp_data.drop(columns = 'Year') 
tmp_1 = tmp_df.to_numpy(dtype = float)

z_score_mean = np.nanmean(tmp_1, axis = 0)
z_score_std = np.nanstd(tmp_1, axis = 0)

# Calculating Avg and Std of macro data accoring to each
# country for the entire time period
df_stat = pd.DataFrame([z_score_mean, z_score_std], columns = col[1:])
df_stat.index = ['Average', 'Std Dev']

country_wise_Z_scores = pd.DataFrame()
i = 0

while i < (tmp_df.shape[0]):
    country_wise_Z_scores[i] = (pd.array(tmp_df.iloc[i], float) - df_stat.iloc[0]) / df_stat.iloc[1]
    i = i + 1

country_wise_Z_scores = country_wise_Z_scores.T
Z_scores = country_wise_Z_scores.copy()
Z_scores.insert(0, "Year", np.array(gdp_data["Year"]))
Z_scores = Z_scores.set_index('Year')
Z_scores.to_csv("./data/z_scores.csv")
z_scores = pd.read_csv("./data/z_scores.csv", header = [0])

st.write(z_scores)


# TTC PD, PiT PD & Asset Correlation###########################################
# FLow Rates
label_tmp = data.columns[0] # Extracting first labeel ('Period) of Exposuree Data
exp_df = (data.copy()).drop(labels = label_tmp, axis = 1).to_numpy()
nrow = exp_df.shape[0]
ncol = exp_df.shape[1]

i = 0
flow_rate_tmp = np.empty([(nrow - 1), (ncol - 1)])

while i < nrow - 2: 
    flow_rate_tmp[i][0:] = exp_df[i+1][1:(ncol)] / exp_df[i][0:(ncol - 1)]
    i = i + 1

flow_rate = pd.DataFrame(flow_rate_tmp, columns = data.columns[1:ncol])

flow_rate = flow_rate.where(cond = flow_rate < 1, other = 1)
flow_rate = flow_rate.where(cond = flow_rate > 0, other = np.nan)

st.subheader("Resultant Flow Rates")
st.write(flow_rate)

# TTC an PiT PD
TTC_PD = np.nanmean(flow_rate, axis = 0)
TTC_PD = np.append(TTC_PD, 1)
i = 0
while i < len(TTC_PD):
    TTC_PD[i] = np.product(TTC_PD[i:])
    i = i + 1

# Function to calcualte the PiT PD for 3 scenarios: Base, Upturn and Downturn
# using Vasicek model.
    
def PiT_PD(ttc_pd, z_scores, country, yr):
    rho = 0.24 - (0.12 * ((1-np.exp(-50 * ttc_pd)) / (1 - np.exp(-50)))) # Asset Correlation
    m_base = pd.array(z_scores.filter(like = country, axis = 1).loc[yr]) # macro variable
    m_up = pd.array(z_scores.filter(like = country, axis = 1).loc[yr]) + df_stat[country][1]
    m_down = pd.array(z_scores.filter(like = country, axis = 1).loc[yr]) - df_stat[country][1] 
    p = norm.ppf(ttc_pd, loc = 0, scale = 1) # quantile of TTC PD
    base_pit = norm.cdf((p - np.multiply(np.sqrt(rho), m_base) / np.sqrt( 1 - rho)), loc = 0, scale = 1)
    up_pit = norm.cdf((p - np.multiply(np.sqrt(rho), m_up) / np.sqrt( 1 - rho)), loc = 0, scale = 1)
    down_pit = norm.cdf((p - np.multiply(np.sqrt(rho), m_down) / np.sqrt( 1 - rho)), loc = 0, scale = 1)
    pit_pd = pd.DataFrame([base_pit, up_pit, down_pit], index = ["Base", "Upturn", "Downturn"])
    return(pit_pd.T, rho)   