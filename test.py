import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Title for the App###########################################
st.title("Simplified Approach for ECL Estimation")

# Loading Data
# data_load_state = st.text('Loading Provision Matrix')
data = pd.read_csv("./data/test_data.csv")
# data_load_state = st.text('Finished Loading Data')

data_up = st.sidebar.file_uploader("Choose a file")
if data_up is not None:
  df = pd.read_csv(data_up)
  #st.write(df)


# Inspecting the Data #########################################
st.subheader("Exposure Data")
st.write(df)

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
col = GDP_data[0] # Columns Names
GDP = GDP[col] # Ading in the Year Column to the dataframe
GDP[::-1].to_csv("./data/gdp.csv")
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


# LGD Calculation ###########################################################
a = exp_df[0:-1, -1]
b = exp_df[0:-1, -2]
c = exp_df[1:, -1]

LGD = pd.DataFrame((a + b - c)/(a + b))
LGD = LGD.where(cond = LGD < 1, other = 1)
LGD = LGD.where(cond = LGD > 0, other = np.nan)
LGD = np.nanmean(LGD)


# ECL Calculation ###########################################################
@st.cache
def ECL(exp, LGD, ttc_pd, z_scores, country, yr, w1 = 0.33, w2 = 0.33, w3 = 0.33 ):
    PD = PiT_PD(ttc_pd, z_scores, country, yr)
    final_exp = np.round(exp[-1], 2)
    tmp_1 = (PD[0].to_numpy() * LGD) # PD * LGD
    ECL_scenarios = np.round(np.einsum("ij, i -> ij", tmp_1, final_exp), 2)
    ECL_final = np.average(ECL_scenarios, weights = [w1, w2, w3], axis = 1)
    return(ECL_final, PD[0], PD[1])


# Requiredd arguments are
# exp, LGD, w1, w2, w3, ttc_pd, z_scores, country, yr
# 'Afghanistan', 'Bahrain', 'Canada', 'Egypt', 'India', 'Oman',
#       'Qatar', 'Saudi Arabia', 'United Arab Emirates', 'United Kingdom',
#       'United States'

# Year Select Box in Sidebar
year_selecttbox = st.sidebar.selectbox(
    "Year",
    reversed((gdp_data['Year']))
)

# Country Select Box in Sidebar
country_selectbox = st.sidebar.selectbox(
    "Economy",
    (col[1:])
)

st.sidebar.subheader("Weights")
w1 = st.sidebar.slider(label = 'Base Weight', value = 50)
w2 = st.sidebar.slider(label = 'Best Weight', value = 20)
w3 = st.sidebar.slider(label = 'Worst Weight', value = 100 - w1 - w2)
colw1, colw2, colw3 = st.columns(3)


ECL_tmp = ECL(exp_df, LGD, TTC_PD, Z_scores, country_selectbox, year_selecttbox, w1, w2, w3)
ECL_final = round(pd.DataFrame(ECL_tmp[0], columns = ['Final ECL']), 2)
ECL_PD = ECL_tmp[1] 

ECL_per = round(np.nansum(ECL_tmp[0]) / np.nansum(exp_df[-1]), 2)

final_df = pd.DataFrame([exp_df[-1],TTC_PD, ECL_tmp[2]], index = ['Final Exposure', 'TTC PD', 'Asset Correlation']).T
final_df = pd.concat([final_df, ECL_PD, ECL_final], axis = 1)
final_df = final_df.set_index(data.columns[1:])

st.subheader('ECL For chosen Economy')
st.write(final_df)


# Summary Table for Results #######################################################
final_ecl = round(np.sum(final_df['Final ECL']), 2)
final_exp = round(np.sum(final_df['Final Exposure']), 2)
tmp_array = np.array([final_exp, final_ecl, ECL_per])

ECL_summary = pd.Series(tmp_array,
                            index = ['Final Exposure', 'Final ECL', 'ECL(%)'],
                            name = 'Result')
st.sidebar.subheader('ECL Summary')
#col1, col2, col3 = st.columns(3)

st.sidebar.metric("Final Exposure", ECL_summary[0])

st.sidebar.metric("Final ECL", ECL_summary[1])

st.sidebar.metric("ECL(%)", ECL_summary[2])
#st.dataframe(ECL_summary)



# Plotting (bcos I felt lkke it) ################################################
final_df_PD = final_df.drop(['Final Exposure', 'Asset Correlation', 'Final ECL'], axis = 1)
final_df_PD = (final_df_PD.drop(index = ['> 360 days']))
final_df_PD = final_df_PD.reset_index(drop = True)

st.subheader("PD Plots")
st.line_chart(final_df_PD)
