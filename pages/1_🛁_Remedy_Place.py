import streamlit as st
import pymysql
import paramiko
from paramiko import SSHClient
from sshtunnel import SSHTunnelForwarder
import os
from os.path import expanduser
from dotenv import load_dotenv
load_dotenv('.env')  # take environment variables from .env.
import datetime
import RP_funcs as RPf
# import take5_functions as RPf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html
import dash_ag_grid as dag 
from st_aggrid import AgGrid
# from helper import *
from sklearn.linear_model import LinearRegression
import hmac


st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_icon=":shark:")

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-color: #584c42;
opacity: 1.0;
background-image: radial-gradient(#fff4e9 0.2px, #584c42 0.2px);
background-size: 4px 4px;
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# st.set_page_config()

# get secrets from st.secrets
host = os.getenv('host_kdb') 
user = os.getenv('user_kdb')
port = int(os.getenv('port_kdb'))
password = os.getenv('password_kdb')
databasename = os.getenv('databasename_kdb')
# host = st.secrets["host_kdb"]#os.getenv('host_kbd') 
# user = st.secrets["user_kdb"]#os.getenv('user_kbd')
# port = st.secrets["port_kdb"]#os.getenv('port_kbd')
# password = st.secrets["password_kdb"]#os.getenv('password_kbd')
# databasename = st.secrets["databasename_kdb"]#os.getenv('databasename_kbd')
host_ssh = os.getenv('host_ssh') 
user_ssh = os.getenv('user_ssh')
port_ssh = int(os.getenv('port_ssh'))
#### read in data

home = os.path.expanduser('~')
mypkey = paramiko.RSAKey.from_private_key_file('G:/My Drive/kordis/db/feature-server-key')
connection, tunnel = RPf.make_conn_ssh(host_ssh,user_ssh,port_ssh,mypkey,host,user,port,password,databasename)

st.write('hi- connection worked!')

query = '''
SELECT 
    t.amount,
    t.transaction_start_date AS date,
    o.title AS organization,
    g.name AS category,
    c.name AS location
FROM 
    general_transactions t
JOIN 
    organizations o ON t.organization_id = o.id
JOIN
    general_categories g ON t.general_category_id = g.id
JOIN
    general_classes c ON t.general_class_id = c.id
WHERE
    t.organization_id = 3
    AND t.transaction_start_date BETWEEN '2020-05-01' AND '2024-06-30';
'''

# read in data and close connection
df = RPf.read_in_SQL(connection, query)
connection.close()
tunnel.stop()

# print dataframe
# st.dataframe(df.head(20))

# Rename the column
if df is not None:
    df['monthyear'] = df['date'].dt.strftime('%b %y')  # Reformat to 'Mon YY' format
    df['MY'] = df['date'].dt.to_period('M').dt.to_timestamp()


#### get data within dates. Baseline is 13 months.
maxdate = max(df.date)
mindate = maxdate - pd.DateOffset(months=12)

with st.sidebar:
    options_loc = st.multiselect('Select the Remedy Place Locations you want to perform analsysis on:', 
                            df.location.unique(), df.location.unique())
    options_cat = st.multiselect('Select the Remedy Place Categories you want to perform analsysis on:', 
                            df.category.unique(), df.category.unique())
    startdate = st.date_input("Please enter a starting date:", mindate.date())
    enddate = st.date_input("Please enter a ending date:", maxdate.date())
startdate = pd.to_datetime(startdate)
enddate = pd.to_datetime(enddate)

# # trim data based on selected/standard dates
df_new = df[(df['date'] >= startdate) & (df['date'] <= enddate) & 
        (df.category.isin(options_cat)) & (df.location.isin(options_loc))]

# st.write(df_new)

pivot_df = RPf.create_RP_pivot_table(df_new)

st.dataframe(pivot_df)

df_rev = df_new.groupby(['location','MY'])['amount'].sum()#.reset_index()
# st.dataframe(df_rev)

tot_rev_by_date = df_rev.reset_index().groupby('MY')['amount'].sum().reset_index()
# st.dataframe(tot_rev_by_date)

df_date_cat = df_new.reset_index().groupby(['location','category','MY'])['amount'].sum().reset_index()
# st.write('by date and category:')
# st.dataframe(df_date_cat)

tot_date_cat = df_date_cat.reset_index().groupby(['category','MY'])['amount'].sum().reset_index()


c1, c2 = st.columns(2)
with c1:
    # st.header("Overall Revenue Trendline")
    fig1 = px.bar(df_rev.reset_index(), x='MY', y='amount', color='location', title="Revenue by Location")
    fig1.add_scatter(x=tot_rev_by_date['MY'], y=tot_rev_by_date['amount'], mode='lines+markers', name='Total')
    st.plotly_chart(fig1)

    ind = df_date_cat.location == 'West Hollywood'
    fig3 = px.scatter(df_date_cat[ind], x='MY', y='amount', color='category', title='Revenue by Category (West Hollywood)')
    fig3.update_traces(mode='lines+markers')
    st.plotly_chart(fig3)

with c2:
    fig2 = px.bar(tot_date_cat, x='MY', y='amount', color='category', title="Revenue by Category (total)")
    # fig2.add_scatter(x=tot_cars_by_date['Date'], y=tot_cars_by_date['value'], mode='lines+markers', name='Total')
    st.plotly_chart(fig2)

    ind = df_date_cat.location == 'Flatiron'
    fig4 = px.scatter(df_date_cat[ind], x='MY', y='amount', color='category', title='Revenue by Category (Flatiron)')
    fig4.update_traces(mode='lines+markers')
    st.plotly_chart(fig4)

# ext_melt = pd.melt(extra, 
#                      id_vars=['Location', 'location', 'Date'],   # others: Pmix_perc	Big5_perc	BayTimes
#                      var_name='metric', 
#                      value_name='value').dropna(subset=['value'])
# ext_melt = ext_melt[(ext_melt.location.isin(options))]

# ext_avg = ext_melt[(ext_melt['Date'] >= startdate) & 
#                    (ext_melt['Date'] <= enddate) & 
#                    (ext_melt.metric.isin(['BayTimes','Pmix_perc','Big5_perc']))]
# ext_sum = ext_melt[(ext_melt['Date'] >= startdate) & 
#                    (ext_melt['Date'] <= enddate) & 
#                    (ext_melt.metric.isin(['CarsServ','EmpHours']))]


# ###### crate dataframes for figures
# #### create revenue by location dataframe
# ind = (df_new.Account_Num >4000) & (df_new.Account_Num <4999)
# df_rev = df_new[ind].groupby(['location','monthdt'])['value'].sum()#.reset_index()
# tot_rev_by_date = df_rev.reset_index().groupby('monthdt')['value'].sum().reset_index()




#     # st.header("Gross Profit Trendline")
#     fig3 = px.bar(df_gross.reset_index(), x='monthdt', y='value', color='location', title="Gross Profit by Location")
#     fig3.add_scatter(x=tot_gross_by_date['monthdt'], y=tot_gross_by_date['value'], mode='lines+markers', name='Total')
#     st.plotly_chart(fig3)

#     # st.header("Cash Trendline")
#     fig5 = px.bar(df_cash.reset_index(), x='monthdt', y='value', color='location', title="Gross Profit by Location")
#     fig5.add_scatter(x=tot_cash_by_date['monthdt'], y=tot_cash_by_date['value'], mode='lines+markers', name='Total')
#     st.plotly_chart(fig5)



#     # st.header("4-Wall EBITDA Trendline")
#     fig4 = px.bar(df_4webitda.reset_index(), x='monthdt', y='value', color='location', title="4-Wall EBITDA by Location")
#     fig4.add_scatter(x=tot_ebitda_by_date['monthdt'], y=tot_ebitda_by_date['value'], mode='lines+markers', name='Total')
#     st.plotly_chart(fig4)

#     # st.header("4-Wall EBITDA per car Trendline")
#     fig6 = px.bar(df_ebitda_by_car.reset_index(), x='monthdt', y='value', color='location', title="4-Wall EBITDA by Car by Location")
#     fig6.add_scatter(x=tot_ebitdacar_by_date['monthdt'], y=tot_ebitdacar_by_date['value'], mode='lines+markers', name='Total')
#     st.plotly_chart(fig6)



# ####### Create BOXES Showing comparison for previous month and vs. budget for ARO, CPD, LHPC,.... 
# box_height = 140

# row0 = st.columns(6)
# row1 = st.columns(6)

# # ARO	CPD  	LHPC	P-Mix %	  Big 5 %	Bay Times
# ind = [( 2, 'ARO'),( 1, 'CPD'),(51, 'LHPC'),(61, 'P-Mix %'),(62, 'Big 5 %'),(63, 'Bay Times')]

# last2mos = pivot_df.iloc[:,-5:-3].loc[ind,:]
# last2mos['diffs'] = last2mos.iloc[:,1].sub(last2mos.iloc[:,0], axis = 0) 
# last2mos['diffperc'] = last2mos['diffs'] / last2mos.iloc[:,0]
# last2mos = last2mos.reset_index().drop(columns=['Account_Num', 'Account'])
# last2mos.index = pd.RangeIndex(start=0, stop=len(last2mos), step=1)

# formatting = [
#     (0, dollar_form),
#     (1, format_two_decimals),
#     (2, format_two_decimals),
#     (3, pmix_form),
#     (4, big5_form),
#     (5, baytime_form),
# ]
# for index, func in formatting:
#     last2mos.iloc[index, 1] = func(last2mos.iloc[index, 1])
#     # last2mos.loc[index, 'values'] = last2mos[index][2]
# cnt = 0
# for col in row0:
#     tile = col.container()#height=60)
#     tile.write(ind[cnt][1])
#     cnt += 1

# cnt = 0
# for col in row1:
#     tile = col.container(height=box_height)
#     if cnt in [2, 5]:
#         tile.write(last2mos.iloc[cnt,1] + arrow_form_perc_opp(last2mos.iloc[cnt]['diffperc']))
#     else: 
#         tile.write(last2mos.iloc[cnt,1] + arrow_form_perc(last2mos.iloc[cnt]['diffperc']))
#     tile.write("All")
#     tile.write('(budget #s)')
#     cnt += 1

# ind = (df['monthdt'] >= enddate - pd.DateOffset(months=1)) & (df['monthdt'] <= enddate)
# df2months = df[ind]

# # calc ARO dataframe by location
# aro_df = df_rev / ext_cars_loc
# aro_df = aro_df.reset_index()
# ind = (aro_df['monthdt'] >= enddate - pd.DateOffset(months=1)) & (aro_df['monthdt'] <= enddate)
# aro_df = aro_df[ind].groupby(['location','monthdt'])['value'].mean().reset_index()
# aro_df = aro_df.pivot_table(index=['location'], columns='monthdt', values='value', aggfunc='mean')
# aro_df.columns = aro_df.columns.strftime('%b %y')
# aro_df['diffs'] = aro_df.iloc[:,1].sub(aro_df.iloc[:,0], axis = 0)
# aro_df['diffperc'] = aro_df['diffs'] / aro_df.iloc[:,0]

# ### calculate CPD and LHPC for last month
# ind = (ext_sum['Date'] >= enddate - pd.DateOffset(months=1)) & (ext_sum['Date'] <= enddate)
# ext2_sum  = ext_sum.loc[ind,:]
# ext2_sum = ext2_sum.merge(workdays, left_on='Date', right_on='date')
# # get number of stores that are serving cars by month
# ind = (ext2_sum.metric == 'CarsServ')  
# n_stores_df = ext2_sum.loc[ind,:].Date.value_counts().reset_index()  # get number of stores open by month
# ext2_sum = ext2_sum.merge(n_stores_df, left_on='Date', right_on='Date')
# ext2_sum = ext2_sum.pivot_table(index=['location','Date','workdays','count'], columns = ['metric'], values='value', aggfunc='mean').reset_index()
# ext2_sum['CPD'] = (ext2_sum.CarsServ / ext2_sum.workdays) #/ ext2_sum['count']
# ext2_sum['LHPC'] = ext2_sum.EmpHours / ext2_sum.CarsServ 

# # st.write(ext2_sum) 
# ### CPD df
# cpd_df = ext2_sum.pivot_table(index=['location'], columns='Date', values='CPD', aggfunc='mean')#.reset_index()
# cpd_df.columns = cpd_df.columns.strftime('%b %y')
# cpd_df['diffs'] = cpd_df.iloc[:,1].sub(cpd_df.iloc[:,0], axis = 0) 
# cpd_df['diffperc'] = cpd_df['diffs'] / cpd_df.iloc[:,0]
# ### LHPC df
# lhpc_df = ext2_sum.pivot_table(index=['location'], columns='Date', values='LHPC', aggfunc='mean')#.reset_index()
# lhpc_df.columns = lhpc_df.columns.strftime('%b %y')
# lhpc_df['diffs'] = lhpc_df.iloc[:,1].sub(lhpc_df.iloc[:,0], axis = 0) 
# lhpc_df['diffperc'] = lhpc_df['diffs'] / lhpc_df.iloc[:,0]

# ## get Pmix, Big5%, Bay Times
# ind = (ext_avg['Date'] >= enddate - pd.DateOffset(months=1)) & (ext_avg['Date'] <= enddate)

# ext2_avg = ext_avg.loc[ind,:]
# # st.write("ext2:", ext2_avg)
# ext2_avg = ext2_avg.groupby(['location','metric','Date'])['value'].mean().reset_index()
# ext2_avg = ext2_avg.pivot_table(index=['location','metric'], columns='Date', values='value', aggfunc='mean')#.reset_index()
# ext2_avg.columns = ext2_avg.columns.strftime('%b %y')
# ext2_avg['diffs'] = ext2_avg.iloc[:,1].sub(ext2_avg.iloc[:,0], axis = 0) 
# ext2_avg['diffperc'] = ext2_avg['diffs'] / ext2_avg.iloc[:,0]

# # Sort by the second column
# sec_col = ext2_avg.columns[1]  # get the most recent month
# ext2_avg = ext2_avg.reset_index()
# aro_df = aro_df.sort_values(by=sec_col, ascending=False).reset_index()
# cpd_df = cpd_df.sort_values(by=sec_col, ascending=False).reset_index()
# lhpc_df = lhpc_df.sort_values(by=sec_col, ascending=True).reset_index()

# pmix_df = ext2_avg.loc[ext2_avg.metric == 'Pmix_perc'].sort_values(by=sec_col, ascending=False).reset_index()
# big5_df = ext2_avg.loc[ext2_avg.metric == 'Big5_perc'].sort_values(by=sec_col, ascending=False).reset_index()
# baytime_df = ext2_avg.loc[ext2_avg.metric == 'BayTimes'].sort_values(by=sec_col, ascending=True).reset_index()


# # Initialize a dictionary to store the containers
# grid = {}
# # # Define the number of rows and columns
# num_rows = len(options)
# num_cols = 6

# # # Create the grid of containers
# for row in range(num_rows):
#     grid[row] = st.columns(num_cols)
# #### ARO
# for row in range(num_rows):
#     tile = grid[row][0].container(height=box_height)
#     tile.write(dollar_form(aro_df.loc[row,sec_col]) + arrow_form_num(aro_df.iloc[row]['diffs']))
#     tile.write(aro_df.loc[row,'location'])
#     tile.write('(budget #s)')
# #### CPD
# for row in range(num_rows):
#     tile = grid[row][1].container(height=box_height)
#     tile.write(numb_form(cpd_df.loc[row,sec_col]) + arrow_form_num(cpd_df.iloc[row]['diffs']))
#     tile.write(cpd_df.loc[row,'location'])
#     tile.write('(budget #s)')
# #### LHPC
# for row in range(num_rows):
#     tile = grid[row][2].container(height=box_height)
#     tile.write(format_two_decimals(lhpc_df.loc[row,sec_col]) + arrow_form_num_opp(lhpc_df.iloc[row]['diffs']))
#     tile.write(lhpc_df.loc[row,'location'])
#     tile.write('(budget #s)')
# #### PMix %
# for row in range(num_rows):
#     tile = grid[row][3].container(height=box_height)
#     tile.write(pmix_form(pmix_df.loc[row,sec_col]) + arrow_form_perc(pmix_df.iloc[row]['diffs']))
#     tile.write(pmix_df.loc[row,'location'])
#     tile.write('(budget #s)')
# #### Big 5%
# for row in range(num_rows):
#     tile = grid[row][4].container(height=box_height)
#     tile.write(big5_form(big5_df.loc[row,sec_col]) + arrow_form_perc(big5_df.iloc[row]['diffs']))
#     tile.write(big5_df.loc[row,'location'])
#     tile.write('(budget #s)')
# #### Bay Times
# for row in range(num_rows):
#     tile = grid[row][5].container(height=box_height)
#     tile.write(baytime_form(baytime_df.loc[row,sec_col]) + arrow_form_num_opp(baytime_df.iloc[row]['diffs']))
#     tile.write(baytime_df.loc[row,'location'])
#     tile.write('(budget #s)')




# ### Test Area
# st.markdown("### Test Area - Future Improvements")
# ############################# Trend line test (linear)
# # Prepare data for forecasting
# df_grouped = ext_cars_by_loc.groupby('Date').sum().reset_index()
# # Forecast for the next 3 months
# future_dates = pd.date_range(start=df_grouped['Date'].max() + pd.DateOffset(months=1), periods=3, freq='M')
# # Linear Regression Model for forecasting
# X = np.arange(len(df_grouped)).reshape(-1, 1)
# y = df_grouped['value']
# model = LinearRegression()
# model.fit(X, y)
# # Predict the next 3 months
# X_future = np.arange(len(df_grouped), len(df_grouped) + 3).reshape(-1, 1)
# y_future = model.predict(X_future)
# # Create a DataFrame for forecasted values
# forecast_df = pd.DataFrame({
#     'Date': future_dates,
#     'value': y_future
# })
# # Combine the original and forecasted DataFrames
# combined_df = pd.concat([df_grouped, forecast_df])
# # Plotting
# fig2 = px.bar(ext_cars_by_loc, x='Date', y='value', color='location', title="Cars Serviced by Location (w/linear regression trendline)")
# # Adding the forecasted values as a line
# fig2.add_scatter(x=combined_df['Date'], y=combined_df['value'], mode='lines', name='Trend Line')
# st.plotly_chart(fig2)
# ######################################################
