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
