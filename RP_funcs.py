'''
Helper functions to create, add entries, etc. for T5 Oil
'''
import pymysql
import paramiko
from paramiko import SSHClient
from sshtunnel import SSHTunnelForwarder
from os.path import expanduser
import streamlit as st
import pandas as pd
import numpy as np
import hmac
import time


def make_conn_ssh(host_ssh, user_ssh, port_ssh, pk_ssh,
                  db_host, db_user, db_port, db_password, db_name):
    tunnel = SSHTunnelForwarder(
        (host_ssh, port_ssh),
        ssh_username=user_ssh,
        ssh_pkey=pk_ssh,
        remote_bind_address=(db_host, db_port),
        local_bind_address=('127.0.0.1',)
    )
    tunnel.start()
    conn = pymysql.connect(
        host='127.0.0.1',
        user=db_user,
        password=db_password,
        db=db_name,
        port=tunnel.local_bind_port,
        connect_timeout=60,       # Increase connect timeout
        read_timeout=60,          # Increase read timeout
        write_timeout=60          # Increase write timeout
    )
    return conn, tunnel

def read_in_SQL(connection, query):
    try:
        # Read data into a Pandas DataFrame
        df = pd.read_sql(query, connection)
        return df
    except pymysql.MySQLError as e:
        print(f"An error occurred: {e.args[0]}, {e.args[1]}")
        connection.rollback()
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
def filter_add_accounts(df, low_n, high_n):
    filtered_pivot_table = df[
        (df.index.get_level_values('Account_Num') > low_n) & 
        (df.index.get_level_values('Account_Num') < high_n) ]
    try:
        return filtered_pivot_table.sum(axis=0)   # Sum up along the columns
    except: 
        return 0

def create_RP_pivot_table(df):
    '''
    Takes the result dataframe and the RP location and returns 
    a pivot dataframe across all months
    '''
    # import pandas as pd
    import numpy as np
    pivot_table = df.pivot_table(index=['category', 'location'], columns='MY', values='amount', aggfunc='sum')
    pivot_table = pivot_table.reindex(columns=pivot_table.columns.sort_values())
    pivot_table.columns = pivot_table.columns.strftime('%b %y')
    # pivot_table.replace(0.0, np.nan, inplace=True)
    return(pivot_table)



def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["pagepassword"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

