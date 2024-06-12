import streamlit as st
import pandas as pd
# import os

st.set_page_config(page_title = "Remedy Place - Streamlit Example Webpages", page_icon = "🛁" )

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


st.write("""
        # Remedy Place - Streamlit Example Webpages

        This website will highlight a handful of Streamlit possibilities for Remedy Place.
        """)
