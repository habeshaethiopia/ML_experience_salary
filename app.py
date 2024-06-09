import GUI
import streamlit as st

from GUI.index import home, dataset, predict_salary, about_us
from streamlit_option_menu import option_menu




if __name__=='__main__':
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Home", "Dataset", "Predict Salary", "About Us"],
            icons=["house", "book", "envelope", "info-circle"],
            menu_icon="cast",
            default_index=0,
        )

    # Page selection
    if selected == 'Home':
        home()
    elif selected == 'Dataset':
        dataset()
    elif selected == 'Predict Salary':
        predict_salary()
    elif selected == 'About Us':
        about_us()