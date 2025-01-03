import GUI
import streamlit as st
from streamlit_option_menu import option_menu
from GUI.index import home, dataset, predict_salary, about_us, graph_analysis, salary_insights

if __name__ == '__main__':
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Home", "Dataset", "Predict Salary", "Graph Analysis", "Salary Insights", "About Us"],
            icons=["house", "book", "envelope", "bar-chart", "clipboard-data", "info-circle"],  # Added icon
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
    elif selected == 'Graph Analysis':
        graph_analysis()
    elif selected == 'Salary Insights':
        salary_insights()
    elif selected == 'About Us':
        about_us()
