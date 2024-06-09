import streamlit as st
import pandas as pd
from math import ceil


from GUI import model, df, job_titles, education_list

# Define page functions
def home():
    st.title("Home")
    st.write(
        """
    ## Welcome to the Salary Predictor App 🎉
    This project aims to predict the salary based on various features using a trained machine learning model.
    """
    )
    st.image(
        "https://cdn-payscale.com/content/Research-img.png",
        use_column_width=True,
    )


def dataset():
    st.title("Dataset")
    st.write(
        """
    ## Dataset Statistics 📊
    Here we can display some statistics and graphs about our dataset.
    """
    )
    st.write(df.describe())

    # Display dataset preview
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Plot some graphs
    st.write("### Age Distribution")
    st.bar_chart(df["Age"])

    st.write("### Salary Distribution")
    st.bar_chart(df["Salary"])


def predict_salary():
    st.title("Predict Salary")
    st.write("## Enter the details to predict the salary 💼")

    # Create input fields
    age = st.number_input("Age", min_value=0, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education_level = st.selectbox("Education Level", education_list)
    job_title = st.selectbox("Job Title", job_titles)
    years_experience = st.number_input("Years of Experience", min_value=0.0, step=0.1)

    edu = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
    education_level = edu[education_level]

    # Create predict button
    if st.button("Predict Salary", key="predict"):
        # Create a dataframe from the inputs
        input_data = pd.DataFrame(
            {
                "Age": [age],
                "Gender": [gender == "Male"],
                "Education Level": [education_level],
                "Years of Experience": [years_experience],
            }
        )
        # Create columns for each job title, all set to 0
        for title in job_titles:
            input_data[title] = 0

        # Set the job title that was selected to 1
        input_data[job_title] = 1

        # Use the model to make a prediction
        prediction = model.predict(input_data)

        # Display the prediction
        st.success(f"The predicted salary is: {ceil(prediction[0])}$ yearly")
        # Display the "Try Again" button
        if st.button("Try Again", key="try_again"):
            # Clear all the input fields and reset the page
            st.experimental_rerun()

def about_us():
    st.title('About Us')
    st.write("""
    ## Team Members

    We are a group of dedicated students working together on a project to predict annual salaries based on various factors such as age, sex, job title, and years of experience. Our team members bring diverse skills and expertise to this project, aiming to provide valuable insights and solutions.

     ### Group Members

    | **Name**          | **ID**        |
    |-------------------|---------------|
    | Adane Moges       | ETS0079/13    |
    | Abdulmajid Awol   | ETS0016/13    |
    | Abel Atkelet      | ETS0020/13    |
    | Amanuel Mandefrow | ETS0122/13    |
    | Elias Balude      | ETS0237/12    |

    ________________________________________________________________

    We have combined our knowledge in data science, machine learning, and software development to create a comprehensive and user-friendly application. This project not only showcases our technical skills but also our ability to work collaboratively to achieve common goals.

    ### Our Goal

    Our goal is to develop a reliable salary prediction model that can assist both professionals and employers in understanding salary dynamics and making informed decisions. We hope that our work can contribute to better career planning and compensation strategies.

    ### Contact Us

    For any inquiries or feedback, please feel free to contact us at [hear](mailto:adanemoges6@gmail.com).

    Thank you for using our Salary Predictor application!
    """)




# Run the app
if __name__ == "__main__":
    pass
    st.write(
        "The app is running..."
    )  # This line can be omitted as Streamlit apps are run with `streamlit run script_name.py`
