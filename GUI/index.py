import streamlit as st
import pandas as pd
from math import ceil
import matplotlib.pyplot as plt  # Required for plotting
import seaborn as sns  # Required for visualizations
from GUI import model, df, job_titles, education_list


from GUI import model, df, job_titles, education_list

# Define page functions
def home():
    
    st.title("Experience Salary Predictor")
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
    
    st.header("Features")
    st.write("""
    - **Dataset Upload**: Upload your dataset for training.
    - **Model Training**: Train a machine learning model with your dataset.
    - **Salary Prediction**: Predict salaries based on gender, job Title, age and experience input.
    """)
    
    st.header("Instructions")
    st.write("""
    1. Navigate to the 'predict salary' page to upload your dataset and train the model.
    2. input experience and  other parameters the see the predicted salary.
    3. Visit the 'Dataset' page to learn more about this project and the dataset we used to trai the model.
    4. For any questions, go to the 'About us' page.
    """)


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
    age = st.number_input("Age", min_value=18, step=1)
    gender = st.selectbox("Gender", ["Not Selected","Male", "Female"])
    education_level = st.selectbox("Education Level", ["Not Selected"]+education_list)
    job_title = st.selectbox("Job Title", ["Not Selected"]+job_titles)
    years_experience = st.number_input("Years of Experience", min_value=0, step=1)
    if gender == "Not Selected":
        st.warning("Please select a gender.")
        return
    if education_level == "Not Selected":
        st.warning("Please select an education level.")
        return
    if job_title == "Not Selected":
        st.warning("Please select a job title.")
        return




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
def graph_analysis():
    st.title("Graph Analysis")
    st.write("## Insights into Salary Trends 📊")
    st.write("Explore key trends and relationships in the dataset through visualizations. Each chart is accompanied by a description for better understanding.")

    # Gender Mapping
    if 'Gender' in df.columns:
        if df['Gender'].dtype == 'int64':  # Dynamically map gender if encoded
            gender_mapping = df[['Gender', 'Salary']].groupby('Gender').mean().sort_values(by='Salary').index
            gender_map = {gender_mapping[0]: "Female", gender_mapping[1]: "Male"}
            df['Gender'] = df['Gender'].map(gender_map)

    # Gender Distribution
    st.subheader("1. Gender Distribution")
    if 'Gender' in df.columns:
        gender_distribution_chart = plt.figure(figsize=(8, 5))
        sns.countplot(x='Gender', data=df, palette='pastel')
        plt.title('Gender Distribution')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        st.pyplot(gender_distribution_chart)
        st.write("**Description**: This chart shows the count of employees grouped by gender, providing insight into gender representation in the dataset.")

    # Education Level Distribution
    st.subheader("2. Education Level Distribution")
    if 'Education Level' in df.columns:
        if df['Education Level'].dtype == 'int64':  # Map encoded values to original names
            edu_map = {0: "High School", 1: "Bachelor's", 2: "Master's", 3: "PhD"}
            df['Education Level'] = df['Education Level'].map(edu_map)

        education_distribution_chart = plt.figure(figsize=(8, 5))
        sns.countplot(x='Education Level', data=df, palette='muted')
        plt.title('Education Level Distribution')
        plt.xlabel('Education Level')
        plt.ylabel('Count')
        st.pyplot(education_distribution_chart)
        st.write("**Description**: This chart shows the distribution of employees by education level, offering insights into workforce qualifications.")

    # Top 10 Highest Paying Jobs
    st.subheader("3. Top 10 Highest Paying Jobs")
    if 'Job Title' in df.columns and 'Salary' in df.columns:
        top_10_jobs = df.groupby('Job Title')['Salary'].mean().nlargest(10)
        top_jobs_chart = plt.figure(figsize=(12, 6))
        sns.barplot(x=top_10_jobs.index, y=top_10_jobs.values, palette='Blues_d')
        plt.title('Top 10 Highest Paying Jobs')
        plt.xlabel('Job Title')
        plt.ylabel('Mean Salary')
        plt.xticks(rotation=60)
        st.pyplot(top_jobs_chart)
        st.write("**Description**: This chart displays the top 10 highest-paying jobs based on average salary. It highlights roles with significant earning potential.")

    # Age Distribution
    st.subheader("4. Age Distribution")
    if 'Age' in df.columns:
        age_distribution_chart = plt.figure(figsize=(10, 5))
        sns.histplot(df['Age'], kde=True, color='blue')
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        st.pyplot(age_distribution_chart)
        st.write("**Description**: This chart illustrates the distribution of ages in the dataset, highlighting the predominant age groups.")

    # Years of Experience Distribution
    st.subheader("5. Years of Experience Distribution")
    if 'Years of Experience' in df.columns:
        experience_distribution_chart = plt.figure(figsize=(10, 5))
        sns.histplot(df['Years of Experience'], kde=True, color='orange')
        plt.title('Years of Experience Distribution')
        plt.xlabel('Years of Experience')
        plt.ylabel('Frequency')
        st.pyplot(experience_distribution_chart)
        st.write("**Description**: This chart shows the distribution of employees by years of experience, providing insight into their professional background.")

    # Salary Distribution
    st.subheader("6. Salary Distribution")
    if 'Salary' in df.columns:
        salary_distribution_chart = plt.figure(figsize=(10, 5))
        sns.histplot(df['Salary'], kde=True, color='green')
        plt.title('Salary Distribution')
        plt.xlabel('Salary')
        plt.ylabel('Frequency')
        st.pyplot(salary_distribution_chart)
        st.write("**Description**: This chart depicts the distribution of salaries in the dataset, identifying common ranges and outliers.")

    # Gender vs. Salary
    st.subheader("7. Gender vs. Salary")
    if 'Gender' in df.columns and 'Salary' in df.columns:
        gender_salary_chart = plt.figure(figsize=(8, 5))
        sns.barplot(x='Gender', y='Salary', data=df, palette='coolwarm', ci=None)
        plt.title('Mean Salary by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Mean Salary')
        st.pyplot(gender_salary_chart)
        st.write("**Description**: This chart compares the average salaries of males and females, providing insights into any gender-based pay disparities.")

    # Education Level vs. Salary
    st.subheader("8. Education Level vs. Salary")
    if 'Education Level' in df.columns and 'Salary' in df.columns:
        education_salary_chart = plt.figure(figsize=(8, 5))
        sns.boxplot(x='Education Level', y='Salary', data=df, palette='Set3')
        plt.title('Salary Distribution by Education Level')
        plt.xlabel('Education Level')
        plt.ylabel('Salary')
        st.pyplot(education_salary_chart)
        st.write("**Description**: This boxplot shows how salaries vary by education level, highlighting the financial advantages of higher education.")

    # Salary vs. Years of Experience
    st.subheader("9. Salary vs. Years of Experience")
    if 'Years of Experience' in df.columns and 'Salary' in df.columns:
        experience_salary_chart = plt.figure(figsize=(10, 6))
        sns.lineplot(x='Years of Experience', y='Salary', data=df, marker='o', color='blue')
        plt.title('Salary vs. Years of Experience')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')
        st.pyplot(experience_salary_chart)
        st.write("**Description**: This chart shows how salaries increase with years of experience, emphasizing the benefits of professional growth.")

    # Salary vs. Age Group
    st.subheader("10. Salary vs. Age Group")
    if 'Age' in df.columns and 'Salary' in df.columns:
        valid_df = df[df['Age'] <= 50]  # Exclude retirement-age anomalies
        age_groups = pd.cut(valid_df['Age'], bins=[20, 30, 40, 50], labels=['20-30', '30-40', '40-50'])
        avg_salary_age = valid_df.groupby(age_groups)['Salary'].mean()
        age_salary_chart = plt.figure(figsize=(8, 5))
        avg_salary_age.plot(kind='bar', color='skyblue')
        plt.title('Average Salary by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Average Salary')
        st.pyplot(age_salary_chart)
        st.write("**Description**: This chart shows the average salary for different age groups, highlighting how earnings grow over time.")
    
    # Gender-wise Salary Distribution
    st.subheader("11. Gender-wise Salary Distribution")
    if 'Gender' in df.columns and 'Salary' in df.columns:
        gender_salary_chart = plt.figure(figsize=(8, 5))
        sns.boxplot(x='Gender', y='Salary', data=df, palette='Set2')
        plt.title('Gender-wise Salary Distribution')
        plt.xlabel('Gender')
        plt.ylabel('Salary')
        st.pyplot(gender_salary_chart)
        st.write("**Description**: This boxplot highlights salary variations by gender, offering clarity on gender-based salary trends.")


def salary_insights():
    st.title("Salary Insights")
    st.write("## Discover Key Trends and Highlights 💡")

    # Check if the dataset has necessary columns
    if df.empty:
        st.warning("The dataset is empty. Please upload or reload the data.")
        return

    # Average Salary
    if 'Salary' in df.columns:
        avg_salary = df['Salary'].mean()
        st.metric(label="Average Salary", value=f"${avg_salary:,.2f}")

        # Highest Salary
        max_salary = df['Salary'].max()
        max_salary_job = df[df['Salary'] == max_salary]['Job Title'].iloc[0] if 'Job Title' in df.columns else "N/A"
        st.write(f"**Highest Salary**: ${max_salary:,.2f} (Role: {max_salary_job})")

        # Lowest Salary
        min_salary = df['Salary'].min()
        min_salary_job = df[df['Salary'] == min_salary]['Job Title'].iloc[0] if 'Job Title' in df.columns else "N/A"
        st.write(f"**Lowest Salary**: ${min_salary:,.2f} (Role: {min_salary_job})")
    else:
        st.warning("The dataset does not contain a 'Salary' column.")

    # Most Common Job Title
    if 'Job Title' in df.columns:
        most_common_job = df['Job Title'].mode()[0]
        st.write(f"**Most Common Job Title**: {most_common_job} (appears {df['Job Title'].value_counts().max()} times)")
    else:
        st.warning("!")

    # Gender Pay Gap
    if 'Gender' in df.columns and 'Salary' in df.columns:
        # Ensure gender uses original names if encoded
        if df['Gender'].dtype == 'int64':  # Assume encoded as integers
            gender_map = {0: "Female", 1: "Male"}
            df['Gender'] = df['Gender'].map(gender_map)

        gender_avg_salary = df.groupby('Gender')['Salary'].mean()
        st.write("### Gender Pay Gap:")
        for gender, avg in gender_avg_salary.items():
            st.write(f"- **{gender}**: ${avg:,.2f}")
    else:
        st.warning("The dataset does not contain 'Gender' or 'Salary' columns.")

    # Education Level Salary Insights
    if 'Education Level' in df.columns and 'Salary' in df.columns:
        # Ensure education level uses original names if encoded
        if df['Education Level'].dtype == 'int64':  # Assume encoded as integers
            edu_map = {0: "High School", 1: "Bachelor's", 2: "Master's", 3: "PhD"}
            df['Education Level'] = df['Education Level'].map(edu_map)

        edu_salary = df.groupby('Education Level')['Salary'].mean().sort_values(ascending=False)
        st.write("### Average Salary by Education Level:")
        for level, avg in edu_salary.items():
            st.write(f"- **{level}**: ${avg:,.2f}")
    else:
        st.warning("The dataset does not contain 'Education Level' or 'Salary' columns.")

    # Experience Insights
    if 'Years of Experience' in df.columns and 'Salary' in df.columns:
        max_experience = df['Years of Experience'].max()
        min_experience = df['Years of Experience'].min()
        experience_salary = df.groupby('Years of Experience')['Salary'].mean().sort_values(ascending=False)
        top_experience = experience_salary.index[0]
        st.write(f"**Longest Career Path**: {max_experience} years")
        st.write(f"**Highest Paying Experience Level**: {top_experience} years (${experience_salary[top_experience]:,.2f})")
    else:
        st.warning("The dataset does not contain 'Years of Experience' or 'Salary' columns.")

    # Fun Fact
    if 'Job Title' in df.columns:
        total_jobs = len(df['Job Title'].unique())
        st.write(f"### Did You Know? There are **{total_jobs} unique job titles** in this dataset!")
    else:
        st.warning("!")

    # Closing with insights
    st.write("## Key Takeaways 🧐")
    st.write("""
    - **Top-Paying Role**: The highest salary is attributed to a standout job, providing a clear benchmark.
    - **Education Level Impact**: Advanced degrees like PhDs dominate salary charts.
    - **Gender Trends**: Gender pay gap persists but shows interesting trends across the dataset.
    - **Career Growth**: Experience significantly boosts earning potential, rewarding long-term dedication.
    """)



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
    ### Source Codes
    The source code for this project is available on [Github](https://github.com/habeshaethiopia/ML_experience_salary.git)

    Thank you for using our Salary Predictor application!
    """)




# Run the app
if __name__ == "__main__":
    pass
    st.write(
        "The app is running..."
    )  # This line can be omitted as Streamlit apps are run with `streamlit run script_name.py`
