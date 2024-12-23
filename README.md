# Experience Salary Prediction
## Link to  colab [here](https://colab.research.google.com/drive/1TVu_MI1hGtt8GigJNBnI-QXm50MpLUXe#scrollTo=6e4344f0)

A project to predict salaries based on machine learning experience using various data and models.
you can access the website [here](https://habeshaethiopia-ml-experience-salary-app-6qlnac.streamlit.app/)
# **Experience Salary Prediction**

A machine learning project designed to predict salaries based on professional experience. The project utilizes various datasets and machine learning models, providing insights into salary trends and predictions through a user-friendly web application.  

## **Links**  
- **Colab Notebook**: [Open Colab](https://colab.research.google.com/drive/1TVu_MI1hGtt8GigJNBnI-QXm50MpLUXe#scrollTo=6e4344f0)  
- **Web Application**: [Visit the App](https://habeshaethiopia-ml-experience-salary-app-6qlnac.streamlit.app/)  

## **Table of Contents**  
1. [Installation](#installation)  
2. [Usage](#usage)  
3. [Project Structure](#project-structure)  
4. [Contributing](#contributing)  
5. [License](#license)  

---

## **Installation**  

### Step 1: Clone the Repository  
Clone the project repository to your local machine:  
```bash
git clone https://github.com/habeshaethiopia/ML_experience_salary.git
cd ML_experience_salary
```  

### Step 2: Set Up a Virtual Environment  
Create and activate a virtual environment to isolate project dependencies:  
- **For Linux/macOS**:  
  ```bash
  python -m venv venv
  source venv/bin/activate
  ```  
- **For Windows**:  
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```  

### Step 3: Install Dependencies  
Install the required Python packages listed in `requirements.txt`:  
```bash
pip install -r requirements.txt
```  

---

## **Usage**  

### Step 1: Prepare the Dataset  
- Place the dataset file(s) for training and evaluation in the `dataset` directory.  
- Ensure the data is properly formatted and cleaned for model consumption.  

### Step 2: Train the Model  
1. Open the main script `app.py` and adjust parameters, such as hyperparameters or file paths, to fit your requirements.  
2. Run the script to train and evaluate the model:  
   ```bash
   python app.py
   ```  

### Step 3: Launch the Web Application  
1. Navigate to the `GUI` directory, which contains the Streamlit web application files:  
   ```bash
   cd GUI
   ```  
2. Run the application using the following command:  
   ```bash
   python app.py
   ```  
3. Open the provided localhost link in your browser to access the interface.  

---

## **Project Structure**  

The project is organized into the following directories and files:  

- **`dataset/`**: Directory containing the raw and preprocessed dataset files.  
- **`model/`**: Stores the trained model files for reuse and evaluation.  
- **`GUI/`**: Contains the Streamlit-based graphical user interface files for end-user interaction.  
- **`app.py`**: Main Python script for data processing, training, and evaluation of the machine learning model.  
- **`requirements.txt`**: Lists all the dependencies required for the project.  

---

## **Contributing**  

We welcome contributions to improve this project! To contribute:  

1. **Fork the Repository**:  
   Click the "Fork" button at the top of the repository page on GitHub.  

2. **Create a Feature Branch**:  
   ```bash
   git checkout -b feature-branch-name
   ```  

3. **Make Changes**:  
   Implement your changes and commit them with descriptive messages:  
   ```bash
   git commit -am "Add a new feature"
   ```  

4. **Push to Your Fork**:  
   ```bash
   git push origin feature-branch-name
   ```  

5. **Submit a Pull Request**:  
   Open a pull request on the main repository and describe your changes.  

---

## **License**  

This project is licensed under the **MIT License**, a permissive license that allows anyone to freely use, modify, and distribute the software, provided they include proper attribution and retain the copyright notice.  

For more details, view the [LICENSE](LICENSE) file in the repository.  

---  
Let me know if you need further refinements!