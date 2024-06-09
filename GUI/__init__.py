# Load the trained models
import pickle
import sys
import os

# Add the path to the model directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "model")))
print(sys.path)

model = pickle.load(
    open(
        "model/salary predictor(dtr).pkl",
        "rb",
    )
)
df = pickle.load(open("model/df.pkl", "rb"))

# Define the unique job titles
job_titles = df.columns[5:].tolist()
education_list = ["High School", "Bachelor's", "Master's", "PhD"]
