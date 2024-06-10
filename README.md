

# Experience Salary Prediction

A project to predict salaries based on machine learning experience using various data and models.
you can access the website [here](https://habeshaethiopia-ml-experience-salary-app-ksze0a.streamlit.app/)

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/habeshaethiopia/ML_experience_salary.git
   cd ML_experience_salary
   ```

2. **Create a virtual environment:**
   ```sh
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the dataset:**
   - Place your dataset files in the `dataset` directory.

2. **Train the model:**
   - Modify `app.py` for any custom settings.
   - Run the training script:
     ```sh
     python app.py
     ```

3. **Run the GUI:**
   - Navigate to the `GUI` directory:
     ```sh
     cd GUI
     ```
   - Run the GUI application:
     ```sh
     python app.py
     ```

## Project Structure

- **dataset/**: Contains the data files used for training and testing.
- **model/**: Contains trained model files.
- **GUI/**: Contains the graphical user interface for interacting with the model.
- **app.py**: Main script for training and evaluating the model.
- **requirements.txt**: List of required Python packages.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License.

---
