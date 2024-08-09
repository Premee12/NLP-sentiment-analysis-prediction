# NLP Sentiment Analysis Project

## Project Overview

This project involves building a robust sentiment analysis model using machine learning techniques. The model was trained using sample Amazon review data and incorporates advanced methods such as vectorization, stemming, and parameter tuning to enhance prediction accuracy. Model explainability is achieved using SHAP, providing valuable insights into feature importance. The solution utilizes Flask to create a web API for real-time sentiment prediction and bulk analysis through CSV file uploads. This solution provides actionable insights into customer sentiment, enabling better understanding and improvement of business strategies.

## Features

- **Real-time Sentiment Prediction**: Analyze individual text inputs to predict sentiment.
- **Bulk Sentiment Analysis**: Upload a CSV file for bulk predictions and receive a CSV file with sentiment results.
- **Model Explainability**: Utilize SHAP for understanding feature importance and model decisions.

## To Run This Project

Follow these steps to set up and run the project:

### Step 1: Clone the Repository

Clone this repository to your local machine:
git clone <repository-url>


### Step 2: Create and Activate a Virtual Environment

Create a virtual environment and activate it:


### Step 3: Install Dependencies

Install the required packages:


### Step 4: Run the Flask Application

Start the Flask application:


### Step 5: Access the Application

The app will run on port 8000. Open your web browser and navigate to:


### Step 6: Test Prediction

Use the provided interface to test real-time sentiment prediction by entering text into the input field and submitting.

### Step 7: Bulk Prediction

To use bulk prediction:
1. Upload a CSV file containing a field of texts through the provided form.
2. The prediction will return another CSV document in your downloads folder.
3. The resulting file will contain two fields: the original text and the predicted sentiment.

## Tools and Technologies

- **Machine Learning**: For sentiment prediction model training.
- **Vectorization and Stemming**: For text preprocessing.
- **SHAP**: For model explainability.
- **Flask**: For building the web API.
- **Pandas**: For data manipulation and CSV handling.
- **Matplotlib**: For generating visualizations.



## Contact

For any questions or feedback, feel free to reach out to me at abayomialliayomide@gmail.com.

---

Thank you for checking out this project!



# This is a NLP end to end project built using Flask API


## To run this project


Step 1: Cloned this repository and clone it
```
Step 2:  Create a virtual environment. Activate the new environment
```


Step 3: Install the requirements file
```
pip install -r requirements.txt
```

Step 4: Run the app
```
flask --app api.py run
```

Step 5: The app will run on port 5000. 
```
localhost:5000
```
