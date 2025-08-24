# ML Algorithm Tutor with Langchain

A comprehensive, interactive Machine Learning tutoring chatbot built with Streamlit and Google's Gemini AI, featuring a modular OOP architecture, hands-on practice problems, code execution, and rich visualizations.

## Architecture Overview

The application is organized into modular, object-oriented components:

### `/app` Package Structure:
- **`ui.py`** - Main UI class handling all Streamlit components and user interactions
- **`curriculum.py`** - Curriculum management with ML topics, categories, learning paths, and progress tracking
- **`tutor.py`** - AI tutor logic using Google's Gemini model for explanations, practice problems, and code review
- **`memory.py`** - Memory management for chat history and session state
- **`practice_problems.py`** - Generates quizzes, coding, and dataset problems for ML topics
- **`code_executor.py`** - Securely executes and evaluates user-submitted code
- **`visualizations.py`** - Interactive visualizations for ML algorithms using Plotly and Matplotlib

### Entry Points:
- **`main.py`** - Main application entry point (runs the Streamlit app)
- **`run_app.py`** - Enhanced runner with dependency checking and setup

## Features

- **Interactive ML Curriculum**: Browse topics by category, difficulty, and learning path
- **AI-Powered Explanations**: Get detailed, structured explanations with code examples
- **Practice Problems**: Solve quizzes, coding challenges, and dataset-based problems for hands-on learning
- **Code Execution & Feedback**: Write, run, and get instant feedback on your code
- **Streaming Responses**: Real-time AI response generation
- **Visualizations**: Interactive charts and algorithm visualizations for deeper understanding
- **Progress Tracking**: Track your learning progress, quiz scores, and completed topics
- **Chat History**: Review your learning conversation and clear history as needed
- **Responsive UI**: Clean, modern interface with custom styling

## Installation & Setup

1. **Clone and navigate to the project:**
   ```bash
   git clone <your-repo-url>
   cd ML-LangBot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project root with your Google API key:
   ```env
   GOOGLE_API_KEY=your_actual_google_api_key_here
   ```

4. **Run the application:**
   ```bash
   python run_app.py
   ```
   Or directly:
   ```bash
   streamlit run main.py
   ```

The app will open in your browser at http://localhost:8501

## Usage

1. **Browse Curriculum**: Use the sidebar to explore ML topics by category, difficulty, or learning path
2. **Ask Questions**: Type any ML algorithm or topic in the input field to get an AI-powered explanation
3. **Practice Mode**: Enable practice mode to solve quizzes, coding, and dataset problems for selected topics
4. **Code Execution**: Write and evaluate your code for coding challenges, with instant feedback
5. **Visualizations**: View interactive visualizations for supported algorithms
6. **Track Progress**: Monitor your progress, quiz scores, and completed topics
7. **View/Reset History**: Review or clear your chat history at any time

## Supported ML Topics

- Data Preprocessing: Missing Data, Encoding, Feature Scaling
- Regression: Linear Regression, Polynomial Regression, SVR, Decision Tree Regression, Random Forest Regression
- Classification: Logistic Regression, K-NN, SVM, Naive Bayes, Decision Tree Classification, Random Forest Classification
- Clustering: K-Means, Hierarchical Clustering
- Association Rule Learning: Apriori, Eclat
- Reinforcement Learning: UCB, Thompson Sampling, Q-Learning
- NLP: Bag-of-Words, TF-IDF, Word2Vec, BERT
- Deep Learning: ANN, CNN, RNN, LSTM
- Dimensionality Reduction: PCA, LDA, Kernel PCA
- Model Selection & Boosting: Cross Validation, Grid Search, XGBoost, AdaBoost

> **Note:** Practice problems and visualizations are available for a subset of topics. More are being added!

## Contributing

The modular OOP structure makes it easy to contribute:
- Add new curriculum topics or learning paths in `curriculum.py`
- Enhance UI components in `ui.py`
- Improve AI explanations or practice problems in `tutor.py` and `practice_problems.py`
- Extend code execution or feedback in `code_executor.py`
- Add or improve visualizations in `visualizations.py`
- Report bugs or suggest features via issues or pull requests

## Requirements

See `requirements.txt` for all dependencies. Key packages:
- `streamlit`, `langchain`, `langchain-google-genai`, `google-generativeai`, `python-dotenv`
- `plotly`, `matplotlib`, `seaborn`, `numpy`, `pandas`, `scikit-learn`, `xgboost`

## Deployment

   https://ml-langbot.streamlit.app
   
