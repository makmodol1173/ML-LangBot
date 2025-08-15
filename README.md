# ML Algorithm Tutor with Langchain

A comprehensive Machine Learning tutoring chatbot built with Streamlit and Google's Gemini AI, now refactored with clean OOP architecture.

## Architecture Overview

The application is now organized into modular, object-oriented components:

### `/app` Package Structure:
- **`ui.py`** - Main UI class handling all Streamlit components and user interactions
- **`curriculum.py`** - Curriculum management with ML topics and categories
- **`tutor.py`** - AI tutor logic using Google's Gemini model for explanations
- **`memory.py`** - Memory management for chat history and session state

### Entry Points:
- **`main.py`** - Clean application entry point
- **`run_app.py`** - Enhanced runner with dependency checking and setup

## Features

- **Interactive ML Curriculum**: Browse topics by category
- **AI-Powered Explanations**: Get detailed explanations with code examples
- **Chat History**: Track your learning conversation
- **Streaming Responses**: Real-time AI response generation
- **Quick Topics**: Fast access to popular ML algorithms
- **Responsive UI**: Clean, modern interface with custom styling

## Installation & Setup

1. **Clone and navigate to the project:**
   \`\`\`bash
   cd ml-tutor-oop
   \`\`\`

2. **Install dependencies:**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Set up environment variables:**
   \`\`\`bash
   cp .env.example .env
   # Edit .env and add your Google API key
   \`\`\`

4. **Run the application:**
   \`\`\`bash
   python run_app.py
   \`\`\`
   Or directly:
   \`\`\`bash
   python main.py
   \`\`\`

## Configuration

Create a `.env` file with your Google API key:
\`\`\`
GOOGLE_API_KEY=your_actual_google_api_key_here
\`\`\`

## Usage

1. **Browse Curriculum**: Use the sidebar to explore ML topics by category
2. **Ask Questions**: Type any ML algorithm or topic in the input field
3. **Quick Access**: Use the quick topic buttons for popular algorithms
4. **View History**: Track your learning conversation in the chat history
5. **Clear History**: Reset your session anytime

## Supported ML Topics

- Data Preprocessing, Regression, Classification
- Clustering, Association Rule Learning
- Reinforcement Learning, NLP, Deep Learning
- Dimensionality Reduction, Model Selection & Boosting

## Contributing

The new OOP structure makes it easy to contribute:
- Add new curriculum topics in `curriculum.py`
- Enhance UI components in `ui.py`
- Improve AI responses in `tutor.py`
- Extend memory features in `memory.py`
