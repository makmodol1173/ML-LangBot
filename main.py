"""
Main Application Entry Point
Clean entry point that imports and runs the UI
"""

import os
import sys
import warnings
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.ui import MLTutorUI

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*Thread.*missing ScriptRunContext.*")

# Load environment variables
load_dotenv()

# Set environment variable to suppress Streamlit warnings
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'


def main():
    """Main application function"""
    app = MLTutorUI()
    app.run()


if __name__ == "__main__":
    main()