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
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("🧪 Running Practice Problems Feature Test...")
        from test_practice_problems import run_comprehensive_test
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    elif len(sys.argv) > 1 and sys.argv[1] == "--demo":
        print("🎯 Running Practice Problems Demo...")
        from demo_practice_problems import demo_practice_problems
        demo_practice_problems()
        sys.exit(0)
    
    app = MLTutorUI()
    app.run()


if __name__ == "__main__":
    main()
