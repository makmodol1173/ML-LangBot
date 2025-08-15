"""
Enhanced Application Runner
Startup script with dependency checking and setup
"""

import subprocess
import sys
import os

class AppRunner:
    """Application runner with enhanced setup and validation"""
    
    def __init__(self):
        self.required_packages = [
            'streamlit', 
            'langchain', 
            'langchain_google_genai', 
            'google.generativeai',
            'python-dotenv'
        ]
    
    def check_dependencies(self) -> bool:
        """Check if required packages are installed"""
        missing_packages = []
        
        for package in self.required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f" Missing required packages: {', '.join(missing_packages)}")
            print("Please install them using: pip install -r requirements.txt")
            return False
        
        print(" All required packages are installed")
        return True
    
    def check_env_file(self) -> bool:
        """Check if .env file exists and has Google API key"""
        if not os.path.exists('.env'):
            print(" .env file not found")
            print("Please create a .env file with your Google API key:")
            print("GOOGLE_API_KEY=your_api_key_here")
            return False
        
        try:
            with open('.env', 'r') as f:
                content = f.read()
                if 'GOOGLE_API_KEY' not in content or 'your_api_key_here' in content:
                    print(" Please set your actual Google API key in the .env file")
                    return False
        except Exception as e:
            print(f" Error reading .env file: {e}")
            return False
        
        print(" Environment file configured")
        return True
    
    def create_requirements_file(self):
        """Create requirements.txt if it doesn't exist"""
        if not os.path.exists('requirements.txt'):
            requirements = [
                "streamlit>=1.28.0",
                "langchain>=0.1.0",
                "langchain-google-genai>=1.0.0",
                "google-generativeai>=0.3.0",
                "python-dotenv>=1.0.0"
            ]
            
            with open('requirements.txt', 'w') as f:
                f.write('\n'.join(requirements))
            
            print(" Created requirements.txt file")
    
    def run(self):
        """Main method to start the application"""
        print("🤖 ML Algorithm Tutor Chatbot")
        print("=" * 40)
        
        # Create requirements file if needed
        self.create_requirements_file()
        
        # Check dependencies
        if not self.check_dependencies():
            sys.exit(1)
        
        # Check environment configuration
        if not self.check_env_file():
            print("\n Google API key required for ML Algorithm Tutor")
            print("Please set your GOOGLE_API_KEY in the .env file")
            sys.exit(1)
        
        print("\n🚀 Starting ML Algorithm Tutor...")
        print("The app will open in your browser at http://localhost:8501")
        print("Press Ctrl+C to stop the application")
        print("-" * 40)
        
        try:
            # Start Streamlit app
            subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py"])
        except KeyboardInterrupt:
            print("\n\n👋 Application stopped by user")
        except Exception as e:
            print(f"\n Error starting application: {e}")
            print("\n Please check your configuration and try again")

def main():
    """Entry point function"""
    runner = AppRunner()
    runner.run()


if __name__ == "__main__":
    main()
