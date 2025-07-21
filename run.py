#!/usr/bin/env python3
"""
Finance Assistant - Auto Setup and Run Script
This script automatically installs all dependencies and runs the application.
"""

import subprocess
import sys
import os
import time

def print_banner():
    """Print a nice banner"""
    print("=" * 60)
    print("🚀 Finance Assistant - Auto Setup & Run")
    print("=" * 60)
    print()

def install_requirements():
    """Install all required packages from requirements.txt"""
    print("📦 Installing Python packages...")
    try:
        # Check if requirements.txt exists
        if not os.path.exists("requirements.txt"):
            print("❌ requirements.txt not found!")
            return False
        
        # Install packages
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--upgrade"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All packages installed successfully!")
            return True
        else:
            print("❌ Error installing packages:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error during installation: {e}")
        return False

def check_environment():
    """Check if environment variables are set"""
    print("🔑 Checking environment variables...")
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("⚠️  Warning: GEMINI_API_KEY environment variable not set!")
        print("   You'll need to set this for the chatbot to work.")
        print("   Add this to your environment: GEMINI_API_KEY=your_api_key_here")
    else:
        print("✅ GEMINI_API_KEY found!")
    
    return True

def run_streamlit():
    """Run the Streamlit application"""
    print("🌐 Starting Finance Assistant...")
    print("📱 The app will open in your browser automatically!")
    print("🛑 Press Ctrl+C to stop the application")
    print()
    
    try:
        # Choose which app to run based on what exists
        if os.path.exists("Test Frontend Structure/frontend_controller.py"):
            app_file = "Test Frontend Structure/frontend_controller.py"
            print(f"🎯 Running: {app_file}")
        elif os.path.exists("app.py"):
            app_file = "app.py"
            print(f"🎯 Running: {app_file}")
        else:
            print("❌ No main application file found!")
            print("   Looking for: 'Test Frontend Structure/frontend_controller.py' or 'app.py'")
            return False
        
        # Run streamlit
        subprocess.run([
            "streamlit", "run", app_file, "--server.address", "localhost", "--server.port", "8501"
        ])
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error running application: {e}")
        return False

def main():
    """Main setup and run function"""
    print_banner()
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Setup failed at package installation")
        return
    
    print()
    
    # Step 2: Check environment
    check_environment()
    print()
    
    # Step 3: Ask user if they want to run the app
    print("🎉 Setup completed successfully!")
    print()
    
    while True:
        run_app = input("🚀 Do you want to run the Finance Assistant now? (y/n): ").lower().strip()
        if run_app in ['y', 'yes']:
            print()
            run_streamlit()
            break
        elif run_app in ['n', 'no']:
            print()
            print("👍 Setup complete! You can run the app later with:")
            if os.path.exists("Test Frontend Structure/frontend_controller.py"):
                print("   streamlit run \"Test Frontend Structure/frontend_controller.py\"")
            else:
                print("   streamlit run app.py")
            break
        else:
            print("Please enter 'y' for yes or 'n' for no")

if __name__ == "__main__":
    main()
