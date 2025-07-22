#!/usr/bin/env python3
"""
Finance Assis    #    # Run the app
    print("🚀 Starting application...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "Test Frontend Structure/frontend_controller.py"])
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")he app
    print("🚀 Starting application...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "Test Frontend Structure/frontend_controller.py"])
    except KeyboardInterrupt:
        print("\n👋 Stopped by user") Simple Setup & Run
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing Python packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def setup_api_key():
    """Setup API key"""
    api_key = "AIzaSyByTfCk2a6m4gkeJAuCpWGmWi8qfyHBQ3w"
    
    with open(".env", 'w') as f:
        f.write(f"GEMINI_API_KEY={api_key}\n")
    print("✅ API key configured!")
    return True

def main():
    """Main function"""
    print("� Finance Assistant - Setup & Run")
    print("=" * 40)
    
    # Install packages
    if not install_requirements():
        return
    
    # Setup API key
    if not setup_api_key():
        return
    
    # Run the app
    print("🚀 Starting application...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "Test Frontend Structure/frontend_controller.py"])
    except KeyboardInterrupt:
        print("\n� Stopped by user")

if __name__ == "__main__":
    main()
