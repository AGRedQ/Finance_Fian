


# Finance Assistant

## Quick Start (Recommended)

For an automated setup and launch, simply run:
```bash
py run.py
```
## Gemini API

Please remember to have Gemini API ready (since Github prohibits sharing personal API_KEYS)


This will automatically:
- Install all required dependencies from requirements.txt
- Launch the Finance Assistant application
- Open your browser to the application

## 🎓 For Assignment Submission/Teacher Testing

**Important for Teachers/Evaluators:**

1. **Quick Setup**: Run `py run.py` and select option **"1. Use teacher/demo API key"**
2. **API Key**: The student should replace `TEACHER_DEMO_KEY_HERE` in `run.py` (line ~80) with their actual Gemini API key
3. **Full Testing**: This enables complete chatbot functionality including:
   - `/help` - List all commands
   - `/display AAPL` - Show stock analysis with charts
   - `/compare AAPL MSFT` - Compare two stocks
   - `/calculate RSI AAPL` - Calculate technical indicators
   - Chat queries - Ask questions about stocks and technical analysis

**Features to Test:**
- 💬 AI Chatbot with stock analysis commands
- 📊 Real-time stock data visualization  
- 🔧 Technical indicator calculations (RSI, MACD, SMA, etc.)
- 📈 Stock comparison charts
- ⚙️ Settings and customization options

## Manual Installation

If you prefer to install dependencies manually:

### Required Libraries:
- streamlit
- pandas
- numpy
- matplotlib
- yfinance
- plotly
- scikit-learn
- xgboost
- prophet
- pandas-datareader
- google-generativeai
- pytz

### Installation:
```bash
pip install -r requirements.txt
```

### Run the Application:
```bash
streamlit run "Test Frontend Structure/frontend_controller.py"
```

## Features

- 💬 **AI Chatbot**: Interactive chat with stock analysis commands (/help, /display, /compare)
- 📈 **Stock Tracking**: Track multiple tickers with real-time data and currency normalization
- 🔮 **Prediction Assistant**: Advanced ML-based trend prediction with multiple models
- 📊 **Data Visualization**: Customizable charts with multiple themes (Light/Dark/Auto)
- ⚙️ **Settings**: Personalize your experience with chart preferences and history tracking
- 🌍 **Multi-Currency**: Support for international stocks with automatic currency conversion
- 📋 **Activity History**: Track all your interactions and analysis requests
- 🤖 **Real-time API Status**: Monitor Gemini API and market status on dashboard
- 📈 **Market Status**: Live market open/closed status with timezone awareness

## Environment Setup

To use the AI chatbot features, you'll need to set up your Gemini API key:

1. Get a Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set the environment variable:
   - **Windows**: `set GEMINI_API_KEY=your_api_key_here`
   - **Linux/Mac**: `export GEMINI_API_KEY=your_api_key_here`

## Usage

1. **Dashboard**: View tracked tickers, API status, and market status
2. **Chatbot**: Use commands like `/display AAPL` or `/compare AAPL MSFT`
3. **Prediction Assistant**: Analyze stock trends with machine learning models
4. **Settings**: Customize charts, manage tickers, and view activity history

## Support

If you encounter any issues:
1. Make sure all dependencies are installed with `pip install -r requirements.txt`
2. Verify your Gemini API key is set correctly
3. Check that you're using Python 3.8 or higher

