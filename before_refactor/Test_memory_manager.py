import json
class Memory_Manager:   # Top 1 pritority
    def __init__(self, path):
        self.path = path
        self.ticker_memory = {}
        self.website_memory = {}
    # For Tickers

    def add_ticker(self, ticker):
        self.ticker_memory[ticker] = self.ticker_memory.get(ticker, 0) + 1
    
    def remove_ticker(self, ticker):
        if ticker in self.ticker_memory:
            del self.ticker_memory[ticker]

    def get_top_tickers(self, n=3):
        sorted_tickers = sorted(self.ticker_memory.items(), key=lambda item: item[1], reverse=True)
        return sorted_tickers[:n]
    
    def load_file(self):
        try:
            with open(self.path, 'r') as file:
                data = json.load(file)
                self.ticker_memory = data.get('ticker_memory', {})
                self.website_memory = data.get('website_memory', {})
        except FileNotFoundError:
            print(f"File {self.path} not found. Starting with empty memory.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {self.path}. Starting with empty memory.")
    def save_file(self):
        data = {
            'ticker_memory': self.ticker_memory,
            'website_memory': self.website_memory
        }
        with open(self.path, 'w') as file:
            json.dump(data, file, indent=4)





if __name__ == "__main__":
    # Example usage
    memory_manager = Memory_Manager("./memory_data.json")
    memory_manager.load_file()
    memory_manager.add_ticker("AAPL")
    memory_manager.add_ticker("GOOGL") 
    memory_manager.add_ticker("MSFT")
    memory_manager.add_ticker("AAPL")  
    memory_manager.add_ticker("NVDA")  
     
    print(memory_manager.get_top_tickers())  # Should show top tickers

    memory_manager.save_file()
