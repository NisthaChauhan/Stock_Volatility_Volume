'''"""
Ticker Validation and Data Fetching
Validates ticker symbols and fetches historical stock data
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class TickerValidator:
    """Validates and fetches stock data for a given ticker"""
    
    def __init__(self):
        self.ticker = None
        self.data = None
        self.processed_data = None
    
    def validate_ticker(self, ticker_symbol):
        """
        Validates if the ticker exists and has data
        
        Args:
            ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA')
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            ticker_symbol = ticker_symbol.upper().strip()
            stock = yf.Ticker(ticker_symbol)
            
            # Try to get some basic info to validate
            info = stock.info
            
            # Check if we can get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=10)
            test_data = stock.history(start=start_date, end=end_date)
            
            if test_data.empty:
                print(f"❌ No data available for ticker: {ticker_symbol}")
                return False
            
            self.ticker = ticker_symbol
            print(f"✅ Ticker '{ticker_symbol}' is valid!")
            return True
            
        except Exception as e:
            print(f"❌ Error validating ticker '{ticker_symbol}': {str(e)}")
            return False
    
    def fetch_data(self, years=5):
        """
        Fetches historical data for the validated ticker
        
        Args:
            years (int): Number of years of historical data to fetch
            
        Returns:
            pd.DataFrame: Raw stock data
        """
        if not self.ticker:
            raise ValueError("No valid ticker set. Please validate a ticker first.")
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years*365)
            
            stock = yf.Ticker(self.ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data returned for {self.ticker}")
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            self.data = data
            print(f"✅ Fetched {len(data)} days of data for {self.ticker}")
            print(f"   Date range: {data['Date'].min().date()} to {data['Date'].max().date()}")
            
            return data
            
        except Exception as e:
            print(f"❌ Error fetching data: {str(e)}")
            raise
    
    def process_data(self):
        """
        Processes the raw data to calculate volatility and volume metrics
        
        Returns:
            pd.DataFrame: Processed data with calculated metrics
        """
        if self.data is None:
            raise ValueError("No data to process. Please fetch data first.")
        
        df = self.data.copy()
        
        # Calculate Volatility = (High - Low) / Close
        df['Volatility'] = (df['High'] - df['Low']) / df['Close']
        
        # Calculate Volume_delta = Volume_today - Volume_yesterday
        df['Volume_delta'] = df['Volume'].diff()
        
        # Calculate Volatility_delta = Volatility_today - Volatility_yesterday
        df['Volatility_delta'] = df['Volatility'].diff()
        
        # Drop rows with NaN values (first row will have NaN for diff operations)
        df = df.dropna()
        
        self.processed_data = df
        print(f"✅ Data processed successfully")
        print(f"   {len(df)} valid data points after processing")
        
        return df
    
    def get_xy_data(self):
        """
        Extracts X (Volume_delta) and y (Volatility_delta) for modeling
        
        Returns:
            tuple: (X, y) as numpy arrays
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Please process data first.")
        
        X = self.processed_data['Volume_delta'].values
        y = self.processed_data['Volatility_delta'].values
        
        return X, y
    
    def get_full_pipeline(self, ticker_symbol, years=5):
        """
        Convenience method to run full pipeline: validate -> fetch -> process
        
        Args:
            ticker_symbol (str): Stock ticker symbol
            years (int): Years of historical data
            
        Returns:
            tuple: (X, y) for modeling, or (None, None) if failed
        """
        if not self.validate_ticker(ticker_symbol):
            return None, None
        
        try:
            self.fetch_data(years)
            self.process_data()
            return self.get_xy_data()
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            return None, None


# Example usage
if __name__ == "__main__":
    validator = TickerValidator()
    
    # Test with a valid ticker
    ticker = input("Enter ticker symbol (e.g., TSLA, AAPL): ").strip()
    
    X, y = validator.get_full_pipeline(ticker, years=5)
    
    if X is not None:
        print(f"\n📊 Ready for modeling:")
        print(f"   X (Volume_delta): {len(X)} points")
        print(f"   y (Volatility_delta): {len(y)} points")'''

"""
Ticker Validation and Data Fetching
Validates ticker symbols and fetches historical stock data
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class TickerValidator:
    """Validates and fetches stock data for a given ticker"""
    
    def __init__(self):
        self.ticker = None
        self.data = None
        self.processed_data = None
    
    def validate_ticker(self, ticker_symbol):
        """
        Validates if the ticker exists and has data
        
        Args:
            ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA')
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            ticker_symbol = ticker_symbol.upper().strip()
            stock = yf.Ticker(ticker_symbol)
            
            # Try to get some basic info to validate
            info = stock.info
            
            # Check if we can get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=10)
            test_data = stock.history(start=start_date, end=end_date)
            
            if test_data.empty:
                return False
            
            self.ticker = ticker_symbol
            return True
            
        except Exception as e:
            return False
    
    def fetch_data(self, years=5):
        """
        Fetches historical data for the validated ticker
        
        Args:
            years (int): Number of years of historical data to fetch
            
        Returns:
            pd.DataFrame: Raw stock data
        """
        if not self.ticker:
            raise ValueError("No valid ticker set. Please validate a ticker first.")
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years*365)
            
            stock = yf.Ticker(self.ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data returned for {self.ticker}")
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            self.data = data
            return data
            
        except Exception as e:
            print(f"❌ Error fetching data: {str(e)}")
            raise
    
    def process_data(self):
        """
        Processes the raw data to calculate volatility and volume metrics
        
        Returns:
            pd.DataFrame: Processed data with calculated metrics
        """
        if self.data is None:
            raise ValueError("No data to process. Please fetch data first.")
        
        df = self.data.copy()
        
        # Calculate Volatility = (High - Low) / Close
        df['Volatility'] = (df['High'] - df['Low']) / df['Close']
        
        # Calculate Volume_delta = Volume_today - Volume_yesterday
        df['Volume_delta'] = df['Volume'].diff()
        
        # Calculate Volatility_delta = Volatility_today - Volatility_yesterday
        df['Volatility_delta'] = df['Volatility'].diff()
        
        # Drop rows with NaN values (first row will have NaN for diff operations)
        df = df.dropna()
        
        self.processed_data = df
        return df
    
    def get_xy_data(self):
        """
        Extracts X (Volume_delta) and y (Volatility_delta) for modeling
        
        Returns:
            tuple: (X, y) as numpy arrays
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Please process data first.")
        
        X = self.processed_data['Volume_delta'].values
        y = self.processed_data['Volatility_delta'].values
        
        return X, y
    
    def get_full_pipeline(self, ticker_symbol, years=5):
        """
        Convenience method to run full pipeline: validate -> fetch -> process
        
        Args:
            ticker_symbol (str): Stock ticker symbol
            years (int): Years of historical data
            
        Returns:
            tuple: (X, y) for modeling, or (None, None) if failed
        """
        if not self.validate_ticker(ticker_symbol):
            return None, None
        
        try:
            self.fetch_data(years)
            self.process_data()
            return self.get_xy_data()
        except Exception as e:
            return None, None


# Example usage
if __name__ == "__main__":
    validator = TickerValidator()
    
    # Test with a valid ticker
    ticker = input("Enter ticker symbol (e.g., TSLA, AAPL): ").strip()
    
    X, y = validator.get_full_pipeline(ticker, years=5)
    
    if X is not None:
        print(f"\n📊 Ready for modeling:")
        print(f"   X (Volume_delta): {len(X)} points")
        print(f"   y (Volatility_delta): {len(y)} points")