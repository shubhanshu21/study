import pytz
import pandas as pd
from datetime import datetime, time
from holidays import India

# Market timing settings
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30
AVOID_FIRST_MINUTES = 15  # Avoid first 15 minutes of trading (high volatility)
AVOID_LAST_MINUTES = 15   # Avoid last 15 minutes of trading (high volatility)

# Timezone settings
TIMEZONE = pytz.timezone('Asia/Kolkata')

class MarketConfig:
    """Configuration settings for Indian stock market trading"""
    
    def __init__(self):
        self.holidays = self._load_holidays()
        self.pre_market_start = time(9, 0)
        self.market_open = time(MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE)
        self.market_close = time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)
        self.post_market_end = time(16, 0)
        self.timezone = TIMEZONE
        
    def _load_holidays(self):
        """Load Indian market holidays"""
        # Get the India holidays from the holidays package
        india_holidays = India()
        # Filter only the NSE holidays (market holidays)
        # This is a simplification, in reality we would need to check specifically for NSE holidays
        return india_holidays
    
    def is_market_open(self, check_time=None):
        """Check if the market is currently open"""
        if check_time is None:
            check_time = datetime.now(self.timezone)
        
        # Check if today is a holiday
        if check_time.date() in self.holidays:
            return False
        
        # Check if today is a weekend
        if check_time.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
            return False
        
        # Check if current time is within market hours
        current_time = check_time.time()
        return self.market_open <= current_time < self.market_close
    
    def is_pre_market(self, check_time=None):
        """Check if it's pre-market hours"""
        if check_time is None:
            check_time = datetime.now(self.timezone)
        
        # Check if today is a holiday or weekend
        if check_time.date() in self.holidays or check_time.weekday() >= 5:
            return False
        
        # Check if current time is pre-market
        current_time = check_time.time()
        return self.pre_market_start <= current_time < self.market_open
    
    def is_post_market(self, check_time=None):
        """Check if it's post-market hours"""
        if check_time is None:
            check_time = datetime.now(self.timezone)
        
        # Check if today is a holiday or weekend
        if check_time.date() in self.holidays or check_time.weekday() >= 5:
            return False
        
        # Check if current time is post-market
        current_time = check_time.time()
        return self.market_close <= current_time < self.post_market_end
    
    def time_to_next_market_open(self, check_time=None):
        """Calculate time until next market open"""
        if check_time is None:
            check_time = datetime.now(self.timezone)
        
        # If market is already open, return 0
        if self.is_market_open(check_time):
            return 0
        
        # Start with tomorrow if we're past market close today
        next_day = check_time.date()
        if check_time.time() >= self.market_open:
            next_day = (check_time + pd.Timedelta(days=1)).date()
        
        # Find the next trading day
        while True:
            # Check if it's a weekend or holiday
            next_day_date = datetime.combine(next_day, self.market_open)
            next_day_date = self.timezone.localize(next_day_date)
            
            if next_day_date.weekday() < 5 and next_day not in self.holidays:
                break
                
            next_day = (next_day_date + pd.Timedelta(days=1)).date()
        
        # Calculate time difference
        next_market_open = datetime.combine(next_day, self.market_open)
        next_market_open = self.timezone.localize(next_market_open)
        
        return (next_market_open - check_time).total_seconds()
    
    def get_remaining_market_time(self, check_time=None):
        """Calculate remaining time in the current market session"""
        if check_time is None:
            check_time = datetime.now(self.timezone)
        
        # If market is not open, return 0
        if not self.is_market_open(check_time):
            return 0
        
        # Calculate time to market close
        market_close_today = datetime.combine(check_time.date(), self.market_close)
        market_close_today = self.timezone.localize(market_close_today)
        
        return (market_close_today - check_time).total_seconds()
    
    def should_avoid_high_volatility(self, check_time=None):
        """Check if current time is in high volatility period (open/close)"""
        if check_time is None:
            check_time = datetime.now(self.timezone)
        
        # If market is not open, not a high volatility period
        if not self.is_market_open(check_time):
            return False
        
        current_time = check_time.time()
        
        # Check if within first few minutes of market open
        market_open_plus = time(MARKET_OPEN_HOUR, 
                                MARKET_OPEN_MINUTE + AVOID_FIRST_MINUTES)
        
        # Check if within last few minutes before market close
        market_close_minus = time(MARKET_CLOSE_HOUR, 
                                 MARKET_CLOSE_MINUTE - AVOID_LAST_MINUTES)
        
        return (self.market_open <= current_time < market_open_plus or 
                market_close_minus <= current_time < self.market_close)