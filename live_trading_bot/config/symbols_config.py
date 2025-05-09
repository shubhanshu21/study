"""
Configuration for trading symbols.
"""
import os
import pandas as pd
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SymbolsConfig:
    """Configuration for trading symbols"""
    
    def __init__(self):
        # Default symbols if not specified in environment
        self.DEFAULT_SYMBOLS = ["ASIANPAINT", "BAJFINANCE"]
        
        # Load symbols from environment
        self.symbols = os.getenv("SYMBOLS", "ASIANPAINT,BAJFINANCE").split(",")
        
        # Symbol properties
        self.properties = {}
        self._load_symbol_properties()
        
        logger.info(f"Loaded configuration for {len(self.symbols)} symbols")
    
    def _load_symbol_properties(self):
        """Load symbol properties from config file or initialize defaults"""
        try:
            # Check if symbol properties file exists
            properties_file = os.path.join(os.path.dirname(__file__), "symbol_properties.csv")
            
            if os.path.exists(properties_file):
                df = pd.read_csv(properties_file)
                
                for _, row in df.iterrows():
                    symbol = row["symbol"]
                    
                    if symbol in self.symbols:
                        # Load properties
                        self.properties[symbol] = {
                            "lot_size": int(row.get("lot_size", 1)),
                            "tick_size": float(row.get("tick_size", 0.05)),
                            "max_position_size_pct": float(row.get("max_position_size_pct", 0.4)),
                            "atr_multiplier": float(row.get("atr_multiplier", 2.0)),
                            "target_multiplier": float(row.get("target_multiplier", 3.0)),
                            "instrument_token": int(row.get("instrument_token", 0)),
                            "segment": row.get("segment", "NSE"),
                            "precision": int(row.get("precision", 2))
                        }
                
                logger.info(f"Loaded properties for {len(self.properties)} symbols")
                
            else:
                # Initialize with defaults
                for symbol in self.symbols:
                    self.properties[symbol] = {
                        "lot_size": 1,  # Default lot size
                        "tick_size": 0.05,  # Default tick size
                        "max_position_size_pct": 0.4,  # Default position size
                        "atr_multiplier": 2.0,  # Default ATR multiplier
                        "target_multiplier": 3.0,  # Default target multiplier
                        "instrument_token": 0,  # Will be populated later
                        "segment": "NSE",  # Default segment
                        "precision": 2  # Default precision
                    }
                
                logger.warning("Symbol properties file not found, using defaults")
        
        except Exception as e:
            logger.exception(f"Error loading symbol properties: {str(e)}")
            
            # Initialize with defaults
            for symbol in self.symbols:
                self.properties[symbol] = {
                    "lot_size": 1,
                    "tick_size": 0.05,
                    "max_position_size_pct": 0.4,
                    "atr_multiplier": 2.0,
                    "target_multiplier": 3.0,
                    "instrument_token": 0,
                    "segment": "NSE",
                    "precision": 2
                }
    
    def get_property(self, symbol, property_name, default=None):
        """
        Get a specific property for a symbol
        
        Args:
            symbol (str): Trading symbol
            property_name (str): Property name
            default: Default value to return if property not found
            
        Returns:
            Property value if found, default otherwise
        """
        if symbol in self.properties and property_name in self.properties[symbol]:
            return self.properties[symbol][property_name]
        return default
    
    def update_property(self, symbol, property_name, value):
        """
        Update a property for a symbol
        
        Args:
            symbol (str): Trading symbol
            property_name (str): Property name
            value: Property value
        """
        if symbol not in self.properties:
            self.properties[symbol] = {}
            
        self.properties[symbol][property_name] = value
    
    def get_all_symbols(self):
        """Get all configured symbols"""
        return self.symbols
    
    def is_valid_symbol(self, symbol):
        """Check if a symbol is valid"""
        return symbol in self.symbols