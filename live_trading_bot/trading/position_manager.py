"""
Position manager for tracking trading positions.
"""
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PositionManager:
    """
    Position manager for tracking trading positions.
    """
    
    def __init__(self, broker, config, db_service=None):
        """
        Initialize the position manager
        
        Args:
            broker: Broker instance
            config: Application configuration
            db_service (optional): Database service instance
        """
        self.broker = broker
        self.config = config
        self.db_service = db_service
        self.positions = {}  # Symbol -> position details
        
        # Initialize from broker
        self.sync_positions()
        
        logger.info("Position manager initialized")
    
    def sync_positions(self):
        """
        Sync positions with broker
        
        Returns:
            int: Number of positions updated
        """
        try:
            # Get positions from broker
            broker_positions = self.broker.get_positions()
            
            if not broker_positions:
                return 0
                
            # Track number of updates
            updates = 0
            
            # Create a set of existing symbols
            existing_symbols = set(self.positions.keys())
            
            # Process broker positions
            for symbol, pos_data in broker_positions.items():
                # Skip if not a valid position
                quantity = pos_data.get('quantity', 0)
                if quantity <= 0:
                    continue
                    
                # Check if we already have this position
                if symbol in self.positions:
                    position = self.positions[symbol]
                    
                    # Update position data
                    position['quantity'] = quantity
                    position['average_price'] = pos_data.get('average_price', 0)
                    position['last_price'] = pos_data.get('last_price', 0)
                    position['updated_at'] = datetime.now()
                    
                    updates += 1
                else:
                    # Create new position
                    self.positions[symbol] = {
                        'symbol': symbol,
                        'quantity': quantity,
                        'average_price': pos_data.get('average_price', 0),
                        'last_price': pos_data.get('last_price', 0),
                        'created_at': datetime.now(),
                        'updated_at': datetime.now(),
                        'stop_loss': 0,
                        'trailing_stop': 0,
                        'target_price': 0,
                        'days_in_trade': 0
                    }
                    
                    updates += 1
                    
                # Remove from existing symbols
                if symbol in existing_symbols:
                    existing_symbols.remove(symbol)
            
            # Handle positions that no longer exist
            for symbol in existing_symbols:
                # Position has been closed
                if symbol in self.positions:
                    pos_data = self.positions[symbol]
                    
                    # Record the closed position in database
                    if self.db_service:
                        trade_data = {
                            'symbol': symbol,
                            'transaction_type': 'SELL',
                            'price': pos_data.get('last_price', 0),
                            'quantity': pos_data.get('quantity', 0),
                            'timestamp': datetime.now().isoformat(),
                            'exit_type': 'sync_close'
                        }
                        self.db_service.record_trade(trade_data)
                    
                    # Remove position
                    del self.positions[symbol]
                    updates += 1
            
            logger.info(f"Synced positions with broker, {updates} updates")
            
            return updates
            
        except Exception as e:
            logger.exception(f"Error syncing positions: {str(e)}")
            return 0
    
    def get_position(self, symbol):
        """
        Get position for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            dict: Position details if found, None otherwise
        """
        try:
            # Check if position exists
            if symbol not in self.positions:
                return None
                
            return self.positions[symbol]
            
        except Exception as e:
            logger.exception(f"Error getting position: {str(e)}")
            return None
    
    def get_positions(self):
        """
        Get all positions
        
        Returns:
            dict: All positions
        """
        return self.positions
    
    def add_position(self, symbol, quantity, average_price, order_id=None):
        """
        Add a new position
        
        Args:
            symbol (str): Trading symbol
            quantity (int): Number of shares
            average_price (float): Average price
            order_id (str, optional): Order ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if position already exists
            if symbol in self.positions:
                # Update existing position
                position = self.positions[symbol]
                
                # Calculate new average price
                total_value = position['quantity'] * position['average_price']
                new_value = quantity * average_price
                new_quantity = position['quantity'] + quantity
                
                if new_quantity > 0:
                    new_average_price = (total_value + new_value) / new_quantity
                else:
                    new_average_price = average_price
                
                # Update position
                position['quantity'] = new_quantity
                position['average_price'] = new_average_price
                position['updated_at'] = datetime.now()
                
                logger.info(f"Updated position: {symbol}, Quantity: {new_quantity}, Avg. Price: {new_average_price}")
                
                return True
            else:
                # Create new position
                self.positions[symbol] = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'average_price': average_price,
                    'last_price': average_price,
                    'created_at': datetime.now(),
                    'updated_at': datetime.now(),
                    'stop_loss': 0,
                    'trailing_stop': 0,
                    'target_price': 0,
                    'days_in_trade': 0,
                    'order_id': order_id
                }
                
                logger.info(f"Added new position: {symbol}, Quantity: {quantity}, Price: {average_price}")
                
                return True
                
        except Exception as e:
            logger.exception(f"Error adding position: {str(e)}")
            return False
    
    def update_position(self, symbol, **kwargs):
        """
        Update position details
        
        Args:
            symbol (str): Trading symbol
            **kwargs: Position attributes to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if position exists
            if symbol not in self.positions:
                logger.error(f"Position not found: {symbol}")
                return False
                
            # Get position
            position = self.positions[symbol]
            
            # Update position attributes
            for key, value in kwargs.items():
                position[key] = value
                
            # Update timestamp
            position['updated_at'] = datetime.now()
            
            logger.info(f"Updated position details for {symbol}")
            
            return True
            
        except Exception as e:
            logger.exception(f"Error updating position: {str(e)}")
            return False
    
    def remove_position(self, symbol):
        """
        Remove a position
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if position exists
            if symbol not in self.positions:
                logger.error(f"Position not found: {symbol}")
                return False
                
            # Remove position
            del self.positions[symbol]
            
            logger.info(f"Removed position: {symbol}")
            
            return True
            
        except Exception as e:
            logger.exception(f"Error removing position: {str(e)}")
            return False
    
    def update_stop_loss(self, symbol, stop_loss):
        """
        Update stop loss for a position
        
        Args:
            symbol (str): Trading symbol
            stop_loss (float): Stop loss price
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.update_position(symbol, stop_loss=stop_loss)
    
    def update_trailing_stop(self, symbol, trailing_stop):
        """
        Update trailing stop for a position
        
        Args:
            symbol (str): Trading symbol
            trailing_stop (float): Trailing stop price
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.update_position(symbol, trailing_stop=trailing_stop)
    
    def update_target_price(self, symbol, target_price):
        """
        Update target price for a position
        
        Args:
            symbol (str): Trading symbol
            target_price (float): Target price
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.update_position(symbol, target_price=target_price)
    
    def calculate_position_value(self, symbol=None):
        """
        Calculate position value
        
        Args:
            symbol (str, optional): Trading symbol
            
        Returns:
            float: Position value
        """
        try:
            # Calculate for specific symbol
            if symbol:
                # Check if position exists
                if symbol not in self.positions:
                    return 0
                    
                # Get position
                position = self.positions[symbol]
                
                # Calculate value
                return position['quantity'] * position['last_price']
            
            # Calculate for all positions
            total_value = 0
            for position in self.positions.values():
                total_value += position['quantity'] * position['last_price']
                
            return total_value
            
        except Exception as e:
            logger.exception(f"Error calculating position value: {str(e)}")
            return 0
    
    def update_position_prices(self):
        """
        Update position prices from broker
        
        Returns:
            int: Number of positions updated
        """
        try:
            # Track number of updates
            updates = 0
            
            # Update each position
            for symbol, position in self.positions.items():
                # Get current price from broker
                current_price = self.broker.get_ltp(symbol)
                
                if not current_price:
                    continue
                    
                # Update position
                position['last_price'] = current_price
                position['updated_at'] = datetime.now()
                
                updates += 1
            
            logger.info(f"Updated prices for {updates} positions")
            
            return updates
            
        except Exception as e:
            logger.exception(f"Error updating position prices: {str(e)}")
            return 0
    
    def check_stop_losses(self):
        """
        Check stop losses for all positions
        
        Returns:
            list: List of symbols that hit stop loss
        """
        try:
            # Track symbols that hit stop loss
            hit_stop = []
            
            # Check each position
            for symbol, position in self.positions.items():
                # Skip if no stop loss
                if not position.get('trailing_stop'):
                    continue
                    
                # Get current price
                current_price = position.get('last_price')
                
                if not current_price:
                    continue
                    
                # Check if stop loss hit
                if current_price <= position['trailing_stop']:
                    logger.info(f"Stop loss hit for {symbol}: {position['trailing_stop']}")
                    hit_stop.append(symbol)
            
            return hit_stop
            
        except Exception as e:
            logger.exception(f"Error checking stop losses: {str(e)}")
            return []
    
    def check_targets(self):
        """
        Check targets for all positions
        
        Returns:
            list: List of symbols that hit target
        """
        try:
            # Track symbols that hit target
            hit_target = []
            
            # Check each position
            for symbol, position in self.positions.items():
                # Skip if no target
                if not position.get('target_price'):
                    continue
                    
                # Get current price
                current_price = position.get('last_price')
                
                if not current_price:
                    continue
                    
                # Check if target hit
                if current_price >= position['target_price']:
                    logger.info(f"Target hit for {symbol}: {position['target_price']}")
                    hit_target.append(symbol)
            
            return hit_target
            
        except Exception as e:
            logger.exception(f"Error checking targets: {str(e)}")
            return []