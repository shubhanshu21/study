"""
Order manager for executing and tracking orders.
"""
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

class OrderManager:
    """
    Order manager for executing and tracking orders.
    """
    
    def __init__(self, broker, config, db_service=None):
        """
        Initialize the order manager
        
        Args:
            broker: Broker instance
            config: Application configuration
            db_service (optional): Database service instance
        """
        self.broker = broker
        self.config = config
        self.db_service = db_service
        self.orders = {}  # Order ID -> order details
        self.pending_orders = {}  # Order ID -> order details
        
        logger.info("Order manager initialized")
    
    def place_order(self, symbol, order_type, quantity, price=None, trigger_price=None, tag=None):
        """
        Place an order
        
        Args:
            symbol (str): Trading symbol
            order_type (str): Order type (BUY, SELL)
            quantity (int): Number of shares
            price (float, optional): Limit price
            trigger_price (float, optional): Trigger price
            tag (str, optional): Order tag
            
        Returns:
            str: Order ID if successful, None otherwise
        """
        try:
            # Validate inputs
            if not symbol or not order_type or quantity <= 0:
                logger.error(f"Invalid order parameters: symbol={symbol}, order_type={order_type}, quantity={quantity}")
                return None
                
            # Normalize order type
            order_type = order_type.upper()
            if order_type not in ['BUY', 'SELL']:
                logger.error(f"Invalid order type: {order_type}")
                return None
                
            # Generate order ID
            order_id = str(uuid.uuid4())
            
            # Create order object
            order = {
                'id': order_id,
                'symbol': symbol,
                'type': order_type,
                'quantity': quantity,
                'price': price,
                'trigger_price': trigger_price,
                'tag': tag,
                'status': 'PENDING',
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'filled_quantity': 0,
                'average_price': 0,
                'transaction_costs': 0
            }
            
            # Store in pending orders
            self.pending_orders[order_id] = order
            
            # Place order with broker
            broker_order_id = self.broker.place_order(
                symbol=symbol,
                transaction_type=order_type,
                quantity=quantity,
                order_type='MARKET' if price is None else 'LIMIT',
                price=price,
                trigger_price=trigger_price,
                tag=tag
            )
            
            if not broker_order_id:
                logger.error(f"Failed to place order for {symbol}")
                del self.pending_orders[order_id]
                return None
                
            # Update order with broker order ID
            order['broker_order_id'] = broker_order_id
            
            # Move to orders
            self.orders[order_id] = order
            del self.pending_orders[order_id]
            
            # Record order in database
            if self.db_service:
                trade_data = {
                    'symbol': symbol,
                    'order_id': order_id,
                    'transaction_type': order_type,
                    'price': price or 0,
                    'quantity': quantity,
                    'timestamp': datetime.now().isoformat()
                }
                self.db_service.record_trade(trade_data)
                
            logger.info(f"Order placed: {order_type} {quantity} {symbol} - Order ID: {order_id}")
            
            return order_id
            
        except Exception as e:
            logger.exception(f"Error placing order: {str(e)}")
            return None
    
    def cancel_order(self, order_id):
        """
        Cancel an order
        
        Args:
            order_id (str): Order ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if order exists
            if order_id not in self.orders:
                logger.error(f"Order not found: {order_id}")
                return False
                
            # Get order
            order = self.orders[order_id]
            
            # Check if order can be cancelled
            if order['status'] in ['COMPLETE', 'CANCELLED', 'REJECTED']:
                logger.error(f"Cannot cancel order with status {order['status']}")
                return False
                
            # Cancel order with broker
            success = self.broker.cancel_order(order['broker_order_id'])
            
            if not success:
                logger.error(f"Failed to cancel order: {order_id}")
                return False
                
            # Update order status
            order['status'] = 'CANCELLED'
            order['updated_at'] = datetime.now()
            
            logger.info(f"Order cancelled: {order_id}")
            
            return True
            
        except Exception as e:
            logger.exception(f"Error cancelling order: {str(e)}")
            return False
    
    def update_order_status(self, order_id, status, filled_quantity=None, average_price=None):
        """
        Update order status
        
        Args:
            order_id (str): Order ID
            status (str): New status
            filled_quantity (int, optional): Filled quantity
            average_price (float, optional): Average fill price
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if order exists
            if order_id not in self.orders:
                logger.error(f"Order not found: {order_id}")
                return False
                
            # Get order
            order = self.orders[order_id]
            
            # Update order status
            order['status'] = status
            order['updated_at'] = datetime.now()
            
            # Update filled quantity and average price if provided
            if filled_quantity is not None:
                order['filled_quantity'] = filled_quantity
            
            if average_price is not None:
                order['average_price'] = average_price
                
            logger.info(f"Order {order_id} status updated to {status}")
            
            return True
            
        except Exception as e:
            logger.exception(f"Error updating order status: {str(e)}")
            return False
    
    def get_order(self, order_id):
        """
        Get order details
        
        Args:
            order_id (str): Order ID
            
        Returns:
            dict: Order details if found, None otherwise
        """
        try:
            # Check if order exists
            if order_id not in self.orders:
                # Check pending orders
                if order_id in self.pending_orders:
                    return self.pending_orders[order_id]
                    
                logger.warning(f"Order not found: {order_id}")
                return None
                
            return self.orders[order_id]
            
        except Exception as e:
            logger.exception(f"Error getting order: {str(e)}")
            return None
    
    def get_orders(self, symbol=None, status=None, tag=None):
        """
        Get orders filtered by symbol, status, or tag
        
        Args:
            symbol (str, optional): Filter by symbol
            status (str, optional): Filter by status
            tag (str, optional): Filter by tag
            
        Returns:
            list: List of matching orders
        """
        try:
            # Start with all orders
            all_orders = list(self.orders.values()) + list(self.pending_orders.values())
            
            # Apply filters
            filtered_orders = all_orders
            
            if symbol:
                filtered_orders = [o for o in filtered_orders if o['symbol'] == symbol]
                
            if status:
                filtered_orders = [o for o in filtered_orders if o['status'] == status]
                
            if tag:
                filtered_orders = [o for o in filtered_orders if o['tag'] == tag]
                
            return filtered_orders
            
        except Exception as e:
            logger.exception(f"Error getting orders: {str(e)}")
            return []
    
    def sync_orders(self):
        """
        Sync orders with broker to get latest status
        
        Returns:
            int: Number of orders updated
        """
        try:
            # Get orders from broker
            broker_orders = self.broker.get_orders()
            
            if not broker_orders:
                logger.warning("No orders returned from broker")
                return 0
                
            # Track number of updates
            updates = 0
            
            # Update our order records
            for order in self.orders.values():
                if 'broker_order_id' not in order:
                    continue
                    
                # Find matching broker order
                for broker_order in broker_orders:
                    if broker_order.get('order_id') == order['broker_order_id']:
                        # Update status
                        broker_status = broker_order.get('status')
                        if broker_status and broker_status != order['status']:
                            order['status'] = broker_status
                            order['updated_at'] = datetime.now()
                            updates += 1
                            
                        # Update filled quantity
                        broker_filled = broker_order.get('filled_quantity')
                        if broker_filled is not None and broker_filled != order['filled_quantity']:
                            order['filled_quantity'] = broker_filled
                            updates += 1
                            
                        # Update average price
                        broker_avg_price = broker_order.get('average_price')
                        if broker_avg_price is not None and broker_avg_price != order['average_price']:
                            order['average_price'] = broker_avg_price
                            updates += 1
                            
                        break
            
            logger.info(f"Synced orders with broker, {updates} updates")
            
            return updates
            
        except Exception as e:
            logger.exception(f"Error syncing orders: {str(e)}")
            return 0
    
    def place_market_order(self, symbol, order_type, quantity, tag=None):
        """
        Place a market order
        
        Args:
            symbol (str): Trading symbol
            order_type (str): Order type (BUY, SELL)
            quantity (int): Number of shares
            tag (str, optional): Order tag
            
        Returns:
            str: Order ID if successful, None otherwise
        """
        return self.place_order(symbol, order_type, quantity, price=None, tag=tag)
    
    def place_limit_order(self, symbol, order_type, quantity, price, tag=None):
        """
        Place a limit order
        
        Args:
            symbol (str): Trading symbol
            order_type (str): Order type (BUY, SELL)
            quantity (int): Number of shares
            price (float): Limit price
            tag (str, optional): Order tag
            
        Returns:
            str: Order ID if successful, None otherwise
        """
        return self.place_order(symbol, order_type, quantity, price=price, tag=tag)
    
    def place_stop_loss_order(self, symbol, order_type, quantity, trigger_price, price=None, tag=None):
        """
        Place a stop loss order
        
        Args:
            symbol (str): Trading symbol
            order_type (str): Order type (BUY, SELL)
            quantity (int): Number of shares
            trigger_price (float): Trigger price
            price (float, optional): Limit price
            tag (str, optional): Order tag
            
        Returns:
            str: Order ID if successful, None otherwise
        """
        return self.place_order(
            symbol=symbol,
            order_type=order_type,
            quantity=quantity,
            price=price,
            trigger_price=trigger_price,
            tag=tag or "STOPLOSS"
        )