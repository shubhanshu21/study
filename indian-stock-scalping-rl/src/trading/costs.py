"""
Trading costs calculator for Indian markets.
Handles STT, brokerage, GST, and other exchange fees.
"""

from typing import Dict
from ..constants import IndianMarketConstants

class TradingCosts:
    """Class for calculating trading costs in the Indian market"""
    
    @staticmethod
    def calculate_costs(price: float, quantity: int, transaction_type: str) -> Dict[str, float]:
        """
        Calculate all trading costs for a transaction
        
        Args:
            price: Stock price
            quantity: Number of shares
            transaction_type: 'buy' or 'sell'
            
        Returns:
            Dictionary with breakdown of costs
        """
        value = price * quantity
        
        # Base costs
        stt = 0
        if transaction_type.lower() == 'sell':
            stt = value * IndianMarketConstants.STT_RATE
            
        exchange_charge = value * IndianMarketConstants.EXCHANGE_TXN_CHARGE
        sebi_charges = value * IndianMarketConstants.SEBI_CHARGES
        
        # Brokerage
        brokerage = min(value * IndianMarketConstants.BROKERAGE_RATE, 
                       IndianMarketConstants.MAX_BROKERAGE_PER_ORDER)
        
        # Stamp duty only on buy
        stamp_duty = 0
        if transaction_type.lower() == 'buy':
            stamp_duty = value * IndianMarketConstants.STAMP_DUTY
            
        # GST on brokerage and exchange charges
        gst = (brokerage + exchange_charge) * IndianMarketConstants.GST_RATE
        
        # Total costs
        total_costs = stt + exchange_charge + sebi_charges + brokerage + stamp_duty + gst
        
        return {
            'stt': stt,
            'exchange_charge': exchange_charge,
            'sebi_charges': sebi_charges,
            'brokerage': brokerage,
            'stamp_duty': stamp_duty,
            'gst': gst,
            'total': total_costs
        }