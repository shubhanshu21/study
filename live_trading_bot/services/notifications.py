"""
Notification service for trading alerts.
"""
import logging
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

logger = logging.getLogger(__name__)

class NotificationService:
    """
    Notification service for trading alerts.
    Supports email and Telegram notifications.
    """
    
    def __init__(self, config):
        """
        Initialize notification service
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.email_enabled = config.ENABLE_EMAIL_NOTIFICATIONS
        self.telegram_enabled = config.ENABLE_TELEGRAM_NOTIFICATIONS
        
        logger.info(f"Notification service initialized. Email: {self.email_enabled}, Telegram: {self.telegram_enabled}")
    
    def send_alert(self, subject, message):
        """
        Send an alert via configured channels
        
        Args:
            subject (str): Alert subject
            message (str): Alert message
            
        Returns:
            bool: True if at least one notification was sent, False otherwise
        """
        success = False
        
        if self.email_enabled:
            success = self._send_email(subject, message) or success
            
        if self.telegram_enabled:
            success = self._send_telegram(subject, message) or success
            
        # Always log the alert
        logger.info(f"ALERT: {subject}\n{message}")
            
        return success
    
    def _send_email(self, subject, message):
        """
        Send an email alert
        
        Args:
            subject (str): Email subject
            message (str): Email body
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.email_enabled:
            return False
            
        # Check required config
        if not self.config.EMAIL_USERNAME or not self.config.EMAIL_PASSWORD or not self.config.EMAIL_RECIPIENTS:
            logger.error("Email notification failed: Missing configuration")
            return False
            
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config.EMAIL_USERNAME
            msg['To'] = ', '.join(self.config.EMAIL_RECIPIENTS)
            msg['Subject'] = f"RL Trading Bot: {subject}"
            
            # Add timestamp
            timestamp = datetime.now(self.config.TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
            body = f"{message}\n\nSent at: {timestamp}"
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Connect to server
            server = smtplib.SMTP(self.config.EMAIL_SERVER, self.config.EMAIL_PORT)
            
            if self.config.EMAIL_USE_TLS:
                server.starttls()
                
            server.login(self.config.EMAIL_USERNAME, self.config.EMAIL_PASSWORD)
            
            # Send email
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {subject}")
            return True
            
        except Exception as e:
            logger.exception(f"Email notification failed: {str(e)}")
            return False
    
    def _send_telegram(self, subject, message):
        """
        Send a Telegram alert
        
        Args:
            subject (str): Message subject
            message (str): Message body
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.telegram_enabled:
            return False
            
        # Check required config
        if not self.config.TELEGRAM_BOT_TOKEN or not self.config.TELEGRAM_CHAT_ID:
            logger.error("Telegram notification failed: Missing configuration")
            return False
            
        try:
            # Format message
            timestamp = datetime.now(self.config.TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
            text = f"*{subject}*\n\n{message}\n\nSent at: {timestamp}"
            
            # Send message
            url = f"https://api.telegram.org/bot{self.config.TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                "chat_id": self.config.TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, data=data)
            
            if response.status_code == 200:
                logger.info(f"Telegram alert sent: {subject}")
                return True
            else:
                logger.error(f"Telegram notification failed: Status code {response.status_code}, Response: {response.text}")
                return False
                
        except Exception as e:
            logger.exception(f"Telegram notification failed: {str(e)}")
            return False