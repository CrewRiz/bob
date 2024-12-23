from datetime import datetime, timezone, timedelta
import pytz

class TimeUtils:
    """Utility class for handling time-related operations consistently across the system"""
    
    @staticmethod
    def get_current_time():
        """Returns the current time in UTC"""
        return datetime.now(timezone.utc)
    
    @staticmethod
    def get_local_time():
        """Returns the current local time"""
        return datetime.now()
    
    @staticmethod
    def format_timestamp(dt):
        """Formats a datetime object into ISO format string"""
        return dt.isoformat()
    
    @staticmethod
    def parse_timestamp(timestamp_str):
        """Parses an ISO format timestamp string into a datetime object"""
        return datetime.fromisoformat(timestamp_str)
    
    @staticmethod
    def get_time_diff(time1, time2):
        """Returns the time difference between two datetime objects in seconds"""
        return abs((time1 - time2).total_seconds())
    
    @staticmethod
    def is_expired(timestamp, expiry_seconds):
        """Checks if a given timestamp has expired based on expiry_seconds"""
        if isinstance(timestamp, str):
            timestamp = TimeUtils.parse_timestamp(timestamp)
        current_time = TimeUtils.get_current_time()
        return TimeUtils.get_time_diff(current_time, timestamp) > expiry_seconds
