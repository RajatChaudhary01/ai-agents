import datetime

import pytz
from smolagents import tool


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.

    Args:
        timezone: A valid timezone string (e.g., 'America/New_York', 'Europe/London', 'Asia/Kolkata').
    """
    name = timezone.strip() if isinstance(timezone, str) else timezone
    try:
        tz = pytz.timezone(name)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {name} is: {local_time}"
    except pytz.UnknownTimeZoneError:
        return (
            f"Error fetching time for timezone '{timezone}': unknown timezone. "
            "Use an IANA name such as 'Asia/Tokyo', 'Europe/London' or 'America/New_York'."
        )
    except Exception as e:  # noqa: BLE001
        return f"Error fetching time for timezone '{timezone}': {str(e)}"  # noqa: RUF010
