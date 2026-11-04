"""World clock tool: the current local time for any IANA timezone name."""

from datetime import datetime

import pytz
from smolagents import tool

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """Returns the current local date and time in the given timezone.

    Args:
        timezone: An IANA timezone name such as 'Asia/Kolkata', 'Europe/London' or 'America/New_York'.
    """
    zone_name = timezone.strip() if isinstance(timezone, str) else timezone
    try:
        now = datetime.now(pytz.timezone(zone_name))
    except pytz.UnknownTimeZoneError:
        return (
            f"Error fetching time for timezone '{timezone}': unknown timezone. "
            "Use an IANA name such as 'Asia/Tokyo', 'Europe/London' or 'America/New_York'."
        )
    except Exception as exc:  # noqa: BLE001 - e.g. a non-string argument from the model
        return f"Error fetching time for timezone '{timezone}': {exc}"
    return f"The current local time in {zone_name} is: {now.strftime(TIME_FORMAT)}"
