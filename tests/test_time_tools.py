import re

from ai_agents.tools import get_current_time_in_timezone

TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


def test_valid_timezone():
    result = get_current_time_in_timezone(timezone="Asia/Tokyo")
    assert result.startswith("The current local time in Asia/Tokyo is:")
    assert TIMESTAMP.search(result)


def test_invalid_timezone_returns_error_message():
    result = get_current_time_in_timezone(timezone="Mars/Olympus_Mons")
    assert result.startswith("Error fetching time for timezone 'Mars/Olympus_Mons'")
