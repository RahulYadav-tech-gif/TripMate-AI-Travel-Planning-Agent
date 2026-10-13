"""Weather forecast tool for TripMate.

Fetches forecasts from Open-Meteo, parsing natural date references
and validating against the 15-day forecast horizon.
"""

from datetime import date, datetime, timedelta
import logging
import math
import re
import time
from typing import Any, Dict, Optional, Tuple
import requests

logger = logging.getLogger(__name__)

# Known coordinates cache for primary assessment destinations
KNOWN_CITY_COORDINATES: Dict[str, Tuple[float, float]] = {
    "tokyo": (35.6762, 139.6503),
    "bangkok": (13.7563, 100.5018),
    "barcelona": (41.3851, 2.1734),
    "reykjavik": (64.1466, -21.9426),
}

# WMO Weather Interpretation Codes (Open-Meteo)
WMO_CODE_DESCRIPTIONS: Dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

WEEKDAYS_FULL = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
WEEKDAYS_ABBR = {
    "mon": 0, "tue": 1, "wed": 2, "thu": 3,
    "fri": 4, "sat": 5, "sun": 6,
}

OPEN_METEO_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# Open-Meteo provides a 16-day window (today + 15 days ahead).
MAX_FORECAST_HORIZON_DAYS = 15


def parse_target_date(
    date_or_month_str: str,
    ref_date: Optional[date] = None,
) -> Tuple[Optional[date], Optional[str], Optional[str]]:
    """Parse a date_or_month string relative to a reference date.

    Rules for weekday references:
    - 'this <weekday>':
        - If target day is today (diff == 0): resolves to today (0 days).
        - If target day is later in current calendar week: resolves to that day (diff days).
        - If target day already occurred earlier this calendar week: resolves to the upcoming occurrence (diff days).
    - 'next <weekday>':
        - Consistently targets the weekday in the subsequent calendar week.
        - If target day == current day: resolves to 7 days ahead.
        - If target day > current day: resolves to that weekday in the next week (diff + 7 days).
        - If target day < current day: the next calendar week's occurrence is already (diff days) ahead.
    - Plain '<weekday>':
        - If target day is today: resolves to today (0 days).
        - Otherwise: resolves to the earliest upcoming occurrence (diff days).

    Returns:
        (parsed_date, error_status, error_message)
    """
    if ref_date is None:
        ref_date = date.today()

    raw = date_or_month_str.strip().lower()

    if not raw:
        return None, "ambiguous_input", "Date or month reference cannot be empty."

    # 1. Check for standalone month names (e.g., 'December', 'July')
    words = set(re.findall(r"\b[a-zA-Z]+\b", raw))
    for word in words:
        if word in MONTH_NAMES:
            # Check if an explicit day number is present (e.g. 'October 25', 'Dec 15')
            has_day_num = bool(re.search(r"\b\d{1,2}\b", raw))
            if not has_day_num:
                return (
                    None,
                    "unavailable",
                    f"Real weather forecasts are only available up to {MAX_FORECAST_HORIZON_DAYS} days in advance. "
                    f"Forecast data for general period '{date_or_month_str}' is not currently available.",
                )

    # 2. Keywords: 'today', 'tomorrow', 'yesterday'
    if raw == "today":
        return ref_date, None, None
    if raw == "tomorrow":
        return ref_date + timedelta(days=1), None, None
    if raw == "yesterday":
        past_date = ref_date - timedelta(days=1)
        return past_date, None, None

    # 3. Relative patterns: 'in X days' / 'X days from now'
    in_days_match = re.search(r"\bin\s+(\d+)\s+days?\b", raw) or re.search(r"\b(\d+)\s+days?\s+from\s+now\b", raw)
    if in_days_match:
        days_offset = int(in_days_match.group(1))
        return ref_date + timedelta(days=days_offset), None, None

    # 4. Deterministic Weekday references
    matched_weekday_idx: Optional[int] = None
    for day_name, day_idx in WEEKDAYS_FULL.items():
        if re.search(rf"\b{day_name}\b", raw):
            matched_weekday_idx = day_idx
            break
    if matched_weekday_idx is None:
        for day_name, day_idx in WEEKDAYS_ABBR.items():
            if re.search(rf"\b{day_name}\b", raw):
                matched_weekday_idx = day_idx
                break

    if matched_weekday_idx is not None:
        current_weekday = ref_date.weekday()
        diff = (matched_weekday_idx - current_weekday) % 7
        is_next = bool(re.search(r"\bnext\b", raw))
        is_this = bool(re.search(r"\bthis\b", raw))

        if is_next:
            if diff == 0:
                days_ahead = 7
            elif matched_weekday_idx > current_weekday:
                # Occurrence in following week
                days_ahead = diff + 7
            else:
                # e.g., today is Friday (4), next Monday (0): diff is 3 (already in next calendar week)
                days_ahead = diff
        elif is_this:
            if diff == 0:
                days_ahead = 0  # 'this Friday' on Friday is today
            else:
                days_ahead = diff
        else:
            # Plain weekday reference or 'upcoming' / 'coming'
            if diff == 0:
                if re.search(r"\b(upcoming|coming)\b", raw):
                    days_ahead = 7
                else:
                    days_ahead = 0  # Same-day reference
            else:
                days_ahead = diff

        target = ref_date + timedelta(days=days_ahead)
        return target, None, None

    # 5. Standard date format parsing (ISO YYYY-MM-DD or standard formats)
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%B %d, %Y", "%B %d", "%b %d"):
        try:
            parsed_dt = datetime.strptime(raw, fmt)
            if fmt in ("%B %d", "%b %d"):
                parsed_dt = parsed_dt.replace(year=ref_date.year)
                # If parsed date is significantly in the past this calendar year, treat as next year
                if parsed_dt.date() < ref_date - timedelta(days=30):
                    parsed_dt = parsed_dt.replace(year=ref_date.year + 1)
            return parsed_dt.date(), None, None
        except ValueError:
            continue

    # Fallback to dateutil parser if available
    try:
        from dateutil import parser as dt_parser
        parsed_dt = dt_parser.parse(raw, default=datetime(ref_date.year, ref_date.month, ref_date.day))
        return parsed_dt.date(), None, None
    except Exception:
        pass

    return (
        None,
        "ambiguous_input",
        f"Unable to determine a specific upcoming date for forecast from '{date_or_month_str}'.",
    )


def resolve_coordinates(
    city: str,
    timeout: float = 5.0,
    max_retries: int = 2,
) -> Tuple[Optional[Tuple[float, float, str]], str, Optional[str]]:
    """Resolve latitude, longitude, and canonical name for a given city with transient retry.

    Returns:
        ((lat, lon, canonical_name), status, error_message)
        status: 'success', 'unknown_destination', or 'service_unavailable'
    """
    normalized_city = city.strip().lower()

    if normalized_city in KNOWN_CITY_COORDINATES:
        lat, lon = KNOWN_CITY_COORDINATES[normalized_city]
        return (lat, lon, city.strip().title()), "success", None

    params = {"name": city.strip(), "count": 1, "language": "en", "format": "json"}
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(OPEN_METEO_GEO_URL, params=params, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results")
                if results and len(results) > 0:
                    item = results[0]
                    lat = float(item["latitude"])
                    lon = float(item["longitude"])
                    canonical_name = item.get("name", city.strip().title())
                    return (lat, lon, canonical_name), "success", None
                # 200 OK but city was not found in database
                return (
                    None,
                    "unknown_destination",
                    f"Could not find coordinates for city: '{city.strip()}'.",
                )
            # Transient HTTP errors (overloaded 503, rate limit 429, gateway 502/504)
            if resp.status_code in {429, 502, 503, 504} and attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            return (
                None,
                "service_unavailable",
                f"Geocoding service returned HTTP {resp.status_code} while looking up '{city.strip()}'.",
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            logger.warning("Geocoding lookup timed out for city %r", city)
            return (
                None,
                "service_unavailable",
                f"Geocoding service timed out while looking up '{city.strip()}'.",
            )
        except Exception as exc:
            logger.warning("Geocoding lookup failed for city %r: %s", city, exc)
            return (
                None,
                "service_unavailable",
                f"Geocoding service connection failed while looking up '{city.strip()}': {exc}",
            )
    return (
        None,
        "service_unavailable",
        f"Geocoding service unavailable while looking up '{city.strip()}'.",
    )


def fetch_open_meteo_forecast(
    lat: float,
    lon: float,
    target_date: date,
    timeout: float = 5.0,
    max_retries: int = 2,
) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
    """Fetch daily weather forecast for a specific date from Open-Meteo with retry.

    Uses official Open-Meteo daily parameter 'weather_code'.

    Returns:
        (data_dict, status, error_message)
        status: 'success', 'unavailable', or 'service_unavailable'
    """
    date_str = target_date.isoformat()
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "weather_code,temperature_2m_max,temperature_2m_min",
        "timezone": "auto",
        "start_date": date_str,
        "end_date": date_str,
    }

    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp.json(), "success", None
            if resp.status_code == 400:
                try:
                    err_data = resp.json()
                    reason = err_data.get("reason", "")
                    if "out of allowed range" in reason.lower():
                        return (
                            None,
                            "unavailable",
                            f"Real weather forecasts are only available up to {MAX_FORECAST_HORIZON_DAYS} days in advance. "
                            f"Forecast data for {date_str} is outside the allowed range.",
                        )
                except Exception:
                    pass
                return (
                    None,
                    "service_unavailable",
                    f"Weather forecast service returned HTTP 400.",
                )
            if resp.status_code in {429, 502, 503, 504} and attempt < max_retries:
                logger.info("Open-Meteo returned %d (overloaded/throttled). Retrying attempt %d...", resp.status_code, attempt + 1)
                time.sleep(0.5 * (attempt + 1))
                continue
            logger.warning("Open-Meteo forecast API error %d: %s", resp.status_code, resp.text)
            return (
                None,
                "service_unavailable",
                f"Weather forecast service returned HTTP {resp.status_code}.",
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            logger.warning("Open-Meteo forecast request timed out for (%f, %f)", lat, lon)
            return None, "service_unavailable", "Weather forecast service timed out."
        except Exception as exc:
            logger.warning("Open-Meteo forecast request failed: %s", exc)
            return None, "service_unavailable", f"Weather forecast service connection failed: {exc}"

    return None, "service_unavailable", "Weather forecast service is temporarily unavailable."



def get_weather_forecast(city: str, date_or_month: str) -> dict:
    """Retrieve weather forecast for a given city and date or month reference.

    Input: a city name and a date or month reference.
    Output schema:
        {
            "city": "Tokyo",
            "date": "2026-10-14",
            "conditions": "Slight rain",
            "temp_min_c": 18.5,
            "temp_max_c": 22.0,
            "temp_range_c": [18.5, 22.0],
            "status": "success"
        }
    """
    # 1. Input validation
    if not city or not isinstance(city, str) or not city.strip():
        logger.warning("get_weather_forecast called with invalid city: %r", city)
        return {
            "city": str(city) if city is not None else "",
            "status": "invalid_input",
            "message": "City name must be a non-empty string.",
        }

    if not date_or_month or not isinstance(date_or_month, str) or not date_or_month.strip():
        logger.warning("get_weather_forecast called with empty date_or_month for city %r", city)
        return {
            "city": city.strip(),
            "status": "ambiguous_input",
            "message": "Date or month must be provided as a non-empty string.",
        }

    city_clean = city.strip()
    date_or_month_clean = date_or_month.strip()
    today = date.today()

    # 2. Parse target date
    target_date, err_status, err_msg = parse_target_date(date_or_month_clean, ref_date=today)
    if err_status is not None:
        return {
            "city": city_clean,
            "requested_period": date_or_month_clean,
            "status": err_status,
            "message": err_msg,
        }

    assert target_date is not None
    delta_days = (target_date - today).days

    # 3. Horizon validations
    if delta_days < 0:
        return {
            "city": city_clean,
            "date": target_date.isoformat(),
            "status": "invalid_date",
            "message": "Weather forecast cannot be retrieved for past dates.",
        }

    if delta_days > MAX_FORECAST_HORIZON_DAYS:
        return {
            "city": city_clean,
            "requested_period": date_or_month_clean,
            "date": target_date.isoformat(),
            "status": "unavailable",
            "message": (
                f"Real weather forecasts are only available up to {MAX_FORECAST_HORIZON_DAYS} days in advance. "
                f"Forecast data for {target_date.isoformat()} is not currently available."
            ),
        }

    # 4. Resolve City Coordinates (distinguishing unknown destination from service errors)
    geo_res, geo_status, geo_msg = resolve_coordinates(city_clean)
    if geo_status != "success" or not geo_res:
        return {
            "city": city_clean,
            "status": geo_status,
            "message": geo_msg or f"Could not resolve city: '{city_clean}'.",
        }

    lat, lon, canonical_city = geo_res

    # 5. Fetch forecast from Open-Meteo
    data, fetch_status, fetch_msg = fetch_open_meteo_forecast(lat, lon, target_date)
    if fetch_status != "success" or not data:
        return {
            "city": canonical_city,
            "date": target_date.isoformat(),
            "status": fetch_status,
            "message": fetch_msg or "Weather forecast service is temporarily unavailable or unreachable.",
        }

    daily = data.get("daily")
    if not isinstance(daily, dict):
        return {
            "city": canonical_city,
            "date": target_date.isoformat(),
            "status": "incomplete_data",
            "message": "Weather provider response missing daily forecast object.",
        }

    # Validate target date alignment in returned time array
    time_list = daily.get("time")
    if not time_list or not isinstance(time_list, list) or len(time_list) == 0:
        return {
            "city": canonical_city,
            "date": target_date.isoformat(),
            "status": "incomplete_data",
            "message": "Weather provider response missing forecast date array.",
        }

    target_date_str = target_date.isoformat()
    try:
        idx = time_list.index(target_date_str)
    except ValueError:
        return {
            "city": canonical_city,
            "date": target_date_str,
            "status": "incomplete_data",
            "message": f"Requested date '{target_date_str}' not found in weather provider response.",
        }

    # Validate weather_code (supporting both official 'weather_code' and fallback 'weathercode')
    w_code_list = daily.get("weather_code") if "weather_code" in daily else daily.get("weathercode")
    if not w_code_list or not isinstance(w_code_list, list) or len(w_code_list) <= idx:
        return {
            "city": canonical_city,
            "date": target_date_str,
            "status": "incomplete_data",
            "message": "Weather provider response missing weather code data.",
        }

    raw_w_code = w_code_list[idx]
    if raw_w_code is None:
        return {
            "city": canonical_city,
            "date": target_date_str,
            "status": "incomplete_data",
            "message": "Weather provider returned null weather code.",
        }

    try:
        w_code = int(raw_w_code)
    except (ValueError, TypeError):
        return {
            "city": canonical_city,
            "date": target_date_str,
            "status": "incomplete_data",
            "message": f"Invalid weather code format: {raw_w_code}",
        }

    # Validate temperature_2m_max and temperature_2m_min
    max_list = daily.get("temperature_2m_max")
    min_list = daily.get("temperature_2m_min")
    if (
        not max_list or not isinstance(max_list, list) or len(max_list) <= idx
        or not min_list or not isinstance(min_list, list) or len(min_list) <= idx
    ):
        return {
            "city": canonical_city,
            "date": target_date_str,
            "status": "incomplete_data",
            "message": "Weather provider response missing temperature data.",
        }

    raw_max = max_list[idx]
    raw_min = min_list[idx]
    if raw_max is None or raw_min is None:
        return {
            "city": canonical_city,
            "date": target_date_str,
            "status": "incomplete_data",
            "message": "Weather provider returned null temperature values.",
        }

    try:
        temp_max = float(raw_max)
        temp_min = float(raw_min)
    except (ValueError, TypeError):
        return {
            "city": canonical_city,
            "date": target_date_str,
            "status": "incomplete_data",
            "message": "Invalid temperature values in weather provider response.",
        }

    # Validate that temperatures are finite numbers (not NaN or Inf)
    if not math.isfinite(temp_min) or not math.isfinite(temp_max):
        return {
            "city": canonical_city,
            "date": target_date_str,
            "status": "incomplete_data",
            "message": "Non-finite temperature values (NaN/Infinity) in weather provider response.",
        }

    # Validate range consistency: minimum temperature must not exceed maximum
    if temp_min > temp_max:
        return {
            "city": canonical_city,
            "date": target_date_str,
            "status": "incomplete_data",
            "message": f"Inconsistent temperature range: minimum ({temp_min}°C) exceeds maximum ({temp_max}°C).",
        }

    condition_desc = WMO_CODE_DESCRIPTIONS.get(w_code, f"Unspecified conditions (WMO code {w_code})")

    result = {
        "city": canonical_city,
        "date": target_date_str,
        "conditions": condition_desc,
        "temp_min_c": round(temp_min, 1),
        "temp_max_c": round(temp_max, 1),
        "temp_range_c": [round(temp_min, 1), round(temp_max, 1)],
        "status": "success",
    }
    logger.info(
        "get_weather_forecast(city=%r, date=%r) -> %s, temp_range: %s",
        canonical_city,
        target_date_str,
        condition_desc,
        result["temp_range_c"],
    )
    return result
