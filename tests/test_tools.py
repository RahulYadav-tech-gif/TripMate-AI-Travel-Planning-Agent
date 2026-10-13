"""Unit tests for Destination and Weather tools.

Covers chunking, semantic search, date parsing, and Open-Meteo edge cases.
"""

from pathlib import Path
import re
import pytest

from src.rag.ingest import load_all_destination_chunks
from src.tools.destination import search_destination_guide


def test_load_all_destination_chunks():
    """Verify that all 4 city guides are ingested into exactly 20 section chunks."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    chunks = load_all_destination_chunks(data_dir)
    assert len(chunks) == 20

    cities = {c.city for c in chunks}
    assert cities == {"bangkok", "barcelona", "reykjavik", "tokyo"}

    sections = {c.section for c in chunks}
    assert sections == {
        "VISA & ENTRY",
        "BEST TIME TO VISIT",
        "LOCAL CUSTOMS",
        "PACKING TIPS",
        "SAFETY & HEALTH",
    }


def test_search_destination_guide_tokyo_visa():
    """Verify semantic search retrieves Tokyo Visa & Entry chunk."""
    results = search_destination_guide("What are the visa rules for Tokyo?")
    assert isinstance(results, list)
    assert len(results) > 0
    assert any("VISA & ENTRY" in chunk for chunk in results)
    assert any("Japan" in chunk or "Tokyo" in chunk for chunk in results)


def test_search_destination_guide_bangkok_packing():
    """Verify semantic search retrieves Bangkok packing recommendations."""
    results = search_destination_guide("What should I pack for Bangkok?")
    assert isinstance(results, list)
    assert len(results) > 0
    assert any("PACKING TIPS" in chunk for chunk in results)
    assert any("breathable" in chunk.lower() or "modest" in chunk.lower() for chunk in results)


def test_search_destination_guide_unindexed_destination():
    """Verify that RAG returns retrieved knowledge base chunks and contains no unindexed city content."""
    results = search_destination_guide("What is the visa policy for Sydney Australia?")
    assert isinstance(results, list)
    # The knowledge base only contains Tokyo, Bangkok, Barcelona, Reykjavik.
    # None of the chunks should contain Sydney or Australia info.
    for chunk in results:
        assert "sydney" not in chunk.lower()


def test_search_destination_guide_malformed_inputs():
    """Verify error handling for empty, whitespace, and non-string inputs."""
    assert search_destination_guide("") == []
    assert search_destination_guide("   ") == []
    assert search_destination_guide(None) == []  # type: ignore


# ==============================================================================
# Weather Forecast Tool Unit Tests (Phase 3 Refined)
# ==============================================================================

from unittest.mock import patch, MagicMock
from datetime import date, timedelta
import requests
from src.tools.weather import get_weather_forecast, parse_target_date


def test_parse_target_date_keywords_and_relative():
    """Verify relative keyword and day offset parsing."""
    today = date(2026, 10, 13)
    # Today / Tomorrow / Yesterday
    assert parse_target_date("today", ref_date=today) == (today, None, None)
    assert parse_target_date("tomorrow", ref_date=today) == (today + timedelta(days=1), None, None)
    assert parse_target_date("yesterday", ref_date=today) == (today - timedelta(days=1), None, None)

    # In X days
    assert parse_target_date("in 3 days", ref_date=today) == (today + timedelta(days=3), None, None)
    assert parse_target_date("5 days from now", ref_date=today) == (today + timedelta(days=5), None, None)

    # ISO format
    assert parse_target_date("2026-10-20", ref_date=today) == (date(2026, 10, 20), None, None)


def test_parse_target_date_weekdays_deterministic():
    """Verify deterministic rules for this, next, and plain weekday references."""
    # Case 1: Ref date is Tuesday (2026-10-13, weekday = 1)
    tue = date(2026, 10, 13)
    assert parse_target_date("this Tuesday", ref_date=tue)[0] == tue
    assert parse_target_date("Tuesday", ref_date=tue)[0] == tue
    assert parse_target_date("next Tuesday", ref_date=tue)[0] == tue + timedelta(days=7)
    assert parse_target_date("this Friday", ref_date=tue)[0] == tue + timedelta(days=3)
    assert parse_target_date("Friday", ref_date=tue)[0] == tue + timedelta(days=3)
    assert parse_target_date("next Friday", ref_date=tue)[0] == tue + timedelta(days=10)

    # Case 2: Ref date is Friday (2026-10-16, weekday = 4)
    fri = date(2026, 10, 16)
    assert parse_target_date("this Friday", ref_date=fri)[0] == fri
    assert parse_target_date("Friday", ref_date=fri)[0] == fri
    assert parse_target_date("next Friday", ref_date=fri)[0] == fri + timedelta(days=7)
    # Target Monday (weekday 0) from Friday: next calendar week starts in 3 days
    assert parse_target_date("this Monday", ref_date=fri)[0] == fri + timedelta(days=3)
    assert parse_target_date("next Monday", ref_date=fri)[0] == fri + timedelta(days=3)


def test_parse_target_date_general_month_unavailable():
    """Verify standalone general month returns unavailable status."""
    today = date(2026, 10, 13)
    target_date, err_status, msg = parse_target_date("December", ref_date=today)
    assert target_date is None
    assert err_status == "unavailable"


def test_get_weather_forecast_official_weather_code_field():
    """Verify get_weather_forecast works with official 'weather_code' Open-Meteo field."""
    tomorrow_str = (date.today() + timedelta(days=1)).isoformat()
    mock_payload = {
        "daily": {
            "time": [tomorrow_str],
            "weather_code": [0],
            "temperature_2m_max": [24.0],
            "temperature_2m_min": [16.5],
        }
    }
    with patch("src.tools.weather.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_get.return_value = mock_resp

        result = get_weather_forecast("Tokyo", "tomorrow")

        assert result["status"] == "success"
        assert result["city"] == "Tokyo"
        assert result["conditions"] == "Clear sky"
        assert result["temp_min_c"] == 16.5
        assert result["temp_max_c"] == 24.0
        assert result["temp_range_c"] == [16.5, 24.0]
        assert result["date"] == tomorrow_str


def test_get_weather_forecast_legacy_weathercode_field_fallback():
    """Verify backward compatibility fallback for 'weathercode' field."""
    tomorrow_str = (date.today() + timedelta(days=1)).isoformat()
    mock_payload = {
        "daily": {
            "time": [tomorrow_str],
            "weathercode": [61],
            "temperature_2m_max": [20.0],
            "temperature_2m_min": [15.0],
        }
    }
    with patch("src.tools.weather.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_get.return_value = mock_resp

        result = get_weather_forecast("Tokyo", "tomorrow")

        assert result["status"] == "success"
        assert result["conditions"] == "Slight rain"


def test_get_weather_forecast_past_date():
    """Verify past dates return invalid_date status without fabrication."""
    result = get_weather_forecast("Barcelona", "yesterday")
    assert result["status"] == "invalid_date"
    assert "past" in result["message"].lower()


def test_get_weather_forecast_outside_horizon():
    """Verify dates beyond allowed horizon return unavailable status."""
    future_date = (date.today() + timedelta(days=30)).isoformat()
    result = get_weather_forecast("Tokyo", future_date)
    assert result["status"] == "unavailable"
    assert "15 days" in result["message"]


def test_get_weather_forecast_month_reference():
    """Verify general month reference returns unavailable status."""
    result = get_weather_forecast("Reykjavik", "December")
    assert result["status"] == "unavailable"
    assert "15 days" in result["message"]



def test_get_weather_forecast_unknown_city_vs_service_failure():
    """Verify unknown city returns unknown_destination while API failure returns service_unavailable."""
    # 1. Unknown city (geocoding returns 200 with empty results)
    with patch("src.tools.weather.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": []}
        mock_get.return_value = mock_resp

        result = get_weather_forecast("AtlantisFakeCityXYZ", "tomorrow")
        assert result["status"] == "unknown_destination"
        assert "Could not find coordinates" in result["message"]

    # 2. Geocoding service timeout / error
    with patch("src.tools.weather.requests.get", side_effect=requests.exceptions.Timeout):
        result = get_weather_forecast("AtlantisFakeCityXYZ", "tomorrow")
        assert result["status"] == "service_unavailable"
        assert "timed out" in result["message"].lower()

    # 3. Geocoding HTTP 500 error
    with patch("src.tools.weather.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp

        result = get_weather_forecast("AtlantisFakeCityXYZ", "tomorrow")
        assert result["status"] == "service_unavailable"
        assert "HTTP 500" in result["message"]


def test_get_weather_forecast_incomplete_data_validation():
    """Verify incomplete or null weather fields return incomplete_data status."""
    tomorrow_str = (date.today() + timedelta(days=1)).isoformat()

    # Case A: Null weather_code
    payload_null_code = {
        "daily": {
            "time": [tomorrow_str],
            "weather_code": [None],
            "temperature_2m_max": [20.0],
            "temperature_2m_min": [15.0],
        }
    }
    with patch("src.tools.weather.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload_null_code
        mock_get.return_value = mock_resp

        result = get_weather_forecast("Tokyo", "tomorrow")
        assert result["status"] == "incomplete_data"
        assert "null weather code" in result["message"].lower()

    # Case B: Null temperature value
    payload_null_temp = {
        "daily": {
            "time": [tomorrow_str],
            "weather_code": [1],
            "temperature_2m_max": [None],
            "temperature_2m_min": [15.0],
        }
    }
    with patch("src.tools.weather.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload_null_temp
        mock_get.return_value = mock_resp

        result = get_weather_forecast("Tokyo", "tomorrow")
        assert result["status"] == "incomplete_data"
        assert "null temperature" in result["message"].lower()

    # Case C: Date mismatch / missing in time array
    payload_wrong_date = {
        "daily": {
            "time": ["2099-01-01"],
            "weather_code": [1],
            "temperature_2m_max": [20.0],
            "temperature_2m_min": [15.0],
        }
    }
    with patch("src.tools.weather.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload_wrong_date
        mock_get.return_value = mock_resp

        result = get_weather_forecast("Tokyo", "tomorrow")
        assert result["status"] == "incomplete_data"
        assert "not found in weather provider response" in result["message"].lower()


def test_get_weather_forecast_malformed_inputs():
    """Verify input validation on empty or non-string parameters."""
    assert get_weather_forecast("", "tomorrow")["status"] == "invalid_input"
    assert get_weather_forecast(None, "tomorrow")["status"] == "invalid_input"  # type: ignore
    assert get_weather_forecast("Tokyo", "")["status"] == "ambiguous_input"
    assert get_weather_forecast("Tokyo", None)["status"] == "ambiguous_input"  # type: ignore


def test_get_weather_forecast_non_finite_and_inconsistent_temperatures():
    """Check 1: Verify NaN, Infinity, and min > max temperatures return incomplete_data."""
    tomorrow_str = (date.today() + timedelta(days=1)).isoformat()

    # Case 1: NaN temperature
    payload_nan = {
        "daily": {
            "time": [tomorrow_str],
            "weather_code": [0],
            "temperature_2m_max": [float("nan")],
            "temperature_2m_min": [15.0],
        }
    }
    with patch("src.tools.weather.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload_nan
        mock_get.return_value = mock_resp

        result = get_weather_forecast("Tokyo", "tomorrow")
        assert result["status"] == "incomplete_data"
        assert "non-finite" in result["message"].lower()

    # Case 2: Infinity temperature
    payload_inf = {
        "daily": {
            "time": [tomorrow_str],
            "weather_code": [0],
            "temperature_2m_max": [float("inf")],
            "temperature_2m_min": [15.0],
        }
    }
    with patch("src.tools.weather.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload_inf
        mock_get.return_value = mock_resp

        result = get_weather_forecast("Tokyo", "tomorrow")
        assert result["status"] == "incomplete_data"
        assert "non-finite" in result["message"].lower()

    # Case 3: Inconsistent temperatures (min > max)
    payload_inverted = {
        "daily": {
            "time": [tomorrow_str],
            "weather_code": [0],
            "temperature_2m_max": [15.0],
            "temperature_2m_min": [32.0],  # Min exceeds max!
        }
    }
    with patch("src.tools.weather.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload_inverted
        mock_get.return_value = mock_resp

        result = get_weather_forecast("Tokyo", "tomorrow")
        assert result["status"] == "incomplete_data"
        assert "exceeds maximum" in result["message"].lower()


def test_deterministic_date_parsing_comprehensive():
    """Check 2: Verify next Tuesday, this Friday, same-day, invalid date, and 16-day boundary with fixed ref_date."""
    # Fixed Reference Date: Tuesday, 2026-10-13 (weekday 1)
    ref = date(2026, 10, 13)

    # 1. next Tuesday -> 7 days ahead (2026-10-20)
    d, err, _ = parse_target_date("next Tuesday", ref_date=ref)
    assert d == date(2026, 10, 20) and err is None

    # 2. this Friday -> 3 days ahead in current week (2026-10-16)
    d, err, _ = parse_target_date("this Friday", ref_date=ref)
    assert d == date(2026, 10, 16) and err is None

    # 3. Same-day weekday ('Tuesday' and 'this Tuesday' on Tuesday) -> today (2026-10-13)
    d, err, _ = parse_target_date("Tuesday", ref_date=ref)
    assert d == date(2026, 10, 13) and err is None
    d, err, _ = parse_target_date("this Tuesday", ref_date=ref)
    assert d == date(2026, 10, 13) and err is None

    # 4. Invalid date references
    for invalid_val in ["", "   ", "sometime", "winter", "random_string_123"]:
        d, err, msg = parse_target_date(invalid_val, ref_date=ref)
        assert d is None
        assert err in {"ambiguous_input", "unavailable"}

    # 5. Boundary testing relative to ref:
    # 15 days ahead (2026-10-28): within allowed 16-day window (today + 15 days)
    d15, err15, _ = parse_target_date("2026-10-28", ref_date=ref)
    assert d15 == date(2026, 10, 28) and err15 is None
    assert (d15 - ref).days == 15

    # 16 days ahead (2026-10-29): exceeds 15-day maximum horizon
    d16, err16, _ = parse_target_date("2026-10-29", ref_date=ref)
    assert d16 == date(2026, 10, 29) and err16 is None
    assert (d16 - ref).days == 16  # Handled by get_weather_forecast as 'unavailable'


def test_actual_open_meteo_api_live_response_structure():
    """Check 3: Verify live Open-Meteo response structure (time, weather_code, min/max temps)."""
    import math
    result = get_weather_forecast("Tokyo", "tomorrow")
    assert result["status"] == "success"
    assert result["city"] == "Tokyo"
    assert isinstance(result["date"], str)
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", result["date"])

    assert isinstance(result["conditions"], str)
    assert len(result["conditions"]) > 0

    assert isinstance(result["temp_min_c"], float)
    assert isinstance(result["temp_max_c"], float)
    assert math.isfinite(result["temp_min_c"])
    assert math.isfinite(result["temp_max_c"])
    assert result["temp_min_c"] <= result["temp_max_c"]

    assert isinstance(result["temp_range_c"], list)
    assert len(result["temp_range_c"]) == 2
    assert result["temp_range_c"][0] == result["temp_min_c"]
    assert result["temp_range_c"][1] == result["temp_max_c"]



