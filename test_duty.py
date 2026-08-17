import json
import urllib.error

from duty import fetch_standard_duty_rate


def _payload_with_measure(measure_type_desc: str, geo_desc: str, verbose_duty: str) -> dict:
    """Builds a minimal JSON:API payload with the same measure ->
    measure_type / geographical_area / duty_expression relationship
    shape as the real Trade Tariff API response."""
    return {
        "data": {"id": "1", "type": "commodity", "attributes": {}},
        "included": [
            {
                "id": "m1",
                "type": "measure",
                "relationships": {
                    "measure_type": {"data": {"id": "mt1", "type": "measure_type"}},
                    "geographical_area": {"data": {"id": "geo1", "type": "geographical_area"}},
                    "duty_expression": {"data": {"id": "de1", "type": "duty_expression"}},
                },
            },
            {"id": "mt1", "type": "measure_type", "attributes": {"description": measure_type_desc}},
            {"id": "geo1", "type": "geographical_area", "attributes": {"description": geo_desc}},
            {"id": "de1", "type": "duty_expression", "attributes": {"verbose_duty": verbose_duty}},
        ],
    }


class _FakeResponse:
    def __init__(self, payload: dict):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_extracts_third_country_duty_for_erga_omnes(monkeypatch):
    payload = _payload_with_measure("Third country duty", "ERGA OMNES", "12.00%")
    monkeypatch.setattr("duty.urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(payload))

    assert fetch_standard_duty_rate("6109100010") == "12.00%"


def test_ignores_non_erga_omnes_measures(monkeypatch):
    payload = _payload_with_measure("Third country duty", "Andorra", "0.00%")
    monkeypatch.setattr("duty.urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(payload))

    assert fetch_standard_duty_rate("6109100010") is None


def test_ignores_non_third_country_duty_measure_types(monkeypatch):
    payload = _payload_with_measure("Import VAT", "ERGA OMNES", "20.00%")
    monkeypatch.setattr("duty.urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(payload))

    assert fetch_standard_duty_rate("6109100010") is None


def test_returns_none_when_response_has_errors(monkeypatch):
    payload = {"errors": [{"detail": "404 - Not Found"}]}
    monkeypatch.setattr("duty.urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(payload))

    assert fetch_standard_duty_rate("0000000000") is None


def test_returns_none_on_http_error(monkeypatch):
    def raise_http_error(request, timeout=None):
        raise urllib.error.HTTPError(url="", code=404, msg="Not Found", hdrs=None, fp=None)

    monkeypatch.setattr("duty.urllib.request.urlopen", raise_http_error)

    assert fetch_standard_duty_rate("0000000000") is None
