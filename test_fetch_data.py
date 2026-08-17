import json

from fetch_data import fetch_references

SAMPLE_RESPONSE = json.dumps(
    {
        "data": [
            {
                "id": "1",
                "type": "search_reference",
                "attributes": {
                    "title": "3d printer",
                    "referenced_id": "8485",
                    "referenced_class": "Heading",
                    "goods_nomenclature_item_id": "8485000000",
                },
            }
        ]
    }
).encode("utf-8")


class _FakeResponse:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_fetch_references_extracts_title_code_and_level(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return _FakeResponse(SAMPLE_RESPONSE)

    monkeypatch.setattr("fetch_data.urllib.request.urlopen", fake_urlopen)

    references = fetch_references()

    assert references == [{"title": "3d printer", "code": "8485000000", "level": "Heading"}]
