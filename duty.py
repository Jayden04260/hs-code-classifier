"""
duty.py

Looks up the standard ("Third country duty", geographical area ERGA
OMNES - Latin for "towards all", the rate that applies unless a
specific trade agreement or preference overrides it) import duty rate
for a commodity code, from the same free public Trade Tariff API used
elsewhere in this project.

Verified against several real, independently-checkable codes before
being added (see commit history/PR description): T-shirts (6109100010)
returned 12.00%, matching the widely-known real-world rate for apparel;
a random sample of 5 other commodity codes all returned plausible,
correctly-shaped percentages with no missing or malformed data.

Deliberately narrow scope: only the standard/non-preferential rate, not
origin-specific preferential rates, tariff quotas, or anti-dumping
duties - those depend on the specific trading partner and goods
history in ways a code alone can't resolve. Always clearly labelled as
"standard rate" in output - see run.py - not presented as a final,
binding duty figure. This isn't a substitute for a customs broker's
actual determination.
"""

import json
import urllib.error
import urllib.request

COMMODITY_URL = "https://www.trade-tariff.service.gov.uk/api/v2/commodities/{}"
THIRD_COUNTRY_DUTY = "Third country duty"
ERGA_OMNES = "ERGA OMNES"


def fetch_standard_duty_rate(code: str, timeout: int = 30) -> str | None:
    """
    Returns the standard (ERGA OMNES) "Third country duty" rate for a
    10-digit commodity code, as a display string like "12.00%", or None
    if the code isn't declarable at commodity level or has no such
    measure (e.g. it's actually a heading/chapter-level reference, or
    duty-free with no explicit measure recorded).
    """
    request = urllib.request.Request(COMMODITY_URL.format(code), headers={"User-Agent": "hs-code-classifier/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError:
        return None

    if "errors" in payload:
        return None

    return _extract_third_country_duty(payload)


def _extract_third_country_duty(payload: dict) -> str | None:
    included = payload.get("included", [])
    by_ref = {(item["type"], item["id"]): item for item in included}

    for item in included:
        if item["type"] != "measure":
            continue
        rels = item.get("relationships", {})

        measure_type_ref = rels.get("measure_type", {}).get("data")
        geo_ref = rels.get("geographical_area", {}).get("data")
        duty_expr_ref = rels.get("duty_expression", {}).get("data")
        if not (measure_type_ref and geo_ref and duty_expr_ref):
            continue

        measure_type = by_ref.get(("measure_type", measure_type_ref["id"]))
        geographical_area = by_ref.get(("geographical_area", geo_ref["id"]))
        if not (measure_type and geographical_area):
            continue

        is_standard_rate = (
            measure_type["attributes"]["description"] == THIRD_COUNTRY_DUTY
            and geographical_area["attributes"]["description"] == ERGA_OMNES
        )
        if is_standard_rate:
            duty_expression = by_ref.get(("duty_expression", duty_expr_ref["id"]))
            if duty_expression:
                return duty_expression["attributes"]["verbose_duty"]

    return None
