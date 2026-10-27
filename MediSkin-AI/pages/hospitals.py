"""Hospital / dermatology-clinic recommendations (provider-based; no data is invented)."""
from __future__ import annotations

import streamlit as st

from config import config
from modules import hospital_recommendation as hospitals
from utils import ui


def _card(h: hospitals.Hospital) -> str:
    derm = {True: "Dermatology available", False: "Dermatology not listed"}.get(h.dermatology,
                                                                                "Dermatology availability unknown")
    lines = [f"{ui.esc(h.address + ', ' if h.address else '')}{ui.esc(h.city)}", ui.esc(derm)]
    if h.phone:
        lines.append(f"Phone: {ui.esc(h.phone)}")
    if h.website:
        lines.append(f'<a href="{ui.esc(h.website)}" target="_blank" rel="noopener noreferrer">Website</a>')
    if h.distance_km is not None:
        lines.append(f"About {h.distance_km:.1f} km away (straight-line distance)")
    return (f'<div class="ms-card ms-hospital" style="margin-bottom:.8rem;height:auto"><h4>{ui.esc(h.name)}</h4>'
            f'<div class="meta">{"<br>".join(lines)}</div></div>')


def render() -> None:
    ui.page_header("Find Care", "Dermatology clinics and hospitals you can contact.", "local_hospital")

    with st.form("hospital_search"):
        city = st.text_input("City or area", placeholder="e.g. Hyderabad")
        dermatology_only = st.checkbox("Only show facilities with dermatology")
        with st.expander("Add my coordinates for distances (optional)"):
            st.caption("Distances are shown only for facilities that have coordinates in the data source.")
            lat_col, lon_col = st.columns(2)
            latitude = lat_col.number_input("Latitude", value=None, min_value=-90.0, max_value=90.0, format="%.5f")
            longitude = lon_col.number_input("Longitude", value=None, min_value=-180.0, max_value=180.0, format="%.5f")
        st.form_submit_button("Search", type="primary", icon=":material/search:")

    results = hospitals.find_hospitals(city or None, latitude, longitude, dermatology_only)

    if results:
        st.caption(f"{len(results)} result(s) · data source: {config.HOSPITAL_PROVIDER}")
        for hospital in results:
            ui.render_html(_card(hospital))
    elif hospitals.find_hospitals(limit=1):
        ui.empty_state("No matching facilities", "Try a different city or turn off the dermatology filter.",
                       "search_off")
    else:
        ui.empty_state(
            "No facility data configured",
            "This application does not ship with hospital records. Add real entries to "
            "data/metadata/hospitals.json (see README) or connect a provider. Meanwhile, you can search a map.",
            "local_hospital",
        )

    st.link_button("Search dermatologists on a map", hospitals.maps_search_url(city or None),
                   icon=":material/map:")
    st.caption("Always confirm opening hours, availability and referral requirements with the facility directly.")
    ui.disclaimer()
