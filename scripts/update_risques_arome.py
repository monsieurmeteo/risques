#!/usr/bin/env python3
"""Vigilance 11 aléas calculée sur les données AROME HD de monsieurmeteo.

Variante « Option A » (cf. docs/AROME-VIGILANCE.md) : au lieu de lire les
fichiers JSON ``departements/XX.json`` du hub HARMONIE, ce script lit
directement les fichiers communaux binaires **MCV2** déjà publiés par la
plateforme monsieurmeteo/arome-weather-map
(``output/arome/maps/communes/{dept}.bin.gz``, 52 échéances × 37 colonnes ×
communes) et applique EXACTEMENT les mêmes seuils/logique que
``update_risques.py`` (réutilisés par import) pour produire un ``risques.json``
de même schéma.

Aucun décodage GRIB, aucune modification de la plateforme source, aucun
stockage supplémentaire (~40 Mo téléchargés par run).

Seul écart de données géré ici : Météo-France ne publie pas la visibilité en
open data. Une colonne ``visibility_km`` est reconstruite par proxy au
décodage (brouillard AROME = ``condition_code`` 8 / RH ≥ 96 % & LCC ≥ 90 % &
vent < 10 km/h) sans toucher aux seuils de l'aléa brouillard.
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
import sys
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

from update_risques import (  # mêmes seuils / logique, inchangés
    DepartmentSeries,
    FIRE_DISCLAIMER,
    HAZARD_LABELS,
    HAZARD_LEVELS,
    LITTORAL_DISCLAIMER,
    PARIS_TZ,
    department_codes,
    hazard_levels_manifest,
    build_department_risk,
    _department_centroids,
    _nearest_neighbors,
    _fill_isolated_green,
)

LOGGER = logging.getLogger("risques-arome")
PIPELINE_VERSION = "2.18.0-arome-mcv2"
DEFAULT_MCV2_BASE_URL = (
    "https://raw.githubusercontent.com/monsieurmeteo/arome-weather-map/main/output/arome/maps"
)

# Proxy de visibilité (seule colonne absente des MCV2) — cf. docs.
VIS_CLEAR_KM = 10.0
VIS_FOG_KM = 0.4
VIS_DENSE_FOG_KM = 0.1
FOG_RH_PCT = 96.0
FOG_LCC_PCT = 90.0
FOG_WS_KMH = 10.0
DENSE_RH_PCT = 99.0
DENSE_WS_KMH = 5.0


# --------------------------------------------------------------------------
# Décodeur du format binaire MCV2 (spécification vérifiée, docs/AROME-VIGILANCE.md §A.2)
# --------------------------------------------------------------------------

def _decode_mcv2(raw: bytes) -> dict[str, Any]:
    """Décode un fichier MCV2 (zlib décompressé) vers les structures du pipeline.

    Retourne : {run_time, leads, points, column_names, scales, offsets,
    data} avec ``data`` = tableau float64 (n_points, n_leads, n_cols)
    (NaN = -32768), points = liste de dicts communaux.
    """

    if raw[:4] != b"MCV2":
        raise ValueError("Magic MCV2 absent")

    n, nleads, ncols = struct.unpack_from("<HHH", raw, 4)
    run_text = raw[10:50].split(b"\0", 1)[0].decode("utf-8", "replace").strip()
    off = 50

    points: list[dict[str, Any]] = []
    for _ in range(n):
        insee = raw[off : off + 5].decode("ascii", "replace")
        off += 5
        name_len = raw[off]
        off += 1
        name = raw[off : off + name_len].decode("utf-8", "replace")
        off += name_len
        lat, lon = struct.unpack_from("<ff", raw, off)
        off += 8
        population = struct.unpack_from("<I", raw, off)[0]
        off += 4
        points.append(
            {"code_insee": insee, "name": name, "latitude": lat,
             "longitude": lon, "population": population}
        )

    column_names: list[str] = []
    scales: list[float] = []
    offsets: list[float] = []
    for _ in range(ncols):
        cname = raw[off : off + 32].split(b"\0", 1)[0].decode("utf-8", "replace")
        off += 32
        scale, offset = struct.unpack_from("<ff", raw, off)
        off += 8
        column_names.append(cname)
        scales.append(scale)
        offsets.append(offset)

    leads = list(struct.unpack_from(f"<{nleads}H", raw, off))
    off += 2 * nleads
    if off % 2 == 1:  # alignement sur 2 octets avant les données
        off += 1

    data_start = len(raw) - n * nleads * ncols * 2
    if off != data_start:
        raise ValueError(
            f"En-tête MCV2 incohérent : header {off} vs données attendues {data_start}"
        )

    q = np.frombuffer(raw, dtype="<i2", count=n * nleads * ncols, offset=off)
    q = q.reshape(n, nleads, ncols).astype(np.float64)
    decoded = np.empty_like(q)
    for c in range(ncols):
        column = q[:, :, c]
        real = column * scales[c] - offsets[c]
        decoded[:, :, c] = np.where(column == -32768.0, np.nan, real)

    return {
        "run_time": run_text,
        "leads": leads,
        "points": points,
        "column_names": column_names,
        "data": decoded,  # (n, nleads, ncols) float64
    }


# --------------------------------------------------------------------------
# Chargement réseau + conversion vers DepartmentSeries
# --------------------------------------------------------------------------

def _http_get(session: requests.Session, url: str) -> bytes:
    for attempt in range(3):
        try:
            response = session.get(url, timeout=120)
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == 2:
                raise
    raise RuntimeError("unreachable")


def load_department_series_mcv2(
    session: requests.Session, base_url: str, code: str
) -> tuple[DepartmentSeries | None, str | None]:
    """Télécharge ``{code}.bin.gz``, décode MCV2 et renvoie une DepartmentSeries
    (mêmes conventions que ``update_risques.load_department_series``) + run_time."""

    url = f"{base_url}/communes/{code}.bin.gz"
    try:
        raw_gz = _http_get(session, url)
    except requests.RequestException as error:
        LOGGER.warning("Département %s indisponible (%s)", code, error)
        return None, None

    header = _decode_mcv2(zlib.decompress(raw_gz))
    names = header["column_names"]
    base_cols = {name: idx for idx, name in enumerate(names)}
    data = header["data"]  # (n, nleads, ncols)

    # Colonne proxy visibility_km ajoutée en fin de matrice.
    def _idx(*names_: str) -> int | None:
        for candidate in names_:
            if candidate in base_cols:
                return base_cols[candidate]
        return None
    i_hum = _idx("humidity_pct")
    i_lcc = _idx("cloud_low_pct")
    i_ws = _idx("wind_speed_kmh")
    i_cc = _idx("condition_code")

    # Proxy de visibilité par point (constante sur les 52 échéances, faute de
    # champ visibilité dans l'open data Météo-France) : clair 10 km ;
    # brouillard 0,4 km ; brouillard dense 0,1 km.
    vis_extra = np.full(data.shape[0], VIS_CLEAR_KM, dtype=np.float64)
    humidity = data[:, 0, i_hum] if i_hum is not None else None
    lcc = data[:, 0, i_lcc] if i_lcc is not None else None
    ws = data[:, 0, i_ws] if i_ws is not None else None
    cc = data[:, 0, i_cc] if i_cc is not None else None

    fog = np.zeros(data.shape[0], dtype=bool)
    if cc is not None:
        fog |= (np.nan_to_num(cc) == 8.0)
    if humidity is not None and lcc is not None and ws is not None:
        fog |= (
            (np.nan_to_num(humidity, nan=0.0) >= FOG_RH_PCT)
            & (np.nan_to_num(lcc, nan=0.0) >= FOG_LCC_PCT)
            & (np.nan_to_num(ws, nan=1e9) < FOG_WS_KMH)
        )
    dense = np.zeros(data.shape[0], dtype=bool)
    if humidity is not None and ws is not None:
        dense = (
            (np.nan_to_num(humidity, nan=0.0) >= DENSE_RH_PCT)
            & (np.nan_to_num(ws, nan=1e9) <= DENSE_WS_KMH)
        )
    # Proxy de visibilité : clair 10 km ; brouillard 0,4 km ; brouillard dense 0,1 km.
    vis_extra[fog & dense] = VIS_DENSE_FOG_KM
    vis_extra[fog & ~dense] = VIS_FOG_KM
    vis_extra[~fog & dense] = VIS_FOG_KM

    full_names = names + ["visibility_km"]
    ncols = len(full_names)
    # forecast[lead] : (n_points x ncols)
    n, nleads = data.shape[0], data.shape[1]
    forecast: list[np.ndarray] = []
    for lead in range(nleads):
        block = data[:, lead, :]  # (n, ncols_originaux)
        combined = np.concatenate([block, vis_extra[:, None]], axis=1)
        forecast.append(combined)

    # Altitudes fixes par point (colonne altitude_m, constante).
    i_alt = base_cols.get("altitude_m")
    if i_alt is not None:
        altitudes = data[:, 0, i_alt]
    else:
        altitudes = np.full(n, np.nan, dtype=np.float64)

    try:
        run_time = datetime.fromisoformat(
            header["run_time"].replace("Z", "+00:00")
        )
    except ValueError:
        run_time = datetime.now(timezone.utc)
    times = [
        run_time + timedelta(hours=int(lead)) for lead in header["leads"]
    ]

    series = DepartmentSeries(
        code=code,
        times=times,
        columns={name: idx for idx, name in enumerate(full_names)},
        forecast=forecast,
        altitudes=altitudes,
    )
    return series, run_time.isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------
# Orchestration (mêmes étapes que build_risques, source = MCV2 AROME)
# --------------------------------------------------------------------------

def build_risques_arome(
    base_url: str,
    day_count: int,
    geojson_path: Path | None = None,
) -> tuple[dict[str, Any], str | None]:
    session = requests.Session()
    session.headers["User-Agent"] = "monsieurmeteo-vigilance-arome/1.0"

    today = (datetime.now(timezone.utc).astimezone(PARIS_TZ)
             - timedelta(hours=5)).date()  # journée météo 5h→5h (identique à update_risques)
    departments: dict[str, Any] = {}
    national_by_date: dict[str, dict[str, tuple[float, str]]] = {}
    run_time: str | None = None
    missing = 0

    for code in department_codes():
        series, series_run = load_department_series_mcv2(session, base_url, code)
        if series is None:
            missing += 1
            continue
        result = build_department_risk(series, day_count, today)
        if result is None:
            missing += 1
            continue
        risk, raw_by_date = result
        departments[code] = risk
        if run_time is None and series_run:
            run_time = series_run

        for date_str, raw in raw_by_date.items():
            slot = national_by_date.setdefault(date_str, {})

            def consider(field: str, value: float, better: Any) -> None:
                if not np.isfinite(value):
                    return
                current = slot.get(field)
                if current is None or better(value, current[0]):
                    slot[field] = (value, code)

            consider("max_temperature", raw["max_temperature"], lambda new, best: new > best)
            consider("min_temperature", raw["min_temperature"], lambda new, best: new < best)
            consider("max_gust", raw["max_gust"], lambda new, best: new > best)
            consider("total_precip_mm", raw["total_precip_mm"], lambda new, best: new > best)
        LOGGER.info("Département %s : %s échéances traitées", code, len(series.times))

    if missing:
        LOGGER.warning("%s département(s) manquants", missing)
    if len(departments) < 80:
        raise RuntimeError(
            f"Trop de départements manquants ({len(departments)}/96) — "
            "les MCV2 de la plateforme ne sont pas tous publiés."
        )

    # Cohérence spatiale des orages (même règle que la production HARMONIE).
    if geojson_path is not None:
        centroids = _department_centroids(geojson_path)
        if centroids:
            neighbors = _nearest_neighbors(centroids)
            _fill_isolated_green(departments, neighbors)

    ordered_dates = [
        (today + timedelta(days=offset)).isoformat() for offset in range(day_count)
    ]
    national_summary = []
    for date_str in ordered_dates:
        slot = national_by_date.get(date_str, {})

        def field(name: str) -> dict[str, Any] | None:
            entry = slot.get(name)
            if entry is None:
                return None
            value, department = entry
            return {"value": round(value, 1), "department": department}

        national_summary.append({
            "date": date_str,
            "max_temperature": field("max_temperature"),
            "min_temperature": field("min_temperature"),
            "max_gust": field("max_gust"),
            "max_precip": field("total_precip_mm"),
        })

    manifest = {
        "schema_version": 1,
        "status": "ok",
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_time": run_time,
        "source": {
            "model": "AROME HD (Météo-France 0,025° via monsieurmeteo/arome-weather-map)",
            "base_url": base_url,
        },
        "hazards": HAZARD_LABELS,
        "hazard_levels": hazard_levels_manifest(),
        "fire_disclaimer": FIRE_DISCLAIMER,
        "littoral_disclaimer": LITTORAL_DISCLAIMER,
        "national_summary": national_summary,
        "departments": departments,
    }
    return manifest, run_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mcv2-base-url",
        default=DEFAULT_MCV2_BASE_URL,
        help="Racine des données communales MCV2 (par défaut : branche main de la plateforme)",
    )
    parser.add_argument("--output-dir", default="build/arome-national")
    parser.add_argument("--days", type=int, default=2,
                        help="Jours calendaires à conserver (J+2 partiel : AROME borne à 51 h)")
    parser.add_argument("--geojson", default=None,
                        help="Contours départementaux (par défaut : config/ du dépôt)")
    return parser.parse_args()


def _default_geojson() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here / "config", here.parent / "config"):
        if (candidate / "departements-france.geojson").is_file():
            return candidate / "departements-france.geojson"
    return here / "config" / "departements-france.geojson"


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    geojson = Path(args.geojson) if args.geojson else _default_geojson()
    manifest, run_time = build_risques_arome(
        args.mcv2_base_url, args.days, geojson
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "risques.json"
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    LOGGER.info("Publication prête : %s départements, run %s",
                len(manifest["departments"]), run_time)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
