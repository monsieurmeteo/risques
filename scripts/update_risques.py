#!/usr/bin/env python3
"""Calcule une carte de vigilance météo (11 aléas, échelle propre à chaque
aléa, non officielle) à partir des fichiers départementaux déjà publiés par
le hub `harmonie`.

Contrairement au pipeline HARMONIE (qui décode les GRIB du KNMI), ce script
ne télécharge aucune archive météo : il lit les 96 fichiers
``departements/XX.json`` déjà publiés sur la branche ``data`` du dépôt
``harmonie`` (mêmes données, déjà décodées et compactées), en dérive 10
aléas par département pour J / J+1 / J+2, et republie ``risques.json``.

Trois aléas réutilisent directement des codes de risque déjà calculés par
le pipeline HARMONIE (0-4, cf. ``update_harmonie_france.py::VALUE_COLUMNS``) :
orages, grêle (avec une garde : pas de grêle sans pluie en cours), verglas
(dérivé de ``snow_stick_risk_code``). Les autres sont calculés ici à partir
de seuils numériques explicites propres à chaque aléa (température, cumul
de pluie/neige, rafales, visibilité) — voir ``HAZARD_LEVELS`` et les
constantes ``*_THRESHOLDS*`` ci-dessous pour le détail des paliers.

L'aléa « feu » est un simple cocktail météo (température, humidité, vent,
pluie récente) : il ne remplace pas Météo des forêts et doit toujours être
présenté avec cet avertissement.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9 non pris en charge ici
    ZoneInfo = None  # type: ignore[assignment,misc]


LOGGER = logging.getLogger("risques")
PIPELINE_VERSION = "2.18.0"
PARIS_TZ = ZoneInfo("Europe/Paris") if ZoneInfo is not None else timezone.utc
# La journée météo va de 5 h à 5 h en Europe/Paris. Les heures comprises
# entre minuit et 5 h restent donc rattachées à la journée précédente.
DAY_BOUNDARY_HOUR = 5


def _effective_date(moment: datetime) -> date:
    return (moment.astimezone(PARIS_TZ) - timedelta(hours=DAY_BOUNDARY_HOUR)).date()


DEFAULT_HARMONIE_BASE_URL = (
    "https://raw.githubusercontent.com/monsieurmeteo/harmonie/data"
)
DEFAULT_AROME_BASE_URL = (
    "https://raw.githubusercontent.com/alertesmeteo-hub/arome-meteofrance/data"
)

HAZARDS = (
    "vent",
    "pluie_inondation",
    "orages",
    "grele",
    "chaleur",
    "froid",
    "neige",
    "verglas",
    "brouillard",
    "littoral",
    "feu",
)

# Départements côtiers (métropole, façades Manche/Atlantique/Méditerranée +
# Corse) — seuls ces départements peuvent afficher un niveau non nul pour
# l'aléa Littoral, cf. LITTORAL_THRESHOLDS_KMH plus bas.
LITTORAL_DEPARTMENTS = frozenset({
    "59", "62", "80", "76", "14", "50", "35", "22", "29", "56",
    "44", "85", "17", "33", "40", "64",
    "66", "11", "34", "30", "13", "83", "06",
    "2A", "2B",
})

# Départements littoraux façade Méditerranée + Corse — utilisés pour le
# palier plancher Feu ci-dessous (au-delà de 35°C, le risque feu de forêt y
# est significatif même par vent faible, contrairement au reste du pays).
MEDITERRANEAN_DEPARTMENTS = frozenset({"66", "11", "34", "30", "13", "83", "06", "2A", "2B"})

# Palier Feu plancher forcé pour les départements méditerranéens au-delà de
# cette température, indépendamment de l'humidité/du vent (cf. cocktail
# feu ci-dessous) : correspond à « Orange » sur la rampe de couleurs à 5
# paliers (index 2 = _VIGILANCE_ORANGE). Demandé explicitement : une
# canicule calme (sans tramontane/mistral) à plus de 35°C ne doit pas
# afficher Nul sur ces départements, le risque feu de forêt y étant réel
# même sans vent fort.
FIRE_MEDITERRANEAN_HEAT_THRESHOLD_C = 35.0
FIRE_MEDITERRANEAN_MIN_LEVEL = 2

# L'ordre d'affichage (onglets, grille de détail) suit l'ordre d'insertion
# de ce dict, propagé tel quel dans le JSON (``manifest.hazards``) puis lu
# côté widget via ``Object.keys()`` — le réordonner ici suffit à réordonner
# toute l'interface, aucun changement JS n'est nécessaire.
HAZARD_LABELS = {
    "vent": "Vent",
    "pluie_inondation": "Pluie-inondation",
    "orages": "Orages",
    "grele": "Grêle",
    "chaleur": "Chaleur",
    "froid": "Froid",
    "neige": "Neige",
    "verglas": "Verglas",
    "brouillard": "Brouillard",
    "littoral": "Littoral",
    "feu": "Feu",
}

# Chaque aléa a sa propre échelle (nombre de paliers et libellés) au lieu
# d'une échelle 0-4 unique partagée par tous : demandé explicitement pour
# refléter des critères réels (ex. cumul de pluie en mm, rafales en km/h)
# plutôt qu'un simple code générique. Le palier 0 est toujours « Nul ».
#
# Couleurs officielles de la vigilance Météo-France, relevées directement
# sur les remplissages SVG de la carte en production (vigilance.meteofrance.fr) :
# vert #31aa35, jaune #fff600, orange #ffb82b (confirmés aussi via la charte
# des pictogrammes « comportements à adopter » du même site) ; rouge #cc0000
# (non présent sur la carte au moment du relevé faute de département classé
# rouge ce jour-là, mais confirmé via cette même charte). Météo-France
# s'arrête à 4 couleurs — le palier « Extrême » (violet) au-delà du rouge est
# propre à ce projet, sans équivalent officiel.
_VIGILANCE_VERT = "#31aa35"
_VIGILANCE_JAUNE = "#fff600"
_VIGILANCE_ORANGE = "#ffb82b"
_VIGILANCE_ROUGE = "#cc0000"
_VIGILANCE_EXTREME = "#7b1fa2"

_ANCHORS_OFFICIELS = [_VIGILANCE_VERT, _VIGILANCE_JAUNE, _VIGILANCE_ORANGE, _VIGILANCE_ROUGE]
_ANCHORS_AVEC_EXTREME = _ANCHORS_OFFICIELS + [_VIGILANCE_EXTREME]


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(max(0.0, min(255.0, c))):02x}" for c in rgb)


def _interpolate_ramp(anchors: list[str], count: int) -> list[str]:
    """Répartit ``count`` couleurs le long de la séquence ``anchors``
    (vert → ... → violet), interpolées linéairement entre les points
    d'ancrage les plus proches. Quand ``count`` égale le nombre d'ancres
    (4 ou 5 paliers), chaque couleur retombe exactement sur une teinte
    officielle ; au-delà, les teintes intermédiaires sont dégradées entre
    les mêmes ancres plutôt que redéfinies au hasard."""

    if count == 1:
        return [anchors[0]]
    anchor_rgbs = [_hex_to_rgb(color) for color in anchors]
    ramp = []
    for index in range(count):
        position = index * (len(anchors) - 1) / (count - 1)
        lower = int(position)
        upper = min(lower + 1, len(anchors) - 1)
        fraction = position - lower
        rgb = tuple(
            anchor_rgbs[lower][channel]
            + (anchor_rgbs[upper][channel] - anchor_rgbs[lower][channel]) * fraction
            for channel in range(3)
        )
        ramp.append(_rgb_to_hex(rgb))
    return ramp


_RAMP_4 = _interpolate_ramp(_ANCHORS_OFFICIELS, 4)
_RAMP_5 = _interpolate_ramp(_ANCHORS_AVEC_EXTREME, 5)
_RAMP_7 = _interpolate_ramp(_ANCHORS_AVEC_EXTREME, 7)
_RAMP_8 = _interpolate_ramp(_ANCHORS_AVEC_EXTREME, 8)
_RAMP_9 = _interpolate_ramp(_ANCHORS_AVEC_EXTREME, 9)

# Libellés pour les paliers des aléas à seuils numériques (chaleur/pluie/
# vent/froid/neige). Pour vent/pluie/chaleur, le mot générique (« Faible »,
# « Modéré »...) a été retiré sur demande explicite : le libellé est
# uniquement le seuil chiffré, ex. « (≥ 80 km/h) ». Froid/Neige gardent le
# mot + seuil (non concernés par ce changement).
_TIERS_6 = ["Faible", "Modéré", "Marqué", "Fort", "Très fort", "Extrême"]


def _numeric_tier_labels(
    thresholds: tuple[float, ...], unit: str, below: bool = False, bare: bool = False
) -> list[str]:
    """« Nul », puis un libellé par seuil : « Faible (≥ 28°C) » (ou juste
    « (≥ 28°C) » si ``bare``), etc. ``below`` inverse le comparateur pour un
    aléa qui s'aggrave quand la valeur diminue (froid)."""

    comparator = "≤" if below else "≥"
    labels = ["Nul"]
    for index, threshold in enumerate(thresholds):
        value = f"{threshold:g}"
        criterion = f"({comparator} {value} {unit})"
        if bare:
            labels.append(criterion)
        else:
            labels.append(f"{_TIERS_6[index]} {criterion}")
    return labels


HAZARD_LEVELS: dict[str, list[str]] = {
    # Aléas à seuils numériques (6 paliers + Nul) — construits plus bas à
    # partir des constantes *_THRESHOLDS* pour rester en phase avec le
    # calcul réel plutôt que dupliquer les valeurs ici.
    "vent": [],
    "pluie_inondation": [],
    # Aléas à code de risque HARMONIE (0-4, passthrough) : mêmes libellés
    # génériques pour les deux, dans l'esprit fourni pour les orages.
    "orages": ["Nul", "Faible / Modéré", "Marqué / Fort", "Intense / Violent", "Extrême"],
    "grele": ["Nul", "Faible / Modéré", "Marqué / Fort", "Intense / Violent", "Extrême"],
    "chaleur": [],
    "froid": [],
    "neige": [],
    # Verglas : 3 paliers qualitatifs fournis explicitement.
    "verglas": [
        "Nul",
        "Risque de verglas au sol",
        "Risque de pluie verglaçante",
        "Pluie verglaçante durable",
    ],
    "brouillard": ["Nul", "Faible", "Modéré", "Fort", "Sévère"],
    # Littoral : à seuils numériques (rafales), construit plus bas comme
    # Vent/Pluie à partir de LITTORAL_THRESHOLDS_KMH.
    "littoral": [],
    # Feu : inchangé, 0-4 générique.
    "feu": ["Nul", "Faible", "Modéré", "Fort", "Sévère"],
}


def _ramp_for(level_count: int) -> list[str]:
    if level_count == 5:
        return _RAMP_5
    if level_count == 7:
        return _RAMP_7
    if level_count == 8:
        return _RAMP_8
    if level_count == 9:
        return _RAMP_9
    if level_count == 4:
        return _RAMP_4
    raise ValueError(f"Pas de rampe de couleurs définie pour {level_count} paliers")


def _readable_text_color(background_hex: str) -> str:
    """Texte clair ou sombre selon la luminance du fond (formule YIQ) —
    les teintes officielles vert/rouge/violet sont bien plus saturées que
    l'ancienne rampe pastel, un texte foncé fixe y devenait illisible sur
    les paliers rouge/extrême."""

    r, g, b = _hex_to_rgb(background_hex)
    yiq = (r * 299 + g * 587 + b * 114) / 1000
    return "#1c1f26" if yiq >= 150 else "#ffffff"


def hazard_levels_manifest() -> dict[str, dict[str, dict[str, str]]]:
    """Construit la section ``hazard_levels`` de risques.json : libellé,
    couleur et couleur de texte lisible pour chaque palier de chaque aléa,
    à partir de HAZARD_LEVELS."""

    manifest: dict[str, dict[str, dict[str, str]]] = {}
    for hazard, labels in HAZARD_LEVELS.items():
        ramp = _ramp_for(len(labels))
        manifest[hazard] = {
            str(level): {
                "label": label,
                "color": ramp[level],
                "text_color": _readable_text_color(ramp[level]),
            }
            for level, label in enumerate(labels)
        }
    return manifest


FIRE_DISCLAIMER = (
    "Indice non officiel (cocktail météo chaleur/humidité/vent/pluie). "
    "Ne remplace pas Météo des forêts."
)
LITTORAL_DISCLAIMER = (
    "Indice non officiel (rafales et pression, sans donnée de vagues, "
    "marée ni surcote). Ne remplace pas la Vigilance vagues-submersion "
    "de Météo-France."
)


def department_codes() -> list[str]:
    """Les 96 départements de France métropolitaine (dont Corse en 2A/2B)."""

    codes = [f"{i:02d}" for i in range(1, 96) if i != 20]
    codes += ["2A", "2B"]
    return sorted(codes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--harmonie-base-url",
        default=DEFAULT_HARMONIE_BASE_URL,
        help="Racine des données HARMONIE déjà publiées (branche data)",
    )
    parser.add_argument(
        "--arome-base-url",
        default=DEFAULT_AROME_BASE_URL,
        help=(
            "Racine des données AROME déjà publiées (branche data) — second "
            "avis moyenné avec HARMONIE pour orages/grêle, si disponible"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="build/national",
        help="Dossier de publication à produire",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=2,
        help=(
            "Nombre de journées calendaires à conserver (J, J+1...) — "
            "limité à 2 par défaut : avec une prévision HARMONIE de 48h, "
            "J+2 n'a jamais que quelques heures de données réelles (tôt le "
            "matin), donnant une carte quasi vide pour ce jour-là."
        ),
    )
    parser.add_argument(
        "--current-metadata-url",
        default=None,
        help="risques.json déjà publié, pour éviter un retraitement inutile",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retraite même si le run HARMONIE source n'a pas changé",
    )
    parser.add_argument(
        "--geojson",
        default=str(Path(__file__).resolve().parent.parent / "config" / "departements-france.geojson"),
        help="Contours départementaux, pour repérer les voisins de chaque département",
    )
    return parser.parse_args()


def already_published(session: requests.Session, metadata_url: str | None, run_time: datetime | None) -> bool:
    """True seulement si le run HARMONIE ET la version du pipeline sont
    identiques à ce qui est déjà publié.

    Bug constaté en production : un changement de code (seuils, schéma)
    poussé sans que le run HARMONIE source ait changé restait ignoré
    indéfiniment — le run était déjà « publié » au sens de cette fonction,
    donc le job sortait immédiatement sans jamais republier avec le
    nouveau code. Comparer aussi ``pipeline_version`` force un retraitement
    dès qu'un déploiement de code a eu lieu, même sans nouveau run source.
    """

    if not metadata_url or run_time is None:
        return False
    try:
        response = session.get(metadata_url, timeout=30)
        if response.status_code != 200:
            return False
        previous = response.json()
    except (requests.RequestException, ValueError):
        return False
    if previous.get("pipeline_version") != PIPELINE_VERSION:
        return False
    previous_run = previous.get("run_time")
    current_run = run_time.isoformat().replace("+00:00", "Z")
    return previous_run == current_run


def fetch_json(session: requests.Session, url: str) -> dict[str, Any]:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


@dataclass
class DepartmentSeries:
    code: str
    times: list[datetime]
    columns: dict[str, int]
    # forecast[i] est la matrice (points x colonnes) pour l'heure times[i]
    forecast: list[np.ndarray]
    # Altitude (m) de chaque point de grille, même ordre/longueur que les
    # lignes de ``forecast`` — vient de ``payload["points"]``, une valeur
    # fixe par point (pas par heure), séparée des colonnes de prévision.
    altitudes: np.ndarray

    def column(self, step_index: int, name: str) -> np.ndarray:
        index = self.columns.get(name)
        if index is None:
            return np.asarray([], dtype=np.float64)
        matrix = self.forecast[step_index]
        if matrix.size == 0:
            return np.asarray([], dtype=np.float64)
        return matrix[:, index]


def load_department_series(
    session: requests.Session, base_url: str, code: str
) -> DepartmentSeries | None:
    url = f"{base_url}/departements/{code}.json"
    try:
        payload = fetch_json(session, url)
    except requests.RequestException as error:
        LOGGER.warning("Département %s indisponible (%s)", code, error)
        return None
    if payload.get("status") != "ok":
        LOGGER.warning("Département %s : statut invalide", code)
        return None

    value_names: list[str] = payload["columns"]["values"]
    columns = {name: index for index, name in enumerate(value_names)}

    # Altitude par point de grille (``payload["points"]``, schéma décrit par
    # ``payload["columns"]["points"]``) — utilisée pour exclure les sommets
    # non représentatifs des zones habitées de certains calculs (rafales,
    # gel). Absente ou mal formée : on retombe sur des NaN, qui ne filtrent
    # rien (comportement identique à avant l'ajout de ce filtre).
    points_schema: list[str] = ((payload.get("columns") or {}).get("points")) or []
    points_raw = payload.get("points") or []
    if "model_altitude_m" in points_schema and points_raw:
        alt_index = points_schema.index("model_altitude_m")
        altitudes = np.asarray(
            [np.nan if p[alt_index] is None else p[alt_index] for p in points_raw],
            dtype=np.float64,
        )
    else:
        altitudes = np.full(len(points_raw), np.nan, dtype=np.float64)

    times: list[datetime] = []
    forecast: list[np.ndarray] = []
    for iso_time, rows in payload.get("forecast", []):
        try:
            valid_time = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
        except ValueError:
            continue
        times.append(valid_time)
        if rows:
            matrix = np.asarray(
                [[np.nan if v is None else v for v in row] for row in rows],
                dtype=np.float64,
            )
        else:
            matrix = np.empty((0, len(value_names)), dtype=np.float64)
        forecast.append(matrix)

    return DepartmentSeries(
        code=code, times=times, columns=columns, forecast=forecast, altitudes=altitudes
    )


def _nanpercentile_high(values: np.ndarray, percentile: float = 90.0) -> float:
    """Perçentile haut plutôt que le max strict : un département compte des
    dizaines à centaines de points HARMONIE, et prendre le max fait qu'UNE
    seule commune en pointe fait passer tout le département au niveau
    maximal, pour toute la journée (constaté en production : 58% des
    départements en Orages « Sévère » le même jour). Le 90e centile reste
    sensible à un risque réellement étendu, sans être piloté par un seul
    point isolé."""

    finite = values[np.isfinite(values)] if values.size else values
    return float(np.percentile(finite, percentile)) if finite.size else float("nan")


def _nanpercentile_low(values: np.ndarray, percentile: float = 10.0) -> float:
    """Symétrique de ``_nanpercentile_high`` pour les grandeurs qui
    s'aggravent quand elles diminuent (visibilité, température, humidité)."""

    finite = values[np.isfinite(values)] if values.size else values
    return float(np.percentile(finite, percentile)) if finite.size else float("nan")


def _risk_column_level(values: np.ndarray, cap: int = 4, percentile: float = 90.0) -> int:
    finite = values[np.isfinite(values)] if values.size else values
    if not finite.size:
        return 0
    return int(min(cap, max(0, round(_nanpercentile_high(values, percentile)))))


def _localized_risk_level(
    values: np.ndarray, cap: int = 4, min_points: int = 5, min_fraction: float = 0.03
) -> int:
    """Pour orages/grêle : phénomènes intrinsèquement localisés, pas
    étalés uniformément sur tout un département comme une canicule ou un
    coup de froid — un perçentile (même abaissé à 75) exige qu'au moins
    25% du département soit touché, ce qui s'est révélé encore trop
    strict en production : la Seine-et-Marne restait « Nul » alors que
    16% de ses points de grille atteignaient « Marqué/Fort » ou plus (7
    points sur 202 en « Intense/Violent »), entourée de tous côtés par des
    départements classés — Météo-France elle-même déclenche une vigilance
    département entier dès qu'une cellule confirmée le traverse, sans
    exiger qu'une fraction donnée du territoire soit couverte.

    Ici, un niveau compte pour le département dès qu'au moins
    ``min_points`` points (ou ``min_fraction`` du total, le plus grand des
    deux) l'atteignent — un plancher fixe, donc insensible à 1-2 points
    isolés (le bug d'origine : un seul point aberrant faisait basculer
    58% des départements en « Sévère » le même jour), tout en reconnaissant
    une cellule orageuse réelle dès qu'elle couvre une fraction minoritaire
    mais substantielle du territoire."""

    finite = values[np.isfinite(values)] if values.size else values
    if not finite.size:
        return 0
    threshold_count = max(min_points, int(np.ceil(finite.size * min_fraction)))
    for candidate in range(cap, 0, -1):
        if int(np.sum(finite >= candidate)) >= threshold_count:
            return candidate
    return 0


def load_arome_daily_levels(
    session: requests.Session, base_url: str, code: str, day_count: int, today: date
) -> dict[str, dict[str, int]] | None:
    """Second avis pour orages/grêle : le module AROME (dépôt et pipeline
    indépendants, résolution ~1,3 km contre 5,5 km pour HARMONIE) publie
    déjà ses propres ``thunder_risk_code``/``hail_risk_code`` sur la même
    échelle 0-4, mais à partir de critères différents — une CAPE directe
    (pas approximée comme côté HARMONIE) et une réflectivité radar
    modélisée (que HARMONIE ne fournit pas du tout). Les deux modèles se
    corrigent mutuellement plutôt que de dépendre d'un seul.

    Même agrégation que côté HARMONIE (_localized_risk_level par heure,
    max sur la journée météo 6h-6h) pour rester comparable. None si AROME
    est indisponible pour ce département ou n'a pas ces colonnes — le
    pipeline continue alors avec HARMONIE seul, sans faire échouer le run."""

    try:
        payload = fetch_json(session, f"{base_url}/departements/{code}.json")
    except (requests.RequestException, ValueError):
        return None
    if payload.get("status") != "ok":
        return None
    value_names: list[str] = ((payload.get("columns") or {}).get("values")) or []
    if "thunder_risk_code" not in value_names or "hail_risk_code" not in value_names:
        return None
    thunder_index = value_names.index("thunder_risk_code")
    hail_index = value_names.index("hail_risk_code")

    by_day: dict[str, dict[str, int]] = {}
    for entry in payload.get("forecast") or []:
        try:
            iso_time, rows = entry
            valid_time = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        date_str = _effective_date(valid_time).isoformat()
        thunder_values = np.asarray(
            [row[thunder_index] for row in rows if row[thunder_index] is not None],
            dtype=np.float64,
        )
        hail_values = np.asarray(
            [row[hail_index] for row in rows if row[hail_index] is not None],
            dtype=np.float64,
        )
        slot = by_day.setdefault(date_str, {"orages": 0, "grele": 0})
        slot["orages"] = max(slot["orages"], _localized_risk_level(thunder_values))
        slot["grele"] = max(slot["grele"], _localized_risk_level(hail_values))

    ordered_dates = [
        (today + timedelta(days=offset)).isoformat() for offset in range(day_count)
    ]
    return {date_str: by_day[date_str] for date_str in ordered_dates if date_str in by_day}


def _blend_with_arome(daily: list[dict[str, Any]], arome_levels: dict[str, dict[str, int]]) -> None:
    """Moyenne arrondie entre le niveau HARMONIE déjà calculé et le niveau
    AROME du même jour, pour orages et grêle uniquement — les autres aléas
    n'ont pas de second avis indépendant disponible. Un jour sans donnée
    AROME correspondante garde simplement le niveau HARMONIE seul."""

    for day_entry in daily:
        arome_day = arome_levels.get(day_entry.get("date"))
        if not arome_day:
            continue
        hazards = day_entry.get("hazards") or {}
        for hazard in ("orages", "grele"):
            arome_level = arome_day.get(hazard)
            if arome_level is None:
                continue
            harmonie_level = hazards.get(hazard, 0)
            hazards[hazard] = round((harmonie_level + arome_level) / 2)


def _threshold_level(value: float, thresholds: tuple[float, ...]) -> int:
    """Paliers croissants : renvoie le niveau (0 à len(thresholds)) atteint
    par ``value``. ``thresholds`` doit être trié en ordre croissant."""

    if not np.isfinite(value):
        return 0
    level = 0
    for threshold in thresholds:
        if value >= threshold:
            level += 1
    return level


def _threshold_level_below(value: float, thresholds: tuple[float, ...]) -> int:
    """Comme ``_threshold_level`` mais pour un aléa qui s'aggrave quand la
    valeur DIMINUE (visibilité, température minimale). ``thresholds`` doit
    être trié en ordre DÉCROISSANT (du moins sévère au plus sévère)."""

    if not np.isfinite(value):
        return 0
    level = 0
    for threshold in thresholds:
        if value <= threshold:
            level += 1
    return level


# Seuils numériques (ascendants) fournis explicitement pour chaque aléa —
# le nombre de valeurs fixe le nombre de paliers. ``_threshold_level``
# incrémente le niveau à chaque seuil atteint ou dépassé, donc ces tuples
# doivent rester en ordre croissant. Chaleur/Pluie/Vent sont revenus à 7
# seuils (dont Pluie avec un nouveau palier bas à 5mm) après une version
# intermédiaire à 6 ; Froid/Neige restent à 6, non concernés par ce
# dernier ajustement.
CHALEUR_THRESHOLDS = (25.0, 28.0, 31.0, 34.0, 37.0, 40.0, 45.0)
# 8 seuils (pas 7) : contrairement à Chaleur/Vent, Pluie a gagné un palier
# supplémentaire (500mm) en plus du nouveau palier bas (5mm) — le 500mm
# avait été supprimé par erreur lors de l'ajout du 5mm, alors qu'il devait
# rester comme palier « Extrême » le plus haut.
PLUIE_THRESHOLDS_MM = (5.0, 15.0, 30.0, 50.0, 80.0, 150.0, 300.0, 500.0)
VENT_THRESHOLDS_KMH = (60.0, 80.0, 90.0, 100.0, 110.0, 130.0, 150.0, 180.0)
# Littoral : indice non officiel (pas de vagues/marée/surcote dans HARMONIE),
# cocktail rafale + pression basse sur les départements côtiers uniquement
# (cf. LITTORAL_DEPARTMENTS) — mêmes seuils de rafale que Vent, avec un cran
# supplémentaire si une dépression marquée (signature de tempête) accompagne
# le vent, cf. hourly_hazard_levels.
LITTORAL_THRESHOLDS_KMH = (60.0, 70.0, 80.0, 90.0, 110.0, 130.0, 150.0)
LITTORAL_STORM_PRESSURE_HPA = 990.0
NEIGE_THRESHOLDS_CM = (1.0, 3.0, 7.0, 15.0, 30.0, 50.0)
# ``_threshold_level_below`` a besoin de l'ordre inverse (du seuil le plus
# « chaud »/le moins sévère au plus froid) — cf. sa docstring.
FROID_THRESHOLDS_BELOW = (-3.0, -6.0, -10.0, -15.0, -20.0, -30.0)

# Les libellés de légende intègrent directement le seuil réel plutôt qu'un
# mot seul — demandé explicitement, « qu'il faut mettre en légende ».
# Pluie/Vent : uniquement le seuil, sans mot générique (``bare``).
# Complète les entrées vides laissées dans HAZARD_LEVELS plus haut (qui
# doivent rester en phase avec ces constantes plutôt que dupliquer les
# seuils à deux endroits).
HAZARD_LEVELS["pluie_inondation"] = _numeric_tier_labels(PLUIE_THRESHOLDS_MM, "mm", bare=True)
HAZARD_LEVELS["vent"] = _numeric_tier_labels(VENT_THRESHOLDS_KMH, "km/h", bare=True)
HAZARD_LEVELS["littoral"] = _numeric_tier_labels(LITTORAL_THRESHOLDS_KMH, "km/h", bare=True)
HAZARD_LEVELS["neige"] = _numeric_tier_labels(NEIGE_THRESHOLDS_CM, "cm")
HAZARD_LEVELS["froid"] = _numeric_tier_labels(FROID_THRESHOLDS_BELOW, "°C", below=True)

# Chaleur : mot + seuil rétabli (contrairement à Pluie/Vent, restés
# « bare ») — mais au féminin (« la chaleur ») : Modérée/Marquée/Forte/
# Très forte, pas Modéré/Marqué/Fort/Très fort.
_CHALEUR_TIER_NAMES = ["Faible", "Modérée", "Marquée", "Forte", "Très forte", "Intense", "Extrême"]
HAZARD_LEVELS["chaleur"] = ["Nul"] + [
    f"{name} (≥ {threshold:g} °C)"
    for name, threshold in zip(_CHALEUR_TIER_NAMES, CHALEUR_THRESHOLDS)
]


# Altitude au-delà de laquelle un point de grille est exclu de certains
# calculs : un sommet ou un col en très haute montagne n'est représentatif
# d'aucune zone habitée, et produit des valeurs extrêmes mais non
# pertinentes pour une carte de vigilance grand public — constaté en
# production (rafale à 260 km/h relevée en Haute-Savoie, sur un point de
# grille en haute montagne).
VENT_MAX_ALTITUDE_M = 2000.0
GEL_MAX_ALTITUDE_M = 1500.0


def _filtered_by_altitude(
    values: np.ndarray, altitudes: np.ndarray, max_altitude_m: float
) -> np.ndarray:
    if values.size != altitudes.size:
        # Décalage inattendu (schéma de points absent/différent) : on ne
        # filtre pas plutôt que de fausser silencieusement le calcul.
        return values
    mask = np.isfinite(altitudes) & (altitudes <= max_altitude_m)
    if not mask.any():
        # Département entièrement au-dessus du seuil (improbable) : mieux
        # vaut une valeur non filtrée qu'aucune donnée du tout.
        return values
    return values[mask]


def hourly_hazard_levels(
    series: DepartmentSeries,
    step_index: int,
    cumulative_precip_mm: float,
    day_precip_mm: float,
    day_snow_cm: float,
) -> tuple[dict[str, int], dict[str, float]]:
    """Niveau de chaque aléa (dict) + valeurs brutes record de l'heure
    (2e dict : max/min réels, pas le perçentile utilisé pour les niveaux)
    pour un département, à une échéance donnée.

    ``cumulative_precip_mm`` ne se réinitialise jamais (utilisé par Feu,
    proxy de sécheresse récente) ; ``day_precip_mm``/``day_snow_cm`` sont
    des cumuls glissants remis à zéro à chaque changement de journée
    calendaire Europe/Paris (utilisés par Pluie-inondation et Neige, dont
    les seuils sont désormais des cumuls en mm/cm et non plus une valeur
    instantanée).
    """

    def col(name: str) -> np.ndarray:
        return series.column(step_index, name)

    temperature = col("temperature_c")
    humidity = col("humidity_pct")
    wind_speed = col("wind_speed_kmh")
    wind_gust = col("wind_gust_kmh")
    visibility = col("visibility_km")
    precipitation_now = col("precipitation_mm")
    pressure = col("pressure_hpa")

    # Rafales (Vent) et température minimale (Froid/gel) : un sommet ou un
    # col de haute montagne n'est représentatif d'aucune zone habitée —
    # exclu de ces deux calculs seulement (pas de la température maximale,
    # une altitude élevée ne produit pas de chaleur artificiellement
    # élevée, donc rien à corriger côté Chaleur).
    wind_gust_populated = _filtered_by_altitude(wind_gust, series.altitudes, VENT_MAX_ALTITUDE_M)
    temperature_populated_low = _filtered_by_altitude(temperature, series.altitudes, GEL_MAX_ALTITUDE_M)

    # Perçentiles plutôt que max/min strict : même correction que pour les
    # aléas à code de risque (cf. _nanpercentile_high) — une seule commune
    # ne doit pas suffire à faire basculer tout le département.
    max_temperature = _nanpercentile_high(temperature)
    min_temperature = _nanpercentile_low(temperature_populated_low)
    min_humidity = _nanpercentile_low(humidity)
    max_wind = _nanpercentile_high(wind_speed)
    max_gust = _nanpercentile_high(wind_gust_populated)
    min_visibility = _nanpercentile_low(visibility)
    precip_now_repr = _nanpercentile_high(precipitation_now)
    min_pressure = _nanpercentile_low(pressure)

    # Brouillard : la visibilité seule ne suffit pas à distinguer un vrai
    # brouillard (air saturé, calme) d'une simple visibilité réduite par la
    # pluie, la neige ou la poussière — et le brouillard radiatif classique
    # est un phénomène de demi-saison froide, rare en plein été hors
    # brouillard côtier d'advection.
    dewpoint = col("dewpoint_c")
    dewpoint_spread = temperature - dewpoint
    fog_humidity = _nanpercentile_high(humidity)
    fog_spread = _nanpercentile_low(dewpoint_spread)
    fog_month = (
        series.times[step_index].astimezone(PARIS_TZ).month
        if step_index < len(series.times)
        else None
    )

    # Résumé national « records du jour » : mêmes valeurs perçentile que les
    # niveaux d'alerte ci-dessus, pas le max/min brut d'un point isolé.
    # Constaté en production : un point de grille au cisaillement extrême
    # donnait 145 km/h de rafale « record national » en Côte-d'Or alors que
    # le niveau Vent affiché pour ce même département n'était que 2/7 (seuil
    # 90 km/h) — un même département affichant deux valeurs incohérentes
    # entre elles selon l'endroit de la page.
    raw_extremes = {
        "max_temperature": max_temperature,
        "min_temperature": min_temperature,
        "max_gust": max_gust,
    }

    # Cocktail feu : chaleur, air sec ET vent doivent être réunis
    # SIMULTANÉMENT — un seul facteur seul ne doit jamais suffire (constaté
    # en production : des départements de montagne classés « Faible »
    # uniquement parce qu'il n'avait pas plu récemment dans le modèle, sans
    # aucune chaleur ni air sec ni vent réels ce jour-là — l'ancien score
    # additif accordait ce point de sécheresse indépendamment du reste).
    fire_level = 0
    if (
        np.isfinite(max_temperature) and np.isfinite(min_humidity) and np.isfinite(max_wind)
        and max_temperature >= 30.0 and min_humidity <= 40.0 and max_wind >= 20.0
    ):
        fire_level = 1
        if max_temperature >= 33.0 and min_humidity <= 30.0 and max_wind >= 30.0:
            fire_level = 2
        if max_temperature >= 36.0 and min_humidity <= 25.0 and max_wind >= 40.0:
            fire_level = 3
        if max_temperature >= 39.0 and min_humidity <= 20.0 and max_wind >= 50.0:
            fire_level = 4
        # Sécheresse récente confirmée (cumul de pluie quasi nul depuis le
        # début de la série, à partir d'une trentaine d'heures pour éviter
        # l'effet de démarrage à zéro du compteur) : aggrave d'un cran un
        # risque déjà présent par chaleur+air sec+vent, mais ne peut plus,
        # à elle seule, faire naître un risque là où il n'y en a pas.
        if step_index >= 24 and cumulative_precip_mm < 1.0:
            fire_level = min(4, fire_level + 1)

    # Plancher méditerranéen : au-delà de 35°C sur ces départements, le
    # risque feu de forêt est significatif même si le vent ne franchit pas
    # le seuil de 20 km/h exigé par le cocktail ci-dessus (ex. canicule
    # calme, sans tramontane/mistral) — cf. constantes plus haut.
    if (
        series.code in MEDITERRANEAN_DEPARTMENTS
        and np.isfinite(max_temperature)
        and max_temperature > FIRE_MEDITERRANEAN_HEAT_THRESHOLD_C
    ):
        fire_level = max(fire_level, FIRE_MEDITERRANEAN_MIN_LEVEL)

    # Orages/grêle : seuil par nombre de points plutôt que perçentile —
    # cf. _localized_risk_level (phénomènes intrinsèquement localisés).
    orages_level = _localized_risk_level(col("thunder_risk_code"))
    grele_level = _localized_risk_level(col("hail_risk_code"))
    # Grêle : météorologiquement impossible sans précipitation en cours
    # (la grêle est une forme de précipitation convective) — un code de
    # risque non nul sans pluie mesurable à cette heure est ignoré plutôt
    # que reporté tel quel.
    if not np.isfinite(precip_now_repr) or precip_now_repr < 0.1:
        grele_level = 0

    # Littoral : indice non officiel, restreint aux départements côtiers
    # (cf. LITTORAL_DEPARTMENTS) — rafale (même filtre altitude que Vent) et
    # un cran de plus si une dépression marquée accompagne le vent (signature
    # de tempête, proxy de surcote en l'absence de toute donnée de marée ou
    # de vagues dans HARMONIE).
    littoral_level = 0
    if series.code in LITTORAL_DEPARTMENTS:
        littoral_level = _threshold_level(max_gust, LITTORAL_THRESHOLDS_KMH)
        if littoral_level > 0 and np.isfinite(min_pressure) and min_pressure <= LITTORAL_STORM_PRESSURE_HPA:
            littoral_level = min(len(LITTORAL_THRESHOLDS_KMH), littoral_level + 1)

    fog_level = _threshold_level_below(min_visibility, (1.0, 0.5, 0.2, 0.1))
    if fog_level > 0:
        is_saturated = (
            np.isfinite(fog_humidity) and fog_humidity >= 92.0
        ) or (np.isfinite(fog_spread) and fog_spread <= 2.0)
        is_calm = np.isfinite(max_wind) and max_wind <= 15.0
        if not (is_saturated and is_calm):
            # Visibilité basse sans air saturé ni calme plat : ce n'est pas
            # du brouillard (pluie forte, neige, poussière...).
            fog_level = 0
        elif fog_month in (6, 7, 8) and not (
            np.isfinite(fog_humidity) and fog_humidity >= 97.0 and max_wind <= 8.0
        ):
            # Juin-août : le brouillard radiatif classique est rare (hors
            # brouillard côtier d'advection) — signal exigé plus net,
            # sinon plafonné à Faible.
            fog_level = min(fog_level, 1)

    hazards = {
        "orages": orages_level,
        "grele": grele_level,
        # Cumul de pluie du jour (mm), pas un code instantané : le niveau à
        # une heure donnée reflète le cumul depuis le début de la journée
        # météo (6 h, cf. _effective_date) jusqu'à cette heure-là (la frise
        # progresse donc en escalier croissant sur la journée, comme un
        # cumul réel).
        "pluie_inondation": _threshold_level(day_precip_mm, PLUIE_THRESHOLDS_MM),
        # Rafales (et non le vent moyen) : seuils fournis en km/h de rafale.
        "vent": _threshold_level(max_gust, VENT_THRESHOLDS_KMH),
        "neige": _threshold_level(day_snow_cm, NEIGE_THRESHOLDS_CM),
        "verglas": _risk_column_level(col("snow_stick_risk_code"), cap=3),
        "chaleur": _threshold_level(max_temperature, CHALEUR_THRESHOLDS),
        "froid": _threshold_level_below(min_temperature, FROID_THRESHOLDS_BELOW),
        "brouillard": fog_level,
        "littoral": littoral_level,
        "feu": fire_level,
    }
    return hazards, raw_extremes


def build_department_risk(
    series: DepartmentSeries, day_count: int, today: date
) -> tuple[dict[str, Any], dict[str, dict[str, float]]] | None:
    """Renvoie (objet publiable {daily, hourly}, records bruts par jour).

    Le 2e élément (``{date: {max_temperature, min_temperature, max_gust,
    total_precip_mm}}``) n'est PAS publié tel quel dans risques.json (pas de
    valeurs brutes par département, seulement des niveaux) — il sert juste
    à ``build_risques`` à calculer le résumé national (records du jour,
    département par département)."""

    if not series.times:
        return None

    # Cumul glissant des précipitations HARMONIE (proxy de sécheresse
    # récente pour l'aléa Feu) : on ne dispose pas d'observations passées
    # dans ce pipeline, seulement des prévisions — on cumule donc depuis le
    # début de l'échéance disponible.
    running_precip = 0.0
    # Cumuls du jour courant (mm de pluie, cm de neige) pour Pluie-inondation
    # et Neige : remis à zéro à chaque changement de journée météo (6h,
    # cf. _effective_date), contrairement à ``running_precip`` ci-dessus
    # (qui ne se réinitialise jamais, propre au proxy de sécheresse de Feu).
    running_day_precip = 0.0
    running_day_snow = 0.0
    current_local_date: str | None = None
    hourly: list[dict[str, Any]] = []
    raw_by_date: dict[str, dict[str, float]] = {}
    for step_index, valid_time in enumerate(series.times):
        local_date = _effective_date(valid_time).isoformat()
        if local_date != current_local_date:
            running_day_precip = 0.0
            running_day_snow = 0.0
            current_local_date = local_date

        precip = series.column(step_index, "precipitation_mm")
        finite_precip = precip[np.isfinite(precip)] if precip.size else precip
        running_precip += float(np.max(finite_precip)) if finite_precip.size else 0.0

        precip_repr = _nanpercentile_high(precip) if precip.size else float("nan")
        running_day_precip += precip_repr if np.isfinite(precip_repr) else 0.0

        snow_fresh = series.column(step_index, "snow_fresh_cm")
        snow_repr = _nanpercentile_high(snow_fresh) if snow_fresh.size else float("nan")
        running_day_snow += snow_repr if np.isfinite(snow_repr) else 0.0

        levels, raw = hourly_hazard_levels(
            series, step_index, running_precip, running_day_precip, running_day_snow
        )
        hourly.append(
            {
                "time": valid_time.astimezone(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                "hazards": levels,
            }
        )

        day_raw = raw_by_date.setdefault(
            local_date,
            {"max_temperature": float("-inf"), "min_temperature": float("inf"), "max_gust": float("-inf")},
        )
        if np.isfinite(raw["max_temperature"]):
            day_raw["max_temperature"] = max(day_raw["max_temperature"], raw["max_temperature"])
        if np.isfinite(raw["min_temperature"]):
            day_raw["min_temperature"] = min(day_raw["min_temperature"], raw["min_temperature"])
        if np.isfinite(raw["max_gust"]):
            day_raw["max_gust"] = max(day_raw["max_gust"], raw["max_gust"])
        # Le cumul du jour ne fait qu'augmenter (remis à 0 à chaque
        # changement de date) : sa valeur à la dernière heure du jour EST
        # le total du jour.
        day_raw["total_precip_mm"] = running_day_precip

    # Regroupement par journée météo locale (6 h → 6 h).
    days: dict[str, list[dict[str, Any]]] = {}
    for entry in hourly:
        local_date = _effective_date(
            datetime.fromisoformat(entry["time"].replace("Z", "+00:00"))
        ).isoformat()
        days.setdefault(local_date, []).append(entry)

    # J0 doit toujours être la journée météo en cours (6 h → 6 h, Europe/
    # Paris), même si le run HARMONIE source est en retard et ne couvre pas
    # encore (ou plus) la journée en cours : un département sans données
    # pour une date cible reçoit simplement des niveaux à 0 plutôt que de
    # décaler tout l'axe J/J+1/J+2. ``today`` est calculé une seule fois
    # pour tout le run (et non par département) pour que les 96
    # départements du même run partagent exactement la même date J0, même
    # si le traitement chevauche la limite des 6h.
    ordered_dates = [
        (today + timedelta(days=offset)).isoformat() for offset in range(day_count)
    ]
    daily: list[dict[str, Any]] = []
    for date_str in ordered_dates:
        entries = days.get(date_str, [])
        day_levels: dict[str, int] = {}
        for hazard in HAZARDS:
            day_levels[hazard] = max(
                (entry["hazards"][hazard] for entry in entries), default=0
            )
        daily.append({"date": date_str, "hazards": day_levels})

    return {"daily": daily, "hourly": hourly}, raw_by_date


def _department_centroids(geojson_path: Path) -> dict[str, tuple[float, float]]:
    """Centroïde grossier (moyenne des sommets, pas un vrai centroïde
    pondéré par aire) de chaque département, à partir du même GeoJSON que
    la carte — suffisant pour classer les départements par proximité, pas
    pour un calcul géométrique précis. Fichier absent ou illisible :
    dict vide (aucun voisinage connu, la correction de cohérence
    spatiale ci-dessous devient alors un no-op plutôt qu'une erreur)."""

    try:
        payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        LOGGER.warning("GeoJSON départements illisible (%s) : %s", geojson_path, error)
        return {}

    centroids: dict[str, tuple[float, float]] = {}
    for feature in payload.get("features") or []:
        code = str((feature.get("properties") or {}).get("code") or "").upper()
        if not code:
            continue
        geometry = feature.get("geometry") or {}
        rings: list[list[list[float]]] = []
        if geometry.get("type") == "Polygon":
            rings = geometry.get("coordinates") or []
        elif geometry.get("type") == "MultiPolygon":
            for polygon in geometry.get("coordinates") or []:
                rings.extend(polygon)
        points = [point for ring in rings for point in ring]
        if not points:
            continue
        centroids[code] = (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
    return centroids


def _nearest_neighbors(
    centroids: dict[str, tuple[float, float]], count: int = 6
) -> dict[str, list[str]]:
    """Les ``count`` départements les plus proches (centroïde à centroïde)
    de chaque département — un proxy de voisinage géographique qui évite
    d'avoir à calculer une vraie adjacence de polygones (pas de dépendance
    de géométrie supplémentaire dans ce pipeline volontairement léger)."""

    codes = list(centroids.keys())
    neighbors: dict[str, list[str]] = {}
    for code in codes:
        lon0, lat0 = centroids[code]
        ranked = sorted(
            (other for other in codes if other != code),
            key=lambda other: (centroids[other][0] - lon0) ** 2
            + (centroids[other][1] - lat0) ** 2,
        )
        neighbors[code] = ranked[:count]
    return neighbors


def _fill_isolated_green(
    departments: dict[str, Any],
    neighbors: dict[str, list[str]],
    hazard: str = "orages",
    min_neighbor_level: int = 2,
    max_lenient_neighbors: int = 1,
) -> None:
    """Un département classé « Nul » alors que la quasi-totalité de ses
    plus proches voisins sont classés « Marqué/Fort » ou pire reste
    suspect plutôt qu'un vrai répit — bord de cellule mal capté par les
    points de grille de CE département précisément, pas une preuve que le
    danger s'arrête net à sa frontière (constaté en production : la
    Seine-et-Marne et la Dordogne restaient « Nul », chacune entourée de
    tous côtés par des départements classés). Relevé à « Faible/Modéré »
    (1), pas au niveau des voisins : on sait seulement qu'un vrai répit
    total y est peu probable, pas que ce département vit exactement la
    même sévérité qu'eux."""

    # Deux passes (lecture puis écriture) plutôt qu'une seule : muter
    # ``departments`` pendant qu'on le parcourt ferait dépendre le résultat
    # de l'ordre d'itération — un département tout juste relevé pourrait
    # alors, dans la même passe, faire basculer un voisin qui ne l'aurait
    # pas été sur la base des niveaux d'origine, en cascade.
    corrections: list[tuple[str, int, int]] = []
    for code, risk in departments.items():
        own_neighbors = [n for n in neighbors.get(code, []) if n in departments]
        if len(own_neighbors) < 3:
            continue
        for day_index, day_entry in enumerate(risk.get("daily") or []):
            hazards = day_entry.get("hazards") or {}
            if hazards.get(hazard, 0) != 0:
                continue
            neighbor_levels = []
            for neighbor_code in own_neighbors:
                neighbor_daily = departments[neighbor_code].get("daily") or []
                if day_index < len(neighbor_daily):
                    neighbor_hazards = neighbor_daily[day_index].get("hazards") or {}
                    neighbor_levels.append(neighbor_hazards.get(hazard, 0))
            if len(neighbor_levels) < 3:
                continue
            lenient = sum(1 for level in neighbor_levels if level < min_neighbor_level)
            if lenient <= max_lenient_neighbors:
                corrections.append((code, day_index, 1))

    for code, day_index, new_level in corrections:
        departments[code]["daily"][day_index]["hazards"][hazard] = new_level


def department_display_name(series: DepartmentSeries) -> str | None:
    # Le nom du département n'est pas publié tel quel par HARMONIE (seules
    # les communes le sont) ; les communes elles-mêmes ne portent pas le nom
    # du département. On le laisse à None ici : le plugin WordPress associe
    # le code département à un nom via sa propre table statique (96 lignes,
    # déjà nécessaire pour l'INSEE, ne bouge jamais).
    return None


def harmonie_run_time(session: requests.Session, base_url: str) -> datetime | None:
    """Lit le run HARMONIE courant depuis l'index léger, sans télécharger
    les 96 fichiers départementaux — sert uniquement à décider si un
    retraitement est nécessaire avant de faire le travail complet."""

    try:
        index = fetch_json(session, f"{base_url}/index.json")
    except requests.RequestException:
        return None
    run_time_text = (index.get("model") or {}).get("run_time")
    if not run_time_text:
        return None
    try:
        return datetime.fromisoformat(run_time_text.replace("Z", "+00:00"))
    except ValueError:
        return None


def build_risques(
    base_url: str,
    day_count: int,
    geojson_path: Path | None = None,
    arome_base_url: str | None = None,
) -> tuple[dict[str, Any], datetime | None]:
    session = requests.Session()
    session.headers["User-Agent"] = "alertesmeteo-hub-risques/1.0"

    today = _effective_date(datetime.now(timezone.utc))
    departments: dict[str, Any] = {}
    # {date: {field: (best_value, department_code)}} — alimenté au fil des
    # départements pour calculer le résumé national (records du jour, avec
    # le département qui les détient) sans tout garder en mémoire deux fois.
    national_by_date: dict[str, dict[str, tuple[float, str]]] = {}
    run_time: datetime | None = None
    missing = 0
    for code in department_codes():
        series = load_department_series(session, base_url, code)
        if series is None:
            missing += 1
            continue
        result = build_department_risk(series, day_count, today)
        if result is None:
            missing += 1
            continue
        risk, raw_by_date = result
        if arome_base_url:
            arome_levels = load_arome_daily_levels(
                session, arome_base_url, code, day_count, today
            )
            if arome_levels:
                _blend_with_arome(risk["daily"], arome_levels)
        departments[code] = risk
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

        if run_time is None and series.times:
            run_time = series.times[0]
        LOGGER.info("Département %s : %s échéances traitées", code, len(series.times))

    if missing:
        LOGGER.warning(
            "%s département(s) sur %s n'ont pas pu être traités",
            missing,
            len(department_codes()),
        )
    if len(departments) < 80:
        raise RuntimeError(
            f"Trop de départements manquants ({len(departments)}/96) — "
            "le hub harmonie n'est probablement pas encore à jour."
        )

    # Cohérence spatiale des orages : un département isolé à « Nul »
    # entouré de départements marqués/forts est relevé au minimum observé
    # chez ses voisins — cf. _fill_isolated_green. Sans GeoJSON exploitable,
    # aucun voisinage n'est connu et cette étape ne change rien (no-op).
    if geojson_path is not None:
        centroids = _department_centroids(geojson_path)
        if centroids:
            neighbors = _nearest_neighbors(centroids)
            _fill_isolated_green(departments, neighbors)

    # Résumé national par jour : maxi/mini de température, rafale maxi,
    # cumul de pluie maxi, chacun avec le département qui le détient — le
    # plugin résout le code en nom via sa table de départements déjà
    # nécessaire pour l'affichage de la carte.
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

        national_summary.append(
            {
                "date": date_str,
                "max_temperature": field("max_temperature"),
                "min_temperature": field("min_temperature"),
                "max_gust": field("max_gust"),
                "max_precip": field("total_precip_mm"),
            }
        )

    manifest = {
        "schema_version": 1,
        "status": "ok",
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "run_time": (
            run_time.isoformat().replace("+00:00", "Z") if run_time else None
        ),
        "source": {
            "model": "HARMONIE-AROME Cy43 (KNMI)",
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


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = "alertesmeteo-hub-risques/1.0"
    if not args.force:
        current_run = harmonie_run_time(session, args.harmonie_base_url)
        if already_published(session, args.current_metadata_url, current_run):
            LOGGER.info(
                "Run HARMONIE %s déjà publié dans risques.json, rien à faire.",
                current_run,
            )
            return 0

    manifest, run_time = build_risques(
        args.harmonie_base_url, args.days, Path(args.geojson), args.arome_base_url
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "risques.json"
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")

    LOGGER.info(
        "Publication prête : %s départements, run %s",
        len(manifest["departments"]),
        manifest["run_time"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
