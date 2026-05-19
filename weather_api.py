import logging
import re

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def extract_location(question: str) -> str:
    question = question.strip()
    question = question.replace("？", "?").replace("　", " ")

    patterns = [
        r"(.+?)の今日の天気",
        r"今日の(.+?)の天気",
        r"(.+?)の天気",
        r"(.+?)の気温",
        r"(.+?)ってどんな天気",
        r"(.+?)の天気教えて",
    ]

    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            location = match.group(1).strip()
            if location:
                return location

    simple_match = re.fullmatch(r"(.+?)(?:の?(?:天気|気温))?\??", question)
    if simple_match:
        location = simple_match.group(1).strip()
        if location:
            return location

    raise ValueError("質問文から地名を抽出できませんでした。")


def strip_admin_suffix(location_name: str) -> str:
    suffixes = ["都", "道", "府", "県", "市"]
    for suffix in suffixes:
        if location_name.endswith(suffix) and len(location_name) > len(suffix):
            return location_name[:-len(suffix)]
    return location_name


def build_prefecture_candidate(base_name: str) -> str:
    special_map = {
        "東京": "東京都",
        "大阪": "大阪府",
        "京都": "京都府",
    }
    return special_map.get(base_name, base_name + "県")


def build_city_candidate(base_name: str) -> str:
    return base_name + "市"


def build_location_candidates(location_name: str) -> list[str]:
    candidates = []

    # ① 元の入力
    candidates.append(location_name)

    # ② 行政区分を外した素の地名
    base_name = strip_admin_suffix(location_name)
    candidates.append(base_name)

    # ③ 素の地名から都道府県候補を作る
    candidates.append(build_prefecture_candidate(base_name))

    # ④ 素の地名から市候補を作る
    candidates.append(build_city_candidate(base_name))

    # 空文字除去 + 重複除去
    candidates = [c for c in candidates if c]
    candidates = list(dict.fromkeys(candidates))

    return candidates


def _search_geocode_once(name: str) -> dict | None:
    response = requests.get(
        GEOCODING_URL,
        params={
            "name": name,
            "count": 1,
            "language": "ja",
            "format": "json",
            "countryCode": "JP",
        },
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()
    results = data.get("results", [])

    if not results:
        return None

    place = results[0]
    return {
        "name": place.get("name", name),
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "country": place.get("country", ""),
        "admin1": place.get("admin1", ""),
    }


def geocode_location(location_name: str) -> tuple[dict, str]:
    candidates = build_location_candidates(location_name)

    logger.info(f"[地名検索] 入力: {location_name}")
    logger.info(f"[地名検索] 候補一覧: {candidates}")

    for name in candidates:
        try:
            logger.info(f"[地名検索] 検索中: {name}")
            place = _search_geocode_once(name)

            if place:
                logger.info(
                    f"[地名検索] ヒット: {name} -> {place['name']} "
                    f"(lat={place['latitude']}, lon={place['longitude']})"
                )
                return place, name

        except requests.RequestException as e:
            logger.error(f"[地名検索] APIエラー: {e}")
            raise

    logger.warning(f"[地名検索] ヒットなし: {location_name}")
    raise ValueError(f"地名 '{location_name}' が見つかりませんでした。")


def fetch_weather(latitude: float, longitude: float) -> dict:
    response = requests.get(
        FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min",
            "timezone": "Asia/Tokyo",
            "forecast_days": 1,
        },
        timeout=10,
    )
    response.raise_for_status()

    return response.json()


def weather_code_to_japanese(weather_code: int) -> str:
    code_map = {
        0: "快晴",
        1: "晴れ",
        2: "一部くもり",
        3: "くもり",
        45: "霧",
        48: "霧",
        51: "弱い霧雨",
        53: "霧雨",
        55: "強い霧雨",
        61: "弱い雨",
        63: "雨",
        65: "強い雨",
        71: "弱い雪",
        73: "雪",
        75: "強い雪",
        80: "にわか雨",
        81: "強いにわか雨",
        82: "激しいにわか雨",
        95: "雷雨",
    }
    return code_map.get(weather_code, f"天気コード {weather_code}")


def answer_weather_question(question: str) -> dict:
    location_name = extract_location(question)
    place, searched_name = geocode_location(location_name)
    weather_data = fetch_weather(place["latitude"], place["longitude"])

    current = weather_data.get("current", {})
    daily = weather_data.get("daily", {})

    current_temp = current.get("temperature_2m")
    current_code = current.get("weather_code")
    wind_speed = current.get("wind_speed_10m")

    max_temp_list = daily.get("temperature_2m_max", [])
    min_temp_list = daily.get("temperature_2m_min", [])
    daily_code_list = daily.get("weather_code", [])

    max_temp = max_temp_list[0] if max_temp_list else None
    min_temp = min_temp_list[0] if min_temp_list else None
    daily_code = daily_code_list[0] if daily_code_list else current_code

    answer = (
        f"{place['name']}の現在の天気は {weather_code_to_japanese(current_code)}、"
        f"気温は {current_temp}℃ です。"
    )

    if max_temp is not None and min_temp is not None:
        answer += (
            f" 今日の予報は {weather_code_to_japanese(daily_code)}、"
            f"最高気温 {max_temp}℃、最低気温 {min_temp}℃ です。"
        )

    location_label_parts = [place["name"]]
    if place.get("admin1"):
        location_label_parts.append(place["admin1"])
    if place.get("country"):
        location_label_parts.append(place["country"])
    location_label = " / ".join(location_label_parts)

    evidence = (
        f"Open-Meteo API から取得しました。"
        f" 地点: {location_label}, "
        f"検索語: {searched_name}, "
        f"緯度: {place['latitude']}, 経度: {place['longitude']}, "
        f"現在気温: {current_temp}℃, "
        f"風速: {wind_speed} km/h"
    )

    return {
        "answer": answer,
        "evidence": evidence,
        "requested_location": location_name,
        "resolved_location": place["name"],
        "searched_name": searched_name,
    }