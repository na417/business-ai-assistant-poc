import re
from dataclasses import dataclass, field


@dataclass
class RouteDecision:
    route: str
    layer: str
    reason: str
    numeric_score: int = 0
    api_score: int = 0
    matched_rules: list[str] = field(default_factory=list)
    inhibitors: list[str] = field(default_factory=list)


# ----------------------
# キーワード定義
# ----------------------

NUMERIC_KEYWORDS = ["売上", "合計", "平均", "件数", "何件", "いくら"]
API_KEYWORDS = ["天気", "天候", "気温", "雨", "晴れ", "曇り", "雪", "風"]
SEARCH_KEYWORDS = ["方法", "手順", "流れ", "ルール", "条件", "詳しく", "どうやって", "意味", "見方"]


# ----------------------
# REGEX
# ----------------------

NUMERIC_REGEX_RULES = [
    (r".+?の(売上|合計|平均|件数|何件|いくら)", "〇〇の売上系"),
]

API_REGEX_RULES = [
    (r".+?の(天気|天候|気温)", "〇〇の天気系"),
]


# ----------------------
# スコア定義
# ----------------------

NUMERIC_SCORE = {
    "売上": 3,
    "合計": 3,
    "平均": 3,
    "件数": 3,
    "何件": 3,
    "いくら": 3,
}

API_SCORE = {
    "天気": 4,
    "天候": 4,
    "気温": 4,
    "雨": 2,
    "晴れ": 2,
    "曇り": 2,
    "雪": 2,
    "風": 2,
}

INHIBITORS = {
    "方法": 4,
    "手順": 4,
    "ルール": 4,
    "意味": 3,
    "見方": 3,
    "どうやって": 4,
    "とは": 3,
}


# ----------------------
# シグナル抽出
# ----------------------

def detect_signals(question: str):
    numeric_hit = any(k in question for k in NUMERIC_KEYWORDS)
    api_hit = any(k in question for k in API_KEYWORDS)
    search_hit = any(k in question for k in SEARCH_KEYWORDS)

    return numeric_hit, api_hit, search_hit


# ----------------------
# スコア計算
# ----------------------

def calculate_scores(question: str):
    numeric_score = 0
    api_score = 0
    matched = []
    inhibitors = []

    for k, v in NUMERIC_SCORE.items():
        if k in question:
            numeric_score += v
            matched.append(f"NUMERIC:{k}(+{v})")

    for k, v in API_SCORE.items():
        if k in question:
            api_score += v
            matched.append(f"API:{k}(+{v})")

    for k, v in INHIBITORS.items():
        if k in question:
            numeric_score -= v
            api_score -= v
            inhibitors.append(f"{k}(-{v})")

    return numeric_score, api_score, matched, inhibitors


# ----------------------
# メイン
# ----------------------

def route_question(question: str) -> RouteDecision:
    q = question.strip()

    numeric_hit, api_hit, search_hit = detect_signals(q)

    # =========================
    # ① REGEX候補チェック
    # =========================

    regex_candidate = None

    for pattern, reason in NUMERIC_REGEX_RULES:
        if re.search(pattern, q):
            regex_candidate = ("NUMERIC", reason)

    for pattern, reason in API_REGEX_RULES:
        if re.search(pattern, q):
            # すでにNUMERIC候補がある場合は衝突扱い
            if regex_candidate:
                regex_candidate = ("CONFLICT", "NUMERICとAPI両方にヒット")
            else:
                regex_candidate = ("API", reason)

    # =========================
    # ② 衝突判定
    # =========================

    signal_count = sum([numeric_hit, api_hit, search_hit])

    conflict = signal_count >= 2

    # =========================
    # ③ 即決 or スコア
    # =========================

    # --- 即決できるケース ---
    if regex_candidate and not conflict:
        route, reason = regex_candidate
        return RouteDecision(
            route=route,
            layer="REGEX",
            reason=f"{reason}のため {route}",
        )

    # =========================
    # ④ スコア判定
    # =========================

    numeric_score, api_score, matched, inhibitors = calculate_scores(q)

    # 他カテゴリの混入や説明系抑制がない、純粋な短文を救済
    is_pure_numeric = numeric_score >= 3 and api_score <= 0 and not inhibitors
    is_pure_api = api_score >= 2 and numeric_score <= 0 and not inhibitors

    score_gap = abs(numeric_score - api_score)

    # -------------------------
    # ④-1 純粋な短文を先に救済
    # -------------------------

    if is_pure_numeric:
       return RouteDecision(
             route="NUMERIC",
             layer="SCORE",
             reason="純粋なNUMERICキーワードを検知したためNUMERIC",
             numeric_score=numeric_score,
             api_score=api_score,
             matched_rules=matched,
             inhibitors=inhibitors,
        )

    if is_pure_api:
        return RouteDecision(
             route="API",
             layer="SCORE",
             reason="純粋なAPIキーワードを検知したためAPI",
             numeric_score=numeric_score,
             api_score=api_score,
             matched_rules=matched,
             inhibitors=inhibitors,
        )

    # -------------------------
    # ④-2 高スコア優位なら採用
    # -------------------------

    if numeric_score >= 6 and numeric_score > api_score and score_gap >= 2:
         return RouteDecision(
             route="NUMERIC",
             layer="SCORE",
             reason="スコア優位でNUMERIC",
             numeric_score=numeric_score,
             api_score=api_score,
             matched_rules=matched,
             inhibitors=inhibitors,
        )

    if api_score >= 6 and api_score > numeric_score and score_gap >= 2:
         return RouteDecision(
             route="API",
             layer="SCORE",
             reason="スコア優位でAPI",
             numeric_score=numeric_score,
             api_score=api_score,
             matched_rules=matched,
             inhibitors=inhibitors,
        )  

    # SEARCH fallback
    return RouteDecision(
         route="SEARCH",
         layer="SCORE",
         reason="衝突または根拠不足のためSEARCH",
         numeric_score=numeric_score,
         api_score=api_score,
         matched_rules=matched,
         inhibitors=inhibitors,
    )