import os
import re

import pandas as pd

SALES_CSV_PATH = "data/sales.csv"


def load_sales_data() -> pd.DataFrame:
    if not os.path.exists(SALES_CSV_PATH):
        raise FileNotFoundError(f"{SALES_CSV_PATH} が見つかりません。")

    df = pd.read_csv(SALES_CSV_PATH)

    required_columns = {"date", "product", "amount"}
    if not required_columns.issubset(df.columns):
        raise ValueError("sales.csv の列は date, product, amount が必要です。")

    df["product"] = df["product"].astype(str).str.strip()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])

    return df


def is_numeric_question(question: str) -> bool:
    keywords = ["売上", "合計", "平均", "件数", "何件", "いくら"]
    return any(keyword in question for keyword in keywords)


def normalize_product_name(text: str) -> str:
    """
    質問文側・CSV側の表記ゆれを吸収する。
    例:
    商品A -> A
    商品 A -> A
    a -> A
    """
    text = str(text).strip()
    text = re.sub(r"^商品\s*", "", text)
    return text.upper()


def extract_actions_from_question(question: str) -> list[str]:
    """
    質問文から求められている集計種別を複数抽出する。

    重要:
    - 「売上平均」は mean として扱う
    - 「売上合計」は sum として扱う
    - 「売上平均」が含まれるときに、単独の「売上」から sum を重複追加しない
    """
    actions = []

    normalized_question = re.sub(r"\s+", "", question)

    # 件数
    if any(k in normalized_question for k in ["件数", "何件"]):
        actions.append("count")

    # 平均系
    has_mean = any(k in normalized_question for k in ["売上平均", "平均"])
    if has_mean:
        actions.append("mean")

    # 合計系
    has_explicit_sum = any(k in normalized_question for k in ["売上合計", "合計", "いくら"])
    has_plain_sales = "売上" in normalized_question
    has_sales_mean = "売上平均" in normalized_question

    if has_explicit_sum:
        actions.append("sum")
    elif has_plain_sales and not has_sales_mean:
        actions.append("sum")

    return list(dict.fromkeys(actions))


def extract_products_from_question(question: str, valid_products: list[str]) -> list[str]:
    """
    商品リストとの照合を優先して、質問文に含まれる商品名を複数抽出する。
    """
    normalized_question = normalize_product_name(question)

    sorted_products = sorted(
        [str(p).strip() for p in valid_products if str(p).strip()],
        key=len,
        reverse=True
    )

    found_products = []
    for product in sorted_products:
        normalized_product = normalize_product_name(product)
        if normalized_product and normalized_product in normalized_question:
            found_products.append(product)

    unique_products = list(dict.fromkeys(found_products))
    return unique_products


def detect_action_in_text(text: str) -> str | None:
    """
    1つのかたまりの中から集計種別を1つ判定する。
    明確なペア抽出用なので、複数入っていたら曖昧として None を返す。

    重要:
    - 「売上平均」は mean
    - 「売上合計」は sum
    - 「売上平均」を含む場合、単独の「売上」で sum 扱いしない
    """
    cleaned = re.sub(r"\s+", "", text)
    found_actions = []

    if any(k in cleaned for k in ["件数", "何件"]):
        found_actions.append("count")

    if any(k in cleaned for k in ["売上平均", "平均"]):
        found_actions.append("mean")

    has_explicit_sum = any(k in cleaned for k in ["売上合計", "合計", "いくら"])
    has_plain_sales = "売上" in cleaned
    has_sales_mean = "売上平均" in cleaned

    if has_explicit_sum:
        found_actions.append("sum")
    elif has_plain_sales and not has_sales_mean:
        found_actions.append("sum")

    found_actions = list(dict.fromkeys(found_actions))

    if len(found_actions) == 1:
        return found_actions[0]

    return None


def split_question_into_clauses(question: str) -> list[str]:
    """
    明確なペア抽出のために、まず質問を並列ごとに分割する。
    例:
    商品Bの平均と商品Aの売上合計は？
    -> ["商品Bの平均", "商品Aの売上合計"]
    """
    cleaned = question.strip()
    cleaned = re.sub(r"[？?。]", "", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned)

    clauses = re.split(r"[、,]|と", cleaned)
    clauses = [c for c in clauses if c]

    return clauses


def extract_single_product_from_clause(clause: str, valid_products: list[str]) -> str | None:
    """
    1つの句の中から商品を1つだけ抽出する。
    2つ以上見つかる場合は曖昧なので None。
    """
    matched_products = []

    sorted_products = sorted(
        [str(p).strip() for p in valid_products if str(p).strip()],
        key=len,
        reverse=True
    )

    normalized_clause = normalize_product_name(clause)

    for product in sorted_products:
        normalized_product = normalize_product_name(product)
        if normalized_product and normalized_product in normalized_clause:
            matched_products.append(product)

    matched_products = list(dict.fromkeys(matched_products))

    if len(matched_products) == 1:
        return matched_products[0]

    return None


def extract_product_action_pairs(question: str, valid_products: list[str]) -> list[tuple[str, str]]:
    """
    明確な 商品×集計 のペアだけを抽出する。

    ルール:
    - まず「と」「、」で文を区切る
    - 各句の中に
        商品が1つ
        集計種別が1つ
      だけある場合に限ってペアとして採用する
    - 1つでも条件を満たさない句があれば、ペア抽出は不採用にして
      従来の全組み合わせ処理へフォールバックする
    """
    clauses = split_question_into_clauses(question)

    if len(clauses) <= 1:
        return []

    pairs = []

    for clause in clauses:
        product = extract_single_product_from_clause(clause, valid_products)
        action = detect_action_in_text(clause)

        if product is None or action is None:
            return []

        pairs.append((product, action))

    return pairs


def format_result_message(product: str | None, action: str, value: float | int) -> str:
    if product:
        if action == "count":
            return f"商品 {product} の件数は {int(value)} 件です。"
        if action == "mean":
            return f"商品 {product} の平均売上は {value:.0f} 円です。"
        if action == "sum":
            return f"商品 {product} の売上合計は {value:.0f} 円です。"
    else:
        if action == "count":
            return f"売上データの件数は {int(value)} 件です。"
        if action == "mean":
            return f"売上金額の平均は {value:.0f} 円です。"
        if action == "sum":
            return f"売上合計は {value:.0f} 円です。"

    return "計算結果を表示できませんでした。"


def format_evidence_message(product: str | None, action: str) -> str:
    if product:
        if action == "count":
            return f"sales.csv を product={product} で絞り込み、件数を数えました。"
        if action == "mean":
            return f"sales.csv を product={product} で絞り込み、amount の平均を計算しました。"
        if action == "sum":
            return f"sales.csv を product={product} で絞り込み、amount の合計を計算しました。"
    else:
        if action == "count":
            return "sales.csv の全行数を数えました。"
        if action == "mean":
            return "sales.csv の amount 列の平均を計算しました。"
        if action == "sum":
            return "sales.csv の amount 列の合計を計算しました。"

    return "計算根拠を生成できませんでした。"


def calculate_single_product(df: pd.DataFrame, product: str, action: str) -> tuple[str, str]:
    product_df = df[df["product"].astype(str).str.strip() == product]

    if product_df.empty:
        return (
            f"商品 {product} のデータは見つかりませんでした。",
            f"sales.csv を product={product} で検索しましたが一致がありませんでした。"
        )

    if action == "count":
        value = len(product_df)
    elif action == "mean":
        value = product_df["amount"].mean()
    elif action == "sum":
        value = product_df["amount"].sum()
    else:
        return (
            f"商品 {product} についての計算種別を特定できませんでした。",
            "集計種別が不明なため計算できませんでした。"
        )

    return (
        format_result_message(product, action, value),
        format_evidence_message(product, action)
    )


def answer_numeric_question(question: str) -> dict:
    df = load_sales_data()
    valid_products = df["product"].dropna().astype(str).str.strip().unique().tolist()

    results = []
    evidences = []

    # ① 明確な 商品×集計 のペアが取れた場合は、そのペアだけ返す
    pair_requests = extract_product_action_pairs(question, valid_products)
    if pair_requests:
        for product, action in pair_requests:
            answer_text, evidence_text = calculate_single_product(df, product, action)
            results.append(answer_text)
            evidences.append(evidence_text)

        return {
            "answer": "\n".join(results),
            "evidence": "\n".join(evidences),
        }

    # ② ペアが取れない場合は従来の全組み合わせ方式
    products = extract_products_from_question(question, valid_products)
    actions = extract_actions_from_question(question)

    if not actions:
        return {
            "answer": (
                "数値質問として判定しましたが、合計・平均・件数のどれを求めるか"
                "特定できませんでした。"
            ),
            "evidence": "集計種別が不足しているため計算を行いませんでした。"
        }

    # 商品指定あり
    if products:
        for product in products:
            product_df = df[df["product"].astype(str).str.strip() == product]

            if product_df.empty:
                results.append(f"商品 {product} のデータは見つかりませんでした。")
                evidences.append(
                    f"sales.csv を product={product} で検索しましたが一致がありませんでした。"
                )
                continue

            for action in actions:
                if action == "count":
                    value = len(product_df)
                elif action == "mean":
                    value = product_df["amount"].mean()
                elif action == "sum":
                    value = product_df["amount"].sum()
                else:
                    continue

                results.append(format_result_message(product, action, value))
                evidences.append(format_evidence_message(product, action))

        # 複数商品の合算集計（合計）
        if len(products) > 1 and "sum" in actions:
            combined_df = df[df["product"].astype(str).str.strip().isin(products)]

            if not combined_df.empty:
                combined_total = combined_df["amount"].sum()
                product_list_str = "と".join(products)

                results.append(
                    f"商品 {product_list_str} を合わせた売上合計は {combined_total:.0f} 円です。"
                )
                evidences.append(
                    f"sales.csv を product in {products} で絞り込み、amount の合計を計算しました。"
                )

        # 複数商品のまとめ平均（加重平均）
        if len(products) > 1 and "mean" in actions:
            combined_df = df[df["product"].astype(str).str.strip().isin(products)]

            if not combined_df.empty:
                combined_mean = combined_df["amount"].mean()
                product_list_str = "と".join(products)

                results.append(
                    f"商品 {product_list_str} を合わせた平均売上は {combined_mean:.0f} 円です。"
                )
                evidences.append(
                    f"sales.csv を product in {products} でまとめて平均を計算しました。"
                )

    # 商品指定なし
    else:
        for action in actions:
            if action == "count":
                value = len(df)
            elif action == "mean":
                value = df["amount"].mean()
            elif action == "sum":
                value = df["amount"].sum()
            else:
                continue

            results.append(format_result_message(None, action, value))
            evidences.append(format_evidence_message(None, action))

    if not results:
        return {
            "answer": (
                "数値質問として判定しましたが、対象や集計内容が特定できませんでした。"
                " 合計・平均・件数、または sales.csv に存在する商品名を指定してください。"
            ),
            "evidence": "要素不足のため計算を行いませんでした。"
        }

    return {
        "answer": "\n".join(results),
        "evidence": "\n".join(evidences),
    }