import os
import pandas as pd

from router import route_question


EVAL_CSV_PATH = "data/eval_questions.csv"
RESULT_CSV_PATH = "data/eval_results.csv"


def classify_error(expected_route: str, predicted_route: str, error_message: str) -> tuple[str, str]:
    """
    評価用の誤分類ラベルを返す。

    Returns:
        error_type:
            - OK
            - WRONG_TO_SEARCH
            - DANGEROUS_MISROUTE
            - EXCEPTION
            - UNKNOWN
        severity:
            - NONE
            - SAFE
            - DANGEROUS
    """
    if error_message:
        return "EXCEPTION", "DANGEROUS"

    if expected_route == predicted_route:
        return "OK", "NONE"

    # 本来 NUMERIC / API のものを SEARCH に倒した
    if predicted_route == "SEARCH" and expected_route in {"NUMERIC", "API"}:
        return "WRONG_TO_SEARCH", "SAFE"

    # 本来 SEARCH に行くべきものを NUMERIC / API に送った
    if expected_route == "SEARCH" and predicted_route in {"NUMERIC", "API"}:
        return "DANGEROUS_MISROUTE", "DANGEROUS"

    # NUMERIC と API の取り違え
    if expected_route in {"NUMERIC", "API"} and predicted_route in {"NUMERIC", "API"}:
        return "DANGEROUS_MISROUTE", "DANGEROUS"

    return "UNKNOWN", "SAFE"


def evaluate_routes(csv_path: str = EVAL_CSV_PATH) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} が見つかりません。")

    df = pd.read_csv(csv_path)

    required_columns = {"question", "expected_route"}
    if not required_columns.issubset(df.columns):
        raise ValueError("評価CSVには 'question' と 'expected_route' 列が必要です。")

    results = []

    for _, row in df.iterrows():
        question = str(row["question"]).strip()
        expected_route = str(row["expected_route"]).strip().upper()

        category = ""
        if "category" in df.columns and pd.notna(row["category"]):
            category = str(row["category"]).strip()

        note = ""
        if "note" in df.columns and pd.notna(row["note"]):
            note = str(row["note"]).strip()

        try:
            decision = route_question(question)

            predicted_route = decision.route.strip().upper()
            is_correct = predicted_route == expected_route
            error_message = ""

            error_type, severity = classify_error(
                expected_route=expected_route,
                predicted_route=predicted_route,
                error_message=error_message,
            )

            results.append(
                {
                    "question": question,
                    "expected_route": expected_route,
                    "predicted_route": predicted_route,
                    "is_correct": is_correct,
                    "category": category,
                    "note": note,
                    "layer": decision.layer,
                    "reason": decision.reason,
                    "numeric_score": decision.numeric_score,
                    "api_score": decision.api_score,
                    "matched_rules": " | ".join(decision.matched_rules),
                    "inhibitors": " | ".join(decision.inhibitors),
                    "error_type": error_type,
                    "severity": severity,
                    "error_message": error_message,
                }
            )

        except Exception as e:
            predicted_route = "ERROR"
            is_correct = False
            error_message = str(e)

            error_type, severity = classify_error(
                expected_route=expected_route,
                predicted_route=predicted_route,
                error_message=error_message,
            )

            results.append(
                {
                    "question": question,
                    "expected_route": expected_route,
                    "predicted_route": predicted_route,
                    "is_correct": is_correct,
                    "category": category,
                    "note": note,
                    "layer": "",
                    "reason": "",
                    "numeric_score": "",
                    "api_score": "",
                    "matched_rules": "",
                    "inhibitors": "",
                    "error_type": error_type,
                    "severity": severity,
                    "error_message": error_message,
                }
            )

    return pd.DataFrame(results)


def print_summary(result_df: pd.DataFrame) -> None:
    total = len(result_df)
    correct = int(result_df["is_correct"].sum())
    accuracy = correct / total if total > 0 else 0

    dangerous_count = int((result_df["severity"] == "DANGEROUS").sum())
    safe_count = int((result_df["severity"] == "SAFE").sum())

    print("=== Evaluation Result ===")
    print(f"Total              : {total}")
    print(f"Correct            : {correct}")
    print(f"Accuracy           : {accuracy:.2%}")
    print(f"Dangerous errors   : {dangerous_count}")
    print(f"Safe-side errors   : {safe_count}")
    print()

    print("=== Expected Route別件数 ===")
    print(result_df["expected_route"].value_counts())
    print()

    print("=== Predicted Route別件数 ===")
    print(result_df["predicted_route"].value_counts())
    print()

    print("=== Expected × Predicted ===")
    print(pd.crosstab(result_df["expected_route"], result_df["predicted_route"]))
    print()

    if "category" in result_df.columns and result_df["category"].astype(str).str.len().sum() > 0:
        print("=== Category別正解率 ===")
        category_summary = (
            result_df.groupby("category")["is_correct"]
            .agg(["count", "sum", "mean"])
            .rename(columns={"count": "total", "sum": "correct", "mean": "accuracy"})
        )
        print(category_summary)
        print()

    print("=== 誤判定一覧 ===")
    wrong_df = result_df[result_df["is_correct"] == False]

    if wrong_df.empty:
        print("誤判定はありません。")
    else:
        print(
            wrong_df[
                [
                    "question",
                    "category",
                    "expected_route",
                    "predicted_route",
                    "error_type",
                    "severity",
                    "layer",
                    "reason",
                    "numeric_score",
                    "api_score",
                    "matched_rules",
                    "inhibitors",
                    "error_message",
                ]
            ].to_string(index=False)
        )


def main():
    result_df = evaluate_routes(EVAL_CSV_PATH)

    os.makedirs("data", exist_ok=True)
    result_df.to_csv(RESULT_CSV_PATH, index=False, encoding="utf-8-sig")

    print_summary(result_df)
    print()
    print(f"評価結果CSVを保存しました: {RESULT_CSV_PATH}")


if __name__ == "__main__":
    main()