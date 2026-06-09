import os
import sqlite3
import time

import chromadb
import requests
import streamlit as st

from faq_utils import build_normalized_key, calc_keyword_overlap_score, normalize_text
from llm import build_extract_answer, embed_query, generate_rag_answer
from log_db import init_log_db, save_log
from numeric_qa import answer_numeric_question
from router import route_question
from weather_api import answer_weather_question

DB_NAME = "faq.db"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "manual_chunks"

# OpenAI Embedding移行後の暫定しきい値。
# Gemini時代の 0.28 ではOpenAI Embeddingの距離スケールに合わないため再調整。
RAG_DISTANCE_THRESHOLD = 1.45
RAG_GAP_THRESHOLD = 0.02

# 開発用の内部情報を画面に出すかどうか。
# GitHub提出時や面接デモ時は False 推奨。
DEBUG_MODE = False

# FAQ採用の基本しきい値
FAQ_SCORE_THRESHOLD = 0.45
FAQ_MARGIN_THRESHOLD = 0.12

DOMAIN_MAP = {
    "expense": ["経費", "経費精算", "交通費"],
    "attendance": ["在宅勤務", "勤怠", "遅刻", "有給", "休暇"],
    "it": ["パスワード", "PC", "故障", "情報システム", "システム部"],
    "general_apply": ["出張", "備品", "名刺", "購買"],
}

IMPORTANT_RAG_KEYWORDS = [
    "有給",
    "有給休暇",
    "交通費",
    "経費",
    "経費精算",
    "備品",
    "名刺",
    "パスワード",
    "在宅勤務",
    "勤怠",
    "遅刻",
    "出張",
    "購買",
]


def detect_domain(text: str) -> str:
    if not text:
        return "other"

    for domain, keywords in DOMAIN_MAP.items():
        for keyword in keywords:
            if keyword in text:
                return domain

    return "other"


def has_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def score_faq_candidate(
    normalized_user: str,
    user_key: str,
    user_domain: str,
    faq_question: str,
    faq_key: str,
    faq_domain: str,
) -> float:
    normalized_faq = normalize_text(faq_question)
    base_score = calc_keyword_overlap_score(user_key, faq_key)
    score = base_score

    important_keywords = [
        "経費精算",
        "交通費",
        "在宅勤務",
        "勤怠",
        "有給休暇",
        "遅刻",
        "パスワード",
        "PC",
        "出張",
        "備品",
        "名刺",
    ]

    user_keywords = [kw for kw in important_keywords if kw in normalized_user]
    faq_keywords = [kw for kw in important_keywords if kw in normalized_faq]

    for kw in user_keywords:
        if kw in faq_keywords:
            score += 0.5

    for kw in faq_keywords:
        if kw not in user_keywords:
            score -= 0.35

    if normalized_user and normalized_faq:
        if normalized_user in normalized_faq or normalized_faq in normalized_user:
            score += 0.2

    if ("いつまで" in normalized_user and "締切" in normalized_faq) or (
        "締切" in normalized_user and "いつまで" in normalized_faq
    ):
        score += 0.3

    if ("どうすれば" in normalized_user and "方法" in normalized_faq) or (
        "方法" in normalized_user and "どうすれば" in normalized_faq
    ):
        score += 0.2

    user_is_fix_howto = has_any(
        normalized_user,
        ["修正方法", "修正の方法", "方法を教えて", "やり方を教えて"],
    )
    faq_is_fix_howto = has_any(normalized_faq, ["どうすれば", "申請", "修正"])

    if user_is_fix_howto and faq_is_fix_howto:
        score += 0.2

    if has_any(normalized_user, ["申請", "いつまで"]) and has_any(
        normalized_faq, ["申請", "いつまで"]
    ):
        score += 0.1

    if user_domain != "other" and faq_domain != "other":
        if user_domain == faq_domain:
            score += 0.3
        else:
            score -= 0.5

    if normalized_user and normalized_faq:
        length_ratio = min(len(normalized_user), len(normalized_faq)) / max(
            len(normalized_user), len(normalized_faq)
        )
        score *= length_ratio

    return score


def search_faq(user_question: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT question, answer, normalized_key, domain FROM faq")
    rows = cursor.fetchall()
    conn.close()

    user_question = user_question.strip()
    normalized_user = normalize_text(user_question)
    user_key = build_normalized_key(user_question)
    user_domain = detect_domain(normalized_user)

    for faq_question, faq_answer, faq_key, faq_domain in rows:
        if user_question == faq_question:
            return faq_question, faq_answer

    for faq_question, faq_answer, faq_key, faq_domain in rows:
        if normalized_user == normalize_text(faq_question):
            return faq_question, faq_answer

    if len(normalized_user) <= 3:
        return None

    same_domain_rows = []
    if user_domain != "other":
        same_domain_rows = [row for row in rows if row[3] == user_domain]

    candidate_rows = same_domain_rows if same_domain_rows else rows

    best_match = None
    best_score = float("-inf")
    second_score = float("-inf")

    for faq_question, faq_answer, faq_key, faq_domain in candidate_rows:
        score = score_faq_candidate(
            normalized_user=normalized_user,
            user_key=user_key,
            user_domain=user_domain,
            faq_question=faq_question,
            faq_key=faq_key,
            faq_domain=faq_domain,
        )

        if score > best_score:
            second_score = best_score
            best_score = score
            best_match = (faq_question, faq_answer)
        elif score > second_score:
            second_score = score

    if best_score == float("-inf"):
        best_score = None
    if second_score == float("-inf"):
        second_score = None

    margin = None
    if best_score is not None and second_score is not None:
        margin = best_score - second_score

    if best_match and best_score is not None:
        if best_score >= FAQ_SCORE_THRESHOLD:
            if second_score is None or margin is None or margin >= FAQ_MARGIN_THRESHOLD:
                return best_match

    return None


def search_rag(user_question: str, n_results: int = 5):
    query_embedding = embed_query(user_question)

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            "RAGコレクションが見つかりません。manual.txt 更新後は python build_rag.py を実行してください。"
        ) from e

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    matched_keywords = [
        keyword for keyword in IMPORTANT_RAG_KEYWORDS if keyword in user_question
    ]

    keyword_matched = False

    if matched_keywords:
        filtered = [
            (doc, metadata, distance)
            for doc, metadata, distance in zip(documents, metadatas, distances)
            if any(keyword in doc for keyword in matched_keywords)
        ]

        if filtered:
            keyword_matched = True
            documents = [item[0] for item in filtered]
            metadatas = [item[1] for item in filtered]
            distances = [item[2] for item in filtered]

    best_distance = distances[0] if len(distances) >= 1 else None
    second_distance = distances[1] if len(distances) >= 2 else None

    gap = None
    if best_distance is not None and second_distance is not None:
        gap = second_distance - best_distance

    rag_debug = {
        "best_distance": best_distance,
        "second_distance": second_distance,
        "gap": gap,
        "threshold": RAG_DISTANCE_THRESHOLD,
        "gap_threshold": RAG_GAP_THRESHOLD,
        "keyword_matched": keyword_matched,
        "matched_keywords": matched_keywords,
        "judge": "",
        "reason": "",
    }

    if not documents:
        rag_debug["judge"] = "rejected"
        rag_debug["reason"] = "no_documents"
        return [], [], [], rag_debug

    if best_distance is None:
        rag_debug["judge"] = "rejected"
        rag_debug["reason"] = "no_distance"
        return [], [], [], rag_debug

    if keyword_matched:
        rag_debug["judge"] = "accepted"
        rag_debug["reason"] = "keyword_match_priority"
        return documents, metadatas, distances, rag_debug

    if best_distance > RAG_DISTANCE_THRESHOLD:
        rag_debug["judge"] = "rejected"
        rag_debug["reason"] = "distance_threshold_exceeded"
        return [], [], [], rag_debug

    if gap is not None and gap < RAG_GAP_THRESHOLD:
        rag_debug["judge"] = "rejected"
        rag_debug["reason"] = "ambiguous_gap"
        return [], [], [], rag_debug

    rag_debug["judge"] = "accepted"
    rag_debug["reason"] = "within_distance_threshold"
    return documents, metadatas, distances, rag_debug


def should_skip_faq(question: str) -> bool:
    rag_preferred_keywords = [
        "手順",
        "詳しい流れ",
        "とは",
        "どう進める",
    ]
    return any(keyword in question for keyword in rag_preferred_keywords)


def format_rag_debug_info(rag_debug: dict) -> str:
    best = rag_debug.get("best_distance")
    second = rag_debug.get("second_distance")
    gap = rag_debug.get("gap")
    threshold = rag_debug.get("threshold")
    gap_threshold = rag_debug.get("gap_threshold")
    keyword_matched = rag_debug.get("keyword_matched")
    matched_keywords = rag_debug.get("matched_keywords", [])
    judge = rag_debug.get("judge", "")
    reason = rag_debug.get("reason", "")

    best_text = f"{best:.4f}" if best is not None else "None"
    second_text = f"{second:.4f}" if second is not None else "None"
    gap_text = f"{gap:.4f}" if gap is not None else "None"
    threshold_text = f"{threshold:.4f}" if threshold is not None else "None"
    gap_threshold_text = f"{gap_threshold:.4f}" if gap_threshold is not None else "None"
    keywords_text = ",".join(matched_keywords) if matched_keywords else "None"

    return (
        f"RAG judge={judge}, reason={reason}, "
        f"best_distance={best_text}, second_distance={second_text}, "
        f"gap={gap_text}, threshold={threshold_text}, "
        f"gap_threshold={gap_threshold_text}, "
        f"keyword_matched={keyword_matched}, matched_keywords={keywords_text}"
    )


st.set_page_config(
    page_title="業務問い合わせAIアシスタント",
    layout="centered",
)

init_log_db()

st.title("業務問い合わせAIアシスタント")
st.write("FAQ検索・RAG・数値回答・天気APIを試すPoCです。")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY が設定されていません。先に環境変数を設定してください。")
    st.stop()

question = st.text_input(
    "質問を入力してください",
    placeholder="例：在宅勤務の申請はいつまでですか？",
)

if st.button("送信"):
    if question.strip() == "":
        st.warning("質問を入力してください。")
    else:
        cleaned_question = question.strip()
        start_time = time.perf_counter()

        route_decision = route_question(cleaned_question)
        route = route_decision.route

        final_route = route
        log_answer = ""
        log_status = "SUCCESS"
        log_error_message = ""
        rag_debug = None

        st.subheader("回答")

        if DEBUG_MODE:
            st.caption(f"route={route} / layer={route_decision.layer}")
            st.caption(f"router reason: {route_decision.reason}")

            if route_decision.matched_rules:
                st.caption(f"matched: {', '.join(route_decision.matched_rules)}")

            if route_decision.inhibitors:
                st.caption(f"inhibitors: {', '.join(route_decision.inhibitors)}")

            st.caption(
                f"numeric_score={route_decision.numeric_score} / "
                f"api_score={route_decision.api_score}"
            )

        try:
            if route == "NUMERIC":
                result = answer_numeric_question(cleaned_question)

                if not result.get("answer") or not result.get("evidence"):
                    raise ValueError("数値回答の根拠または回答が不足しています。")

                final_route = "NUMERIC"
                log_answer = result["answer"]

                st.success("数値ルートで回答しました。")
                st.write(f"**回答:** {result['answer']}")

                st.markdown("### 根拠")
                st.write(result["evidence"])

            elif route == "API":
                try:
                    result = answer_weather_question(cleaned_question)
                except ValueError:
                    final_route = "API"
                    log_status = "NO_ANSWER"
                    log_answer = "場所を特定できなかったため、天気を取得できませんでした。"

                    st.warning("場所を特定できなかったため、天気を取得できませんでした。")
                    st.info("地名を変えて質問してください。例）北海道→千歳　愛知→名古屋")
                else:
                    if not result.get("answer") or not result.get("evidence"):
                        raise ValueError("天気APIの根拠または回答が不足しています。")

                    final_route = "API"
                    log_answer = result["answer"]

                    st.success("APIルートで回答しました。")
                    st.write(f"**回答:** {result['answer']}")

                    st.markdown("### 根拠")
                    st.write(result["evidence"])
                    st.info("根拠: Open-Meteo API")

            else:
                faq_result = None

                if not should_skip_faq(cleaned_question):
                    faq_result = search_faq(cleaned_question)

                if faq_result:
                    faq_question, faq_answer = faq_result

                    if not faq_answer:
                        raise ValueError("FAQ回答が空でした。")

                    final_route = "FAQ"
                    log_answer = faq_answer

                    st.success("FAQルートで回答しました。")
                    st.write(f"**一致したFAQ:** {faq_question}")
                    st.write(f"**回答:** {faq_answer}")
                    st.info("根拠: SQLite の faq テーブル")

                else:
                    docs, metadatas, distances, rag_debug = search_rag(
                        cleaned_question,
                        n_results=5,
                    )

                    if DEBUG_MODE:
                        st.markdown("### 取得されたRAG文書")
                        for i, doc in enumerate(docs, start=1):
                            st.write(f"**取得文書{i}:**")
                            st.write(doc)

                    if DEBUG_MODE and rag_debug:
                        st.markdown("### RAG観察情報")
                        st.caption(
                            f"judge={rag_debug['judge']} / reason={rag_debug['reason']}"
                        )
                        st.caption(
                            f"best_distance={rag_debug['best_distance']:.4f}"
                            if rag_debug["best_distance"] is not None
                            else "best_distance=None"
                        )
                        st.caption(
                            f"second_distance={rag_debug['second_distance']:.4f}"
                            if rag_debug["second_distance"] is not None
                            else "second_distance=None"
                        )
                        st.caption(
                            f"gap={rag_debug['gap']:.4f}"
                            if rag_debug["gap"] is not None
                            else "gap=None"
                        )
                        st.caption(f"threshold={rag_debug['threshold']:.4f}")
                        st.caption(
                            f"keyword_matched={rag_debug['keyword_matched']} / "
                            f"matched_keywords={rag_debug['matched_keywords']}"
                        )

                    if not docs:
                        final_route = "RAG"
                        log_answer = "文書内に該当情報がありません。"
                        log_status = "NO_ANSWER"

                        st.warning("文書内に該当情報がありません。")
                        st.info(
                            "根拠となる文書が見つからなかった、または検索条件を満たさなかったため、回答を控えました。"
                        )

                    else:
                        used_fallback = False

                        try:
                            answer = generate_rag_answer(cleaned_question, docs)
                        except Exception as e:
                            error_text = str(e)
                            if "429" in error_text or "rate_limit" in error_text.lower():
                                answer = build_extract_answer(docs)
                                used_fallback = True
                            else:
                                raise

                        if not answer or not answer.strip():
                            raise ValueError("RAG回答が空でした。")

                        final_route = "RAG"
                        log_answer = answer

                        no_answer_phrases = [
                            "文書内に該当情報がありません",
                            "該当する情報が見つかりません",
                            "該当する記載は見つかりません",
                            "根拠となる情報が見つかりません",
                        ]

                        if any(phrase in answer for phrase in no_answer_phrases):
                            log_status = "NO_ANSWER"
                            st.warning("文書内に該当情報がありません。")
                            st.info("根拠となる文書が十分でないため、回答を控えました。")
                        else:
                            st.success("RAGルートで回答しました。")

                            if used_fallback:
                                st.warning(
                                    "生成APIでエラーが発生したため、要約ではなく関連文書をそのまま表示しています。"
                                )
                                st.markdown("### 関連文書")
                                st.write(answer)
                            else:
                                st.write(f"**回答:** {answer}")

                            st.markdown("### 根拠")
                            for i, doc in enumerate(docs, start=1):
                                st.write(f"**根拠{i}:**")
                                st.write(doc)

                                if DEBUG_MODE:
                                    if i - 1 < len(metadatas) and metadatas[i - 1]:
                                        source = metadatas[i - 1].get("source", "")
                                        chunk_index = metadatas[i - 1].get("chunk_index", "")
                                        st.caption(
                                            f"source={source}, chunk_index={chunk_index}"
                                        )

                                    if i - 1 < len(distances):
                                        st.caption(f"distance={distances[i - 1]:.4f}")

                        st.info("根拠: ChromaDB の類似検索結果")

        except requests.RequestException:
            final_route = route
            log_status = "ERROR"
            log_error_message = "外部APIとの通信に失敗しました。"
            log_answer = "現在、外部APIとの通信に失敗したため回答できませんでした。"

            st.error("外部APIとの通信に失敗しました。時間をおいて再試行してください。")

        except Exception as e:
            final_route = route if route != "SEARCH" else final_route
            log_status = "ERROR"
            log_error_message = str(e)
            log_answer = "処理中に問題が発生したため回答できませんでした。"

            st.error("処理中に問題が発生しました。")

            if DEBUG_MODE:
                st.exception(e)

        finally:
            processing_time = round(time.perf_counter() - start_time, 4)

            searched_location = ""
            if final_route == "API" and "result" in locals():
                searched_location = result.get("searched_name", "")

            if rag_debug is not None:
                rag_debug_text = format_rag_debug_info(rag_debug)
                if log_error_message:
                    log_error_message = f"{log_error_message} | {rag_debug_text}"
                else:
                    log_error_message = rag_debug_text

            save_log(
                question=cleaned_question,
                route=final_route,
                answer=log_answer,
                processing_time=processing_time,
                status=log_status,
                error_message=log_error_message,
                router_layer=route_decision.layer,
                router_reason=route_decision.reason,
                router_scores=(
                    f"num={route_decision.numeric_score}, "
                    f"api={route_decision.api_score}"
                ),
                searched_location=searched_location,
            )

            if DEBUG_MODE:
                st.caption(
                    f"最終route: {final_route} / status: {log_status} / "
                    f"処理時間: {processing_time} 秒"
                )