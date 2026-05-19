import re
import unicodedata


SYNONYMS = {
    "締め切り": "締切",
    "しめきり": "締切",
    "問い合わせ": "問合せ",
    "問い合せ": "問合せ",
    "申し込み": "申込",
    "申しこみ": "申込",
    "申込み": "申込",
    "いつ迄": "いつまで",
    "期限": "いつまで",
}


REMOVABLE_PHRASES = [
    "について教えてください",
    "について教えて",
    "を教えてください",
    "を教えて",
    "してください",
    "下さい",
    "ですか",
    "ますか",
    "でしょうか",
    "知りたいです",
    "知りたい",
    "すればいいですか",
    "したらいいですか",
    "どこに連絡すればいい",
    "どこに連絡すればいいですか",
]


WEAK_WORDS = {
    "の", "に", "は", "を", "が", "で", "と", "も", "へ", "や", "か",
    "する", "したい", "ある", "いる", "こと", "もの",
}


def normalize_text(text: str) -> str:
    """
    FAQ照合のための簡易正規化
    - 全角/半角のゆれを吸収
    - 小文字化
    - 表記ゆれ辞書を適用
    - 不要な定型句を削る
    - 記号を空白に置換
    - 連続空白を1つにする
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text).lower()

    for src, dst in SYNONYMS.items():
        text = text.replace(src, dst)

    for phrase in REMOVABLE_PHRASES:
        text = text.replace(phrase, "")

    text = re.sub(r"[？?！!。、,，．.・/／()\[\]「」『』:：;；\-ー~〜]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_simple(text: str) -> list[str]:
    """
    FAQ用の簡易キーワード化
    日本語を厳密に形態素解析するのではなく、
    正規化後の文字列を空白で区切り、弱い語を除外する
    """
    normalized = normalize_text(text)

    if not normalized:
        return []

    tokens = normalized.split()

    return [token for token in tokens if len(token) > 1 and token not in WEAK_WORDS]


def build_normalized_key(text: str) -> str:
    """
    FAQ登録時に保存する比較用キー
    日本語は空白区切りが弱いので、正規化後文字列そのものを保存する
    """
    return normalize_text(text).replace(" ", "")


def calc_keyword_overlap_score(user_key: str, faq_key: str) -> float:
    """
    日本語向けに文字2-gramの重なり率を計算する
    """
    if not user_key or not faq_key:
        return 0.0

    def make_ngrams(text: str, n: int = 2) -> set[str]:
        if len(text) < n:
            return {text} if text else set()
        return {text[i:i+n] for i in range(len(text) - n + 1)}

    user_ngrams = make_ngrams(user_key, n=2)
    faq_ngrams = make_ngrams(faq_key, n=2)

    if not user_ngrams or not faq_ngrams:
        return 0.0

    overlap = user_ngrams & faq_ngrams
    return len(overlap) / len(faq_ngrams)