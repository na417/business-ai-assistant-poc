import sqlite3

from faq_utils import build_normalized_key

DB_NAME = "faq.db"

DOMAIN_MAP = {
    "expense": ["経費", "経費精算", "交通費"],
    "attendance": ["在宅勤務", "勤怠", "遅刻", "有給", "休暇"],
    "it": ["パスワード", "PC", "故障", "情報システム", "システム部"],
    "general_apply": ["出張", "備品", "名刺", "購買"],
}

sample_faqs = [
    # 締切系
    ("経費精算はいつまでに申請すればいいですか？", "経費精算は毎月末日までにシステムから申請してください。"),
    ("在宅勤務はいつまでに申請すればいいですか？", "在宅勤務は前日までに申請し、上長の承認を取得してください。"),
    ("交通費はいつまでに申請すればいいですか？", "交通費は利用月の翌月5日までに申請してください。"),
    ("有給休暇はいつまでに申請すればいいですか？", "有給休暇は原則として取得日の3日前までに申請してください。"),

    # 一問一答の対応系
    ("勤怠の修正はどうすればいいですか？", "勤怠システムの修正申請メニューから申請し、上長承認後に反映されます。"),
    ("遅刻する場合はどうすればいいですか？", "始業前に上長へ連絡し、勤怠システムで遅刻申請を行ってください。"),
    ("パスワードを忘れたらどうすればいいですか？", "情報システム部へ連絡し、本人確認後に再発行を受けてください。"),
    ("会社のPCが故障した場合はどうすればいいですか？", "情報システム部へ連絡し、指示に従って対応してください。"),

    # 入口案内系
    ("出張申請はどこから行いますか？", "社内ポータルの申請メニューから出張申請を行ってください。"),
    ("備品や名刺はどこから申請できますか？", "購買申請フォームまたは総務部の備品申請フォームから申請してください。"),
]


def assign_domain(text: str) -> str:
    if not text:
        return "other"

    for domain, keywords in DOMAIN_MAP.items():
        for keyword in keywords:
            if keyword in text:
                return domain

    return "other"


conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS faq (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        normalized_key TEXT NOT NULL,
        domain TEXT NOT NULL DEFAULT 'other'
    )
    """
)

cursor.execute("PRAGMA table_info(faq)")
columns = [row[1] for row in cursor.fetchall()]

if "normalized_key" not in columns:
    cursor.execute("ALTER TABLE faq ADD COLUMN normalized_key TEXT DEFAULT ''")

if "domain" not in columns:
    cursor.execute("ALTER TABLE faq ADD COLUMN domain TEXT DEFAULT 'other'")

cursor.execute("DELETE FROM faq")

faq_rows = [
    (
        question,
        answer,
        build_normalized_key(question),
        assign_domain(question),
    )
    for question, answer in sample_faqs
]

cursor.executemany(
    "INSERT INTO faq (question, answer, normalized_key, domain) VALUES (?, ?, ?, ?)",
    faq_rows
)

conn.commit()
conn.close()

print("faq.db を作成し、サンプルFAQを登録しました。")