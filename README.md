# 業務問い合わせ AI アシスタント

## 概要

社内問い合わせを想定し、質問の種類に応じて処理を自動分岐する AI アシスタントを実装した。  
LLM に依存せず処理を制御可能な設計とし、誤回答を防ぐ判定ロジックとログ観測を備える。

## デモ動画

アプリの動作イメージは以下の動画で確認できます。

[デモ動画を見る](assets/demo.mp4)

動画では、FAQ回答、RAG回答、数値回答、天気API回答、NO_ANSWER時の挙動を確認できます。

---

## アーキテクチャ

```
ユーザー質問
    ↓
router.py（正規表現・キーワードスコアリング・抑制語による判定）
    ↓
├─ NUMERIC  → pandas（LLM非依存の数値処理）
├─ API      → Open-Meteo（外部データのリアルタイム取得）
└─ SEARCH
       ├─ FAQ ヒット  → SQLite から回答
       └─ FAQ 未ヒット → RAG（ChromaDB + Gemini）で回答
```

手順・説明系の質問は FAQ をスキップして RAG へ直接遷移する。  
Gemini API レート制限（429）発生時は、生成を行わず根拠文書を直接提示するフォールバックを実装。

---

## 設計のポイント

### ① 処理の分離

質問の種類ごとに最適な処理を割り当てることで、精度とコストを両立した。

- FAQ：定型回答 → SQLite  一問一答は正規化＋一致判定で処理
- RAG：文書検索 → ChromaDB  長文マニュアルに対するセマンティック検索のため
- NUMERIC：データ計算 → pandas  数値計算をLLMに依存しないため
- API：外部データ → 天気API  外部データをリアルタイム取得するため

### ② ルーティング設計（router.py）

3 層の判定でカテゴリを決定する。

- **REGEX**：定型パターンを即時判定
- **SCORE**：キーワード加点でカテゴリを評価
- **INHIBITOR**：説明系キーワードでスコアを減点

スコア差が小さい場合や衝突時は、安全策として SEARCH へフォールバックする。

### ③ 回答制御（NO_ANSWER）

以下の条件を満たさない場合は回答しない。

- RAG の検索結果が距離閾値を満たさない
- 数値計算に必要な情報が不足
- API で地名が特定できない

### ④ ログによる観測

全ルート共通で `route / status / router_reason / router_scores` を記録。  
RAG 実行時は距離・閾値・判定結果などを補助情報としてログに付加し、  
「なぜそのルート・回答になったか」を後から追跡できる設計とした。

---

## 評価（ルーティング精度）

約60件のテストデータで検証し、調整後の正答率は **100%**（テストデータ上、設計したケースに対して）。  
誤判定は Safe Error（SEARCHへ回避）と Dangerous Error（誤ルート選択）に分類し、  
**Dangerous Error の抑制**を優先して調整した。

---

## 技術スタック

| 技術 | 用途 |
|------|------|
| Python / Streamlit | アプリ本体 |
| Gemini API | RAGの回答生成 |
| ChromaDB | セマンティック検索 |
| SQLite | FAQ の一致判定 |
| pandas | 数値計算（LLM 非依存） |
| Open-Meteo API | 天気データのリアルタイム取得 |

---

## セットアップ

```bash
pip install -r requirements.txt

# 環境変数の設定（Windows）
set GEMINI_API_KEY=your_api_key

# 初期化・起動
python init_db.py
python build_rag.py
streamlit run app.py

# 評価スクリプト
python evaluate.py
```

---

## ファイル構成

```
project/
├── app.py
├── router.py
├── numeric_qa.py
├── weather_api.py
├── faq_utils.py
├── log_db.py
├── llm.py
├── build_rag.py
├── init_db.py
├── evaluate.py
├── requirements.txt
├── README.md
├── docs/
│   └── manual.txt
├── data/
│   ├── sales.csv
│   ├── eval_questions.csv
│   ├── eval_results_v1.csv
│   ├── eval_results_v2.csv
│   └── eval_results_current.csv
└── assets/
    └── demo.mp4
```

---

## 今後の改善

- ルーティングの高度化（LLM による分類との比較）
- FAQ・RAG の検索精度改善
- 評価データの拡充
- ログを活用した改善サイクルの強化

---

## 生成ファイルについて

以下のファイル・フォルダは実行時に生成されるため、GitHubには含めていません。

- faq.db
- log.db
- chroma_db/

FAQデータベースとRAG用のChromaDBは、以下のコマンドで作成できます。

```bash
python init_db.py
python build_rag.py
```