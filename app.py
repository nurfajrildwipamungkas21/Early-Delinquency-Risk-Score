# app.py — EDRS Streamlit (Rule-based) — dynamic + polished narrative
import os, json, re, textwrap, hashlib, requests, io
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Page config + global CSS (professional font)
# -----------------------------
st.set_page_config(page_title="EDRS — Early Delinquency Risk Score", layout="wide")

USE_CDN_FONT = True
font_link = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<!-- Pastikan font ikon Material tetap tersedia agar tidak tampil sebagai teks -->
<link href="https://fonts.googleapis.com/css2?family=Material+Icons&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Icons+Outlined&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet">
""" if USE_CDN_FONT else ""

GLOBAL_CSS = f"""{font_link}
<style>
:root {{
  --font-body: "Inter","Segoe UI","Helvetica Neue",Arial,"Noto Sans",sans-serif;
  --fs-base: 13.5px;
}}
/* Jangan override font untuk ikon Material supaya tidak muncul teks 'keyboard_double_arrow_right' */
.material-icons,
.material-icons-outlined,
.material-symbols-outlined {{
  font-family: 'Material Symbols Outlined','Material Icons','Material Icons Outlined' !important;
  font-weight: normal; font-style: normal; line-height: 1;
}}
/* Terapkan font body ke elemen umum, kecuali ikon */
html, body, [data-testid="stAppViewContainer"] *:not(.material-icons):not(.material-icons-outlined):not(.material-symbols-outlined) {{
  font-family: var(--font-body);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}}
h1,h2,h3,h4 {{ font-weight: 600; letter-spacing:.2px; }}
.legal-text {{ font-size: var(--fs-base); line-height: 1.6; letter-spacing:.1px; }}
.small-note {{ color:#57606a; font-size:12px; }}
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# -----------------------------
# Settings / constants
# -----------------------------
PROMPT_VERSION = "v5-assertive-collateral"
CACHE_DIR = Path("./legal_conclusions"); CACHE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = CACHE_DIR / "index.json"

# Persist lokasi file upload agar jadi default
APP_DIR = Path(__file__).parent
UPLOADED_DIR = APP_DIR / "data_uploaded"; UPLOADED_DIR.mkdir(parents=True, exist_ok=True)
SAVED_PATH = UPLOADED_DIR / "latest_data"  # akan diberi ekstensi sesuai file yang disimpan

ALLOWED_PASAL = {
    "Perikatan/Wanprestasi": [
        "KUHPerdata Pasal 1238 (debitur dinyatakan lalai)",
        "KUHPerdata Pasal 1243 (ganti rugi karena wanprestasi)",
        "KUHPerdata Pasal 1244-1245 (alasan pembebasan/tidak dipenuhinya perikatan)"
    ],
    "Syarat & Asas Perjanjian": [
        "KUHPerdata Pasal 1320 (syarat sah perjanjian)",
        "KUHPerdata Pasal 1338 (pacta sunt servanda/kebebasan berkontrak)",
        "KUHPerdata Pasal 1339 (kepatutan/kebiasaan melengkapi perjanjian)"
    ],
    "Pembatalan/Perubahan": [
        "KUHPerdata Pasal 1266-1267 (pembatalan perjanjian bersyarat)"
    ],
    "PMH (opsional)": [
        "KUHPerdata Pasal 1365 (perbuatan melawan hukum, gunakan hanya bila relevan di luar kontrak)"
    ]
}

# -----------------------------
# Gemini settings — tanam DEMO key + akses secrets aman
# -----------------------------
DEFAULT_DEMO_KEY = "AIzaSyDd19AHP6cciyErg-bky3u07fXVGnXaraE"
DEFAULT_MODEL = "models/gemini-2.5-flash"

def _secret_get(key: str, default: str | None = None):
    try:
        return st.secrets[key]
    except Exception:
        return default

GEMINI_API_KEY = (
    os.environ.get("GEMINI_API_KEY_DEMO")
    or _secret_get("GEMINI_API_KEY_DEMO", None)
    or DEFAULT_DEMO_KEY
)

GEMINI_MODEL = (
    os.environ.get("GEMINI_MODEL")
    or _secret_get("GEMINI_MODEL", None)
    or DEFAULT_MODEL
)

GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_MODEL}:generateContent"

# -----------------------------
# Utility functions
# -----------------------------
def _load_index() -> dict:
    if INDEX_PATH.exists():
        try:
            return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _save_index(idx: dict) -> None:
    INDEX_PATH.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")

def _cache_file(id_val: int, sig: str) -> Path:
    return CACHE_DIR / f"ID-{id_val}-SIG-{sig[:8]}.txt"

def _insight_signature(id_val: int, insight_text: str) -> str:
    h = hashlib.sha1()
    h.update(f"{id_val}|{insight_text}|{PROMPT_VERSION}".encode("utf-8", errors="ignore"))
    return h.hexdigest()

def safe_div(a, b, eps=1e-6):
    return a / (np.abs(b) + eps)

def _call_gemini(prompt_text: str, timeout: int = 40) -> str:
    headers = {"Content-Type": "application/json; charset=utf-8"}
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    r = requests.post(GEMINI_ENDPOINT, params={"key": GEMINI_API_KEY},
                      headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()

def _sanitize_plain(text: str) -> str:
    t = text
    t = re.sub(r'^\s*#{1,6}\s*', '', t, flags=re.MULTILINE)
    t = t.replace('**', '')
    t = re.sub(r'^\s*[-*•]\s+', '', t, flags=re.MULTILINE)
    t = re.sub(r'^\s*\d+\.\s+', '', t, flags=re.MULTILINE)
    t = t.replace('*','').replace('_','').replace('—',' ')
    t = t.replace(';', ',').replace(':', ' ')
    t = re.sub(r'[ \t]+',' ', t)
    t = re.sub(r'\n{3,}','\n\n', t)
    t = re.sub(r'\bini\s+adalah\s+ringkasan\s+internal\b.*?(?:\.\s*|$)', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\bbukan\s+pendapat\s+hukum\s+final\b.*?(?:\.\s*|$)', '', t, flags=re.IGNORECASE)
    return t.strip()

# -----------------------------
# Data loading helpers (persist upload + robust schema)
# -----------------------------
def _saved_file_path() -> Path | None:
    # Cari file tersimpan apapun ekstensinya
    for ext in (".csv", ".xlsx", ".xls", ".parquet"):
        p = SAVED_PATH.with_suffix(ext)
        if p.exists():
            return p
    return None

def _save_uploaded(file: "UploadedFile") -> Path:
    # Simpan sesuai ekstensi asli
    suffix = Path(file.name).suffix.lower()
    dst = SAVED_PATH.with_suffix(suffix if suffix in [".csv", ".xlsx", ".xls", ".parquet"] else ".csv")
    bytes_data = file.getvalue()
    dst.write_bytes(bytes_data)
    return dst

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Samakan nama kolom ke schema pipeline dan buang kolom index otomatis."""
    new_cols = []
    for c in df.columns:
        s = str(c).strip()
        norm = re.sub(r'[\s\.\-]+', '', s).lower()

        new = None
        if norm == 'id':
            new = 'ID'
        elif norm in ('limitbal', 'limitbalance', 'limitamount', 'limit'):
            new = 'LIMIT_BAL'
        elif norm in ('defaultpaymentnextmonth', 'defaultpayment'):
            new = 'default.payment.next.month'
        else:
            m = re.fullmatch(r'pay([0-6])', norm)
            if m: new = f'PAY_{m.group(1)}'
            if not new:
                m = re.fullmatch(r'billamt([1-6])', norm)
                if m: new = f'BILL_AMT{m.group(1)}'
            if not new:
                m = re.fullmatch(r'payamt([1-6])', norm)
                if m: new = f'PAY_AMT{m.group(1)}'

        new_cols.append(new if new else s)

    df = df.rename(columns=dict(zip(df.columns, new_cols)))
    drop_unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if drop_unnamed:
        df = df.drop(columns=drop_unnamed)
    return df

def _read_file_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == '.csv':
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep=';')
        return _normalize_columns(df)

    if suf in ('.xlsx', '.xls'):
        # coba header=0 lalu header=1 (khusus UCI default xls biasanya header=1)
        for hdr in (0, 1):
            try:
                df = pd.read_excel(path, header=hdr)
                df = _normalize_columns(df)
                if {'ID', 'LIMIT_BAL'}.issubset(df.columns):
                    return df
            except Exception:
                pass
        df = pd.read_excel(path, header=0)
        return _normalize_columns(df)

    if suf == '.parquet':
        df = pd.read_parquet(path)
        return _normalize_columns(df)

    raise ValueError(f"Tipe file tidak didukung: {suf}")

@st.cache_data(show_spinner=False)
def load_data(source_hint: str, saved_path_str: str | None) -> pd.DataFrame:
    """
    Prioritas:
    1) file tersimpan (upload terakhir),
    2) sampel bawaan repo.
    """
    saved_path = Path(saved_path_str) if saved_path_str else None
    if saved_path and saved_path.exists():
        return _read_file_any(saved_path)

    # fallback: cari file sampel di repo (opsional)
    candidates = [
        APP_DIR / "data" / "UCI_Credit_Card.csv",
        APP_DIR / "UCI_Credit_Card.csv",
        APP_DIR / "data" / "default of credit card clients.xls",
        APP_DIR / "default of credit card clients.xls",
    ]
    for p in candidates:
        if p.exists():
            return _read_file_any(p)

    raise FileNotFoundError("Tidak ada data tersimpan maupun sampel bawaan. Silakan unggah file.")

# -----------------------------
# Core EDRS pipeline
# -----------------------------
def compute_features(df: pd.DataFrame):
    must_have = ["ID", "LIMIT_BAL", "default.payment.next.month"]
    for c in must_have:
        if c not in df.columns:
            raise ValueError(f"Kolom '{c}' wajib ada.")

    pay_candidates = [f"PAY_{i}" for i in [0,1,2,3,4,5,6]]
    pay_cols = [c for c in pay_candidates if c in df.columns]
    if len(pay_cols) < 3:
        raise ValueError(f"Kolom PAY_* terlalu sedikit: {pay_cols}")
    recent_preference = ["PAY_0","PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
    recent3 = [c for c in recent_preference if c in pay_cols][:3]

    bill_cols = [c for c in [f"BILL_AMT{i}" for i in range(1,7)] if c in df.columns]
    pmt_cols  = [c for c in [f"PAY_AMT{i}"  for i in range(1,7)] if c in df.columns]
    if not {"BILL_AMT1","PAY_AMT1"}.issubset(df.columns):
        raise ValueError("Perlu kolom BILL_AMT1 dan PAY_AMT1.")

    recent_vals = np.maximum(0, df[recent3].fillna(0).values)
    df["count_telat_3m"] = (recent_vals > 0).sum(axis=1)

    pay_mat = np.maximum(0, df[pay_cols].fillna(0).values)
    df["count_telat_6m"]   = (pay_mat > 0).sum(axis=1)
    df["max_tunggakan_6m"] = pay_mat.max(axis=1)

    df["ratio_bayar_last"] = safe_div(df["PAY_AMT1"], df["BILL_AMT1"])

    if len(bill_cols) >= 3:
        bill_mat = df[bill_cols].values.astype(float)
        idx = np.arange(1, len(bill_cols)+1, dtype=float)
        idx = (idx - idx.mean()) / (idx.std() + 1e-6)
        bill_std = (bill_mat - bill_mat.mean(1, keepdims=True)) / (bill_mat.std(1, keepdims=True) + 1e-6)
        df["bill_trend_up"] = (bill_std @ idx) > 0.3
    else:
        df["bill_trend_up"] = False

    dpd_proxy_source = "PAY_0" if "PAY_0" in df.columns else recent3[0]
    df["dpd_proxy_now"] = (df[dpd_proxy_source] > 0).astype(int)
    df["streak_telat2plus"] = ((np.maximum(0, df[recent3].fillna(0).values) >= 2).sum(axis=1) >= 1).astype(int)

    df["edrs_score"] = (
        2 * df["count_telat_3m"]
        + (df["count_telat_6m"] >= 3).astype(int)
        + (df["max_tunggakan_6m"] >= 2).astype(int)
        + (df["ratio_bayar_last"] < 0.7).astype(int)
        + df["bill_trend_up"].astype(int)
        + (df["dpd_proxy_now"] >= 1).astype(int)
        + (df["streak_telat2plus"] >= 1).astype(int)
    )

    def to_bucket(s:int)->str:
        if s >= 6: return "Very High"
        if s >= 4: return "High"
        if s >= 2: return "Med"
        if s >= 1: return "Low"
        return "Very Low"

    df["bucket"] = df["edrs_score"].apply(to_bucket)

    def nba(bucket:str)->str:
        return {
            "Very High": "Telepon atau WA hari ini dan lakukan opsi reschedule bila diperlukan",
            "High":      "Telepon hari ini dan follow-up dengan WA dalam 2 hari",
            "Med":       "WA reminder dan follow-up dalam 2 hari",
            "Low":       "WA otomatis atau reminder mingguan",
            "Very Low":  "Tidak ada tindakan atau mass reminder",
        }[bucket]

    df["next_best_action"] = df["bucket"].apply(nba)

    cols_keep = [
        "ID","LIMIT_BAL","edrs_score","bucket","next_best_action",
        "count_telat_3m","count_telat_6m","max_tunggakan_6m",
        "ratio_bayar_last","bill_trend_up","dpd_proxy_now","streak_telat2plus",
    ]
    if "default.payment.next.month" in df.columns:
        cols_keep += ["default.payment.next.month"]

    out = df[cols_keep].copy()
    bucket_order = ["Very High","High","Med","Low","Very Low"]
    out["bucket_cat"] = pd.Categorical(out["bucket"], categories=bucket_order, ordered=True)
    sort_keys = ["bucket_cat","edrs_score","dpd_proxy_now","ratio_bayar_last"]
    top_prior_all = out.sort_values(sort_keys, ascending=[True, False, False, True]).reset_index(drop=True)
    top_prior = top_prior_all[top_prior_all["bucket"].isin(["Very High","High"])].reset_index(drop=True)

    return df, out, top_prior, top_prior_all

def ratio_text(x: float) -> str:
    if pd.isna(x) or x < 1e-6:
        return "rasio pembayaran terakhir tidak ada"
    return f"rasio pembayaran terakhir sekitar {x*100:.0f}% dari tagihan terakhir"

def generate_insight(row_raw: pd.Series, row_skor: pd.Series, limit_pct: pd.Series) -> str:
    id_val    = int(row_raw["ID"])
    limit_bal = int(row_raw["LIMIT_BAL"])
    pct       = float(limit_pct.get(id_val, np.nan))
    pct_txt   = f"lebih tinggi daripada sekitar {pct*100:.0f}% pelanggan" if not np.isnan(pct) else "dalam kisaran umum portofolio"
    late6     = int(row_skor["count_telat_6m"])
    late3     = int(row_skor["count_telat_3m"])
    max_dpd   = int(row_skor["max_tunggakan_6m"])
    trend_txt = "meningkat" if bool(row_skor["bill_trend_up"]) else "stabil"
    ratio_desc = ratio_text(float(row_skor["ratio_bayar_last"]))
    bucket    = str(row_skor["bucket"])
    score     = int(row_skor["edrs_score"])
    action    = str(row_skor["next_best_action"])

    edrs_def = ("EDRS (Early Delinquency Risk Score) adalah skor aturan untuk mengestimasi risiko "
                "keterlambatan dini. Skor tinggi menandakan risiko menunggak lebih besar. "
                "Kisaran praktis pada data ini sekitar 0 hingga 12. Nilai 0 sampai 1 aman, "
                "2 sampai 3 menengah, 4 sampai 5 tinggi, dan 6 atau lebih sangat tinggi.")
    limit_def = ("LIMIT BAL adalah batas kredit aktif yang disetujui untuk nasabah. "
                 "Semakin besar limit, eksposur potensi kerugian lebih tinggi meskipun skor risiko tetap "
                 "ditentukan oleh perilaku bayar dan indikator lain.")
    reschedule_def = ("Reschedule yang dimaksud adalah penjadwalan ulang secara ringan untuk membantu "
                      "pemulihan kedisiplinan bayar. Contohnya memajukan atau memundurkan tanggal bayar "
                      "pada bulan berjalan, membuat rencana cicilan atas tunggakan, atau penyesuaian jangka "
                      "pendek lain. Bila kendala berlanjut, evaluasi restrukturisasi yang lebih formal dapat dipertimbangkan.")

    lines = [
        f"ID {id_val} memiliki limit kredit aktif (LIMIT_BAL) sebesar {limit_bal:,}. Nilai limit ini {pct_txt}.",
        f"Dalam enam bulan terakhir terjadi {late6} keterlambatan dengan {late3} kejadian pada tiga bulan terakhir.",
        f"Keterlambatan terlama tercatat {max_dpd} bulan.",
        f"Tren tagihan {trend_txt} dan {ratio_desc}.",
        f"Nasabah tergolong {bucket} dengan EDRS score {score}.",
        edrs_def,
        limit_def,
        f"Rekomendasi saat ini adalah {action}. {reschedule_def}"
    ]
    return " ".join(lines)

def _fallback_conclusion(row_raw: pd.Series, row_skor: pd.Series) -> str:
    bucket = str(row_skor["bucket"])
    dpdnow = int(row_skor["dpd_proxy_now"])
    ratio  = float(row_skor["ratio_bayar_last"])
    garis  = []
    if bucket in ("Very High", "High"):
        garis.append("Risiko gagal bayar tinggi sehingga dasar penagihan menekankan wanprestasi sesuai perikatan.")
        garis.append("Sebagai langkah awal tim melakukan klarifikasi kewajiban bayar, mengirim somasi yang proporsional, serta menawarkan restruktur ringan apabila layak.")
    else:
        garis.append("Risiko berada pada tingkat menengah atau lebih rendah sehingga pendekatan persuasif dan penguatan komitmen bayar lebih diutamakan.")
    if dpdnow >= 1:
        garis.append("Status DPD saat ini mengindikasikan keterlambatan yang dapat dikualifikasikan sebagai wanprestasi.")
    if ratio < 0.7:
        garis.append("Rasio pembayaran terakhir berada di bawah ambang normal sehingga mengindikasikan pelemahan kemampuan bayar.")
    garis.append("Dasar hukum mengacu pada hukum perdata, kontrak, dan perjanjian terutama klausul wanprestasi dan denda sesuai kesepakatan.")
    garis.append("Apabila debitur tetap mengelak atau menolak membayar tim menempuh somasi lanjutan dan gugatan perdata untuk pemenuhan perikatan atau ganti rugi.")
    garis.append("Karena adanya pasal sanksi yang disetujui para pihak di dalam perjanjian eksekusi jaminan atau penarikan barang dapat dilakukan apabila perjanjian memuat jaminan atau klausul sanksi yang sah dan seluruh prosedur formal dipenuhi misalnya melalui titel eksekutorial atau penetapan atau putusan pengadilan yang berlaku tanpa tindakan sepihak yang melanggar hukum.")
    garis.append("Izinkan tim collection melakukan dokumentasi seluruh tahapan penagihan sebagai bukti bahwa tata cara telah dilakukan secara prosedural serta simpan seluruh komunikasi tagihan dan pembayaran secara lengkap.")
    return " ".join(garis)

def _build_prompt_step1(id_val: int, row_raw: pd.Series, row_skor: pd.Series, insight_text: str) -> str:
    ctx = {
        "ID": int(row_raw.get("ID")),
        "LIMIT_BAL": int(row_raw.get("LIMIT_BAL")),
        "bucket": str(row_skor.get("bucket")),
        "edrs_score": int(row_skor.get("edrs_score")),
        "dpd_proxy_now": int(row_skor.get("dpd_proxy_now")),
        "ratio_bayar_last": float(row_skor.get("ratio_bayar_last")),
        "count_telat_3m": int(row_skor.get("count_telat_3m")),
        "count_telat_6m": int(row_skor.get("count_telat_6m")),
        "max_tunggakan_6m": int(row_skor.get("max_tunggakan_6m")),
        "next_best_action": str(row_skor.get("next_best_action")),
    }
    ctx_json = json.dumps(ctx, ensure_ascii=False)
    return textwrap.dedent(f"""
    Anda adalah analis hukum internal untuk Indonesia.
    Tulis satu paragraf naratif yang langsung ke inti tanpa heading, tanpa bullet, dan tanpa simbol khusus seperti # * - — : ;.
    Gunakan 7 sampai 12 kalimat, Bahasa Indonesia formal, ringkas dan jelas.
    Sebutkan jenis fasilitas secara eksplisit sebagai "limit kredit aktif (LIMIT_BAL)" saat menyebut besaran limit.
    Fokus pada ranah perdata/kontrak/perjanjian, jelaskan fakta pembayaran, indikasi wanprestasi, posisi hak dan kewajiban para pihak, opsi penanganan yang layak seperti somasi atau restruktur ringan, serta pentingnya dokumentasi bukti.
    Pada tahap ini jangan menyebut pasal atau nomor aturan apa pun.

    Ringkasan risiko:
    {insight_text}

    Data konteks JSON:
    {ctx_json}
    """)

def _build_prompt_step2(draft1_text: str) -> str:
    canon = json.dumps(ALLOWED_PASAL, ensure_ascii=False, indent=2)
    return textwrap.dedent(f"""
        Anda melanjutkan draf naratif berikut dari Tahap 1.
    Hasil akhir harus berupa satu paragraf natural, tanpa heading, tanpa daftar, dan tanpa simbol # * - — : ;.
    Masukkan rujukan pasal HANYA dari daftar kanonik di bawah ini, ditulis natural di dalam kalimat (misal: berdasarkan KUHPerdata Pasal 1238).
    Tegaskan apa yang boleh dilakukan tim collection bila debitur tetap mengelak atau menolak membayar:
    somasi lanjutan, gugatan perdata untuk pemenuhan perikatan atau ganti rugi, serta eksekusi jaminan/penarikan barang
    karena adanya pasal sanksi yang disepakati para pihak dalam perjanjian, dengan syarat perjanjian memang memuat jaminan/klausul sanksi yang sah dan seluruh prosedur hukum dipenuhi
    (contoh: jaminan fidusia dengan titel eksekutorial atau berdasarkan penetapan/putusan pengadilan).
    Tambahkan satu kalimat yang menyatakan bahwa tim collection diizinkan melakukan dokumentasi seluruh tahapan penagihan untuk bukti bahwa tata cara telah dilakukan secara prosedural.
    Hindari tindakan sepihak yang bertentangan dengan hukum, termasuk kekerasan atau ancaman, dan tekankan kepatuhan pada aturan perlindungan konsumen yang berlaku.

        Daftar kanonik pasal yang boleh dirujuk:
        {canon}

        Draf Tahap 1:
        {draft1_text}
        """)

def get_or_generate_conclusion(id_val: int, row_raw: pd.Series, row_skor: pd.Series, insight_text: str) -> str:
    sig = _insight_signature(id_val, insight_text)
    idx = _load_index()
    if str(id_val) in idx and idx[str(id_val)].get("sig") == sig:
        p = Path(idx[str(id_val)]["path"])
        if p.exists():
            try:
                return p.read_text(encoding="utf-8")
            except Exception:
                pass
    try:
        draft1 = _call_gemini(_build_prompt_step1(id_val, row_raw, row_skor, insight_text))
        final  = _call_gemini(_build_prompt_step2(draft1))
        text   = final.strip()
    except Exception:
        text = _fallback_conclusion(row_raw, row_skor)
    p = _cache_file(id_val, sig)
    try:
        p.write_text(text, encoding="utf-8")
        idx[str(id_val)] = {"sig": sig, "path": str(p)}
        _save_index(idx)
    except Exception:
        pass
    return text

def build_excel(top_prior_all: pd.DataFrame, cols_for_display: list) -> bytes:
    buf = io.BytesIO()
    now_str     = datetime.now().strftime("%d %B %Y, %H:%M WIB")
    kategori    = "Prioritas Koleksi — EDRS (Rule-based)"
    total_baris = len(top_prior_all)

    with pd.ExcelWriter(buf, engine="xlsxwriter") as xlw:
        wb = xlw.book

        def write_sheet_with_header(sheet_name, df_data, title):
            ws = wb.add_worksheet(sheet_name)
            fmt_title = wb.add_format({"bold":True,"font_name":"Calibri","font_size":14,"bg_color":"#C6E0B4","align":"left","valign":"vcenter"})
            fmt_label = wb.add_format({"bold":True,"font_name":"Calibri","bg_color":"#F2F2F2"})
            fmt_text  = wb.add_format({"font_name":"Calibri","font_size":11})
            fmt_th    = wb.add_format({"bold":True,"font_name":"Calibri","bg_color":"#F2F2F2","border":1})
            fmt_cell  = wb.add_format({"font_name":"Calibri","font_size":11,"border":1})
            ws.merge_range(0,0,0,4, title, fmt_title)
            ws.write(2,0,"Tanggal Laporan", fmt_label); ws.write(2,1, now_str, fmt_text)
            ws.write(3,0,"Total Baris", fmt_label);     ws.write(3,1, total_baris, fmt_text)
            ws.write(4,0,"Kategori", fmt_label);        ws.write(4,1, kategori, fmt_text)
            start_row = 7
            # header
            for j, col in enumerate(df_data.columns):
                ws.write(start_row, j, col, fmt_th)
            # rows
            for i in range(len(df_data)):
                for j, col in enumerate(df_data.columns):
                    ws.write(start_row+1+i, j, df_data.iloc[i, j], fmt_cell)
            # autosize
            for j, col in enumerate(df_data.columns):
                width = min(max(10, int(df_data[col].astype(str).map(len).quantile(0.90))+2), 40)
                ws.set_column(j, j, width)
            ws.freeze_panes(start_row+1, 0)

        col_display_names = {
            "ID": "ID", "LIMIT_BAL": "LIMIT BAL", "edrs_score": "EDRS score", "bucket": "Bucket",
            "next_best_action": "Next best action", "count_telat_3m": "Count telat 3m",
            "count_telat_6m": "Count telat 6m", "max_tunggakan_6m": "Max tunggakan 6m",
            "ratio_bayar_last": "Ratio bayar last", "bill_trend_up": "Bill trend up",
            "dpd_proxy_now": "DPD proxy now", "streak_telat2plus": "Streak telat 2+",
            "default.payment.next.month": "Default payment next month",
        }

        # All
        write_sheet_with_header(
            "All",
            top_prior_all[cols_for_display].rename(columns=col_display_names),
            "Status: Priorities EDRS (All Buckets, sorted)"
        )
        # Top Very High
        write_sheet_with_header(
            "Top_Very_High",
            top_prior_all[top_prior_all["bucket"]=="Very High"].head(200)[cols_for_display].rename(columns=col_display_names),
            "Status: Top Very High"
        )
        # Top High
        write_sheet_with_header(
            "Top_High",
            top_prior_all[top_prior_all["bucket"]=="High"].head(200)[cols_for_display].rename(columns=col_display_names),
            "Status: Top High"
        )
        # Summary
        summ_df = (top_prior_all.groupby("bucket")
                   .agg(n=("ID","count"),
                        avg_score=("edrs_score","mean"),
                        pay_ratio_lt_0_7=("ratio_bayar_last", lambda s: float((s<0.7).mean())),
                        dpd_now=("dpd_proxy_now","mean"))
                   .sort_index(ascending=False).reset_index())
        summ_df = summ_df.rename(columns={
            "bucket":"Bucket","n":"Jumlah nasabah","avg_score":"Rata-rata skor",
            "pay_ratio_lt_0_7":"Proporsi bayar <70%","dpd_now":"Proporsi DPD proxy"
        })
        write_sheet_with_header("Summary", summ_df, "Status: Ringkasan Bucket")

    buf.seek(0)
    return buf.read()

# -----------------------------
# UI — Sidebar (lebih clean)
# -----------------------------
st.sidebar.header("Pengaturan")

# Uploader saja (tanpa input folder)
uploaded = st.sidebar.file_uploader("Unggah data (CSV / XLS/XLSX)", type=["csv","xls","xlsx","parquet"])

# Simpan file yg baru diunggah agar jadi default
if uploaded is not None:
    dst = _save_uploaded(uploaded)
    st.sidebar.success(f"File tersimpan: {dst.name}. Aplikasi akan memakai data ini sebagai default.")
    # Bersihkan cache dan rerun agar data baru langsung aktif
    st.cache_data.clear()
    st.rerun()

# Info sumber data yang sedang dipakai
_saved = _saved_file_path()
if _saved:
    st.sidebar.caption(f"📄 Data aktif: **{_saved.name}** (tersimpan)")
else:
    st.sidebar.caption("⚠️ Belum ada data tersimpan. Gunakan sampel repo jika tersedia atau unggah file.")

top_n = st.sidebar.number_input("Top N prioritas", min_value=1, value=20, step=1)
show_bucket_only = st.sidebar.multiselect("Filter bucket", ["Very High","High","Med","Low","Very Low"], default=["Very High","High"])

# -----------------------------
# Load + compute
# -----------------------------
try:
    raw_df = load_data("saved-first", str(_saved) if _saved else "")
    base_df, out, top_prior, top_prior_all = compute_features(raw_df.copy())

    # kolom display
    cols_for_display = [
        "ID","LIMIT_BAL","edrs_score","bucket","next_best_action",
        "count_telat_3m","count_telat_6m","max_tunggakan_6m",
        "ratio_bayar_last","bill_trend_up","dpd_proxy_now","streak_telat2plus"
    ] + (["default.payment.next.month"] if "default.payment.next.month" in out.columns else [])

    st.title("EDRS — Early Delinquency Risk Score")
    st.caption("Rule-based scoring untuk prioritas penagihan • UI Streamlit")

    # -------------------------
    # Top Prioritas
    # -------------------------
    st.subheader("Top Prioritas Very High atau High")
    df_show = top_prior[top_prior["bucket"].isin(show_bucket_only)].head(int(top_n)).reset_index(drop=True)
    st.dataframe(df_show[cols_for_display], use_container_width=True, hide_index=True)

    # Download Excel
    with st.spinner("Menyiapkan Excel…"):
        excel_bytes = build_excel(top_prior_all, cols_for_display)
    st.download_button("⬇️ Unduh Excel report", data=excel_bytes, file_name="priorities_edrs_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # -------------------------
    # Viewer per-ID
    # -------------------------
    st.markdown("---")
    st.subheader("Viewer Interaktif — Masukkan ID untuk melihat detail")

    id_min, id_max = int(out["ID"].min()), int(out["ID"].max())
    default_id = int(df_show["ID"].iloc[0]) if len(df_show)>0 else int(out.iloc[0]["ID"])
    id_value = st.number_input("Cari ID", min_value=id_min, max_value=id_max, value=default_id, step=1)

    # ambil baris
    row_raw  = base_df[base_df["ID"]==id_value]
    row_skor = out[out["ID"]==id_value]
    if row_raw.empty or row_skor.empty:
        st.error("ID tidak ditemukan.")
    else:
        row_raw  = row_raw.iloc[0]
        row_skor = row_skor.iloc[0]

        # meta
        meta_cols_display = [c for c in ["ID","LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE"] if c in base_df.columns]
        meta_df = pd.DataFrame({**row_raw[meta_cols_display].to_dict(),
                                **row_skor[["edrs_score","bucket","next_best_action"]].to_dict()}, index=[0])
        st.dataframe(meta_df, use_container_width=True, hide_index=True)

        # PAY_* / BILL / PAY AMT
        pay_cols = [c for c in [f"PAY_{i}" for i in [0,1,2,3,4,5,6]] if c in base_df.columns]
        bill_cols = [c for c in [f"BILL_AMT{i}" for i in range(1,7)] if c in base_df.columns]
        pmt_cols  = [c for c in [f"PAY_AMT{i}"  for i in range(1,7)] if c in base_df.columns]

        if pay_cols:
            st.markdown("#### PAY status keterlambatan per bulan")
            st.dataframe(pd.DataFrame([row_raw[pay_cols]], columns=pay_cols), use_container_width=True, hide_index=True)
        if bill_cols:
            st.markdown("#### BILL AMT 1 sampai 6 (tagihan 6 bulan)")
            st.dataframe(pd.DataFrame([row_raw[bill_cols]], columns=bill_cols), use_container_width=True, hide_index=True)
        if pmt_cols:
            st.markdown("#### PAY AMT 1 sampai 6 (pembayaran 6 bulan)")
            st.dataframe(pd.DataFrame([row_raw[pmt_cols]], columns=pmt_cols), use_container_width=True, hide_index=True)

        # Ringkasan rules
        st.markdown("#### Ringkasan Rules")
        rules_df = pd.DataFrame([{
            "Count telat 3m": int(row_skor["count_telat_3m"]),
            "Count telat 6m": int(row_skor["count_telat_6m"]),
            "Max tunggakan 6m": int(row_skor["max_tunggakan_6m"]),
            "Ratio bayar last": f"{float(row_skor['ratio_bayar_last']):.2f}",
            "Bill trend up": "Yes" if bool(row_skor["bill_trend_up"]) else "No",
            "DPD proxy now": "Yes" if int(row_skor["dpd_proxy_now"]) else "No",
            "Streak telat 2+": "Yes" if int(row_skor["streak_telat2plus"]) else "No",
        }])
        st.dataframe(rules_df, use_container_width=True, hide_index=True)

        # Label target (jika ada)
        if "default.payment.next.month" in row_skor:
            st.markdown("#### Label Target")
            st.dataframe(pd.DataFrame({"Default payment next month":[row_skor["default.payment.next.month"]]}),
                         use_container_width=True, hide_index=True)

        # Insight
        st.markdown("#### Insight")
        limit_pct = base_df["LIMIT_BAL"].rank(pct=True); limit_pct.index = base_df["ID"].values
        insight_text = generate_insight(row_raw, row_skor, limit_pct)
        st.markdown(f"<div class='legal-text'>{insight_text}</div>", unsafe_allow_html=True)

        # Kesimpulan (LLM/fallback)
        st.markdown("#### Kesimpulan")
        with st.spinner("Menyusun narasi…"):
            kesimpulan_text = get_or_generate_conclusion(id_value, row_raw, row_skor, insight_text)
            kesimpulan_text = _sanitize_plain(kesimpulan_text)
        st.markdown(f"<div class='legal-text' style='white-space:pre-wrap'>{kesimpulan_text}</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Gagal memproses: {e}")
    st.stop()
