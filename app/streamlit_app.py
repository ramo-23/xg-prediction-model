"""Interactive Streamlit dashboard for xG visualizations and prediction."""

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.calibration import calibration_curve

try:
    import shap
    shap_available = True
except Exception:
    shap_available = False


# --- Utilities ---
def find_repo_root(start=Path.cwd(), markers=("setup.py", "requirements.txt", "README.md")):
    cur = start.resolve()
    for _ in range(10):
        if any((cur / m).exists() for m in markers):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()

repo_root = find_repo_root()

# --- Streamlit page styling ---
st.set_page_config(
    page_title="xG Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --bg: #FFFFFF;
        --panel: #FFFFFF;
        --panel-hover: #F1F5F9;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #64748B;
        --accent: #3B82F6;
        --accent-hover: #2563EB;
        --accent-light: #DBEAFE;
        --success: #10B981;
        --success-light: #D1FAE5;
        --warning: #F59E0B;
        --warning-light: #FEF3C7;
        --danger: #EF4444;
        --border: #E2E8F0;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        --radius: 12px;
        --radius-lg: 16px;
        --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* === GLOBAL RESETS === */
    * {
        font-synthesis: none;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        background: var(--bg) !important;
        color: var(--text-primary) !important;
    }

    /* Hide Streamlit branding except the header (keep built-in sidebar chevron) */
    #MainMenu, footer {
        visibility: hidden;
    }

    /* === LAYOUT === */
    .main > div {
        /* Allow the main content to use available width and leave room for the sidebar */
        max-width: none;
        margin: 0 1.5rem 0 320px;
        padding: 2rem 1.5rem;
    }

    /* === TYPOGRAPHY === */
    .big-title {
        font-size: clamp(28px, 5vw, 42px);
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text-primary);
        margin: 0 0 0.5rem 0;
        line-height: 1.2;
    }

    h1, h2, h3 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
    }

    h2 { font-size: 1.875rem !important; margin-top: 2rem !important; }
    h3 { font-size: 1.5rem !important; }

    p, .stMarkdown, .stText {
        color: var(--text-secondary) !important;
        line-height: 1.6;
        font-size: 0.938rem;
    }

    /* === CARDS === */
    .card {
        background: var(--panel);
        padding: 1.5rem;
        border-radius: var(--radius);
        /* Cards: no drop shadow to reduce visual noise */
        box-shadow: none;
        border: 1px solid var(--border);
        transition: var(--transition);
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }

    .card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        /* Use a single accent color (no gradients) for a clean accent line */
        background: var(--accent);
        opacity: 0;
        transition: var(--transition);
    }

    .card:hover {
        /* Keep layout stable on hover; subtle border change only */
        transform: none;
        box-shadow: none;
        border-color: var(--accent-light);
    }

    .card:hover::before {
        opacity: 0.9;
    }

    .card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent);
        line-height: 1;
    }

    .card div {
        font-size: 0.875rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }

    /* === METRICS === */
    [data-testid="metric-container"] {
        background: var(--panel);
        padding: 1.25rem;
        border-radius: var(--radius);
        border: 1px solid var(--border);
        box-shadow: var(--shadow-sm);
    }

    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--accent) !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        font-size: 0.875rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }

    /* === BUTTONS === */
    .stButton > button, .stDownloadButton > button {
        background: var(--accent) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 0.5rem !important;
        font-weight: 600 !important;
        font-size: 0.938rem !important;
        cursor: pointer;
        transition: var(--transition) !important;
        box-shadow: var(--shadow-sm);
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        background: var(--accent-hover) !important;
        transform: translateY(-2px);
        box-shadow: var(--shadow);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Secondary button style */
    .stButton > button[kind="secondary"] {
        background: var(--panel) !important;
        color: var(--accent) !important;
        border: 1px solid var(--border) !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background: var(--panel-hover) !important;
        border-color: var(--accent) !important;
    }

    /* === SIDEBAR === */
    [data-testid="stSidebar"] {
        background: var(--panel) !important;
        border-right: 1px solid var(--border);
        box-shadow: var(--shadow);
        padding: 1.5rem 1rem !important;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary) !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--text-primary) !important;
        font-weight: 500;
    }

    /* Sidebar radio buttons */
    [data-testid="stSidebar"] .stRadio > label {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        margin-bottom: 0.5rem !important;
    }

    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
        background: var(--panel-hover);
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: var(--transition);
        border: 1px solid transparent;
    }

    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
        background: var(--accent-light);
        border-color: var(--accent);
    }

    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label[data-checked="true"] {
        background: var(--accent);
        color: white;
        font-weight: 600;
    }

    /* === FORMS & INPUTS === */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {
        border: 1px solid var(--border) !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem 1rem !important;
        background: var(--panel) !important;
        color: var(--text-primary) !important;
        transition: var(--transition) !important;
    }

    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-light) !important;
        outline: none !important;
    }

    /* Sliders */
    .stSlider > div > div > div > div {
        background: var(--accent) !important;
    }

    .stSlider [role="slider"] {
        background: var(--accent) !important;
        border: 3px solid white !important;
        box-shadow: var(--shadow) !important;
    }

    /* === DATAFRAMES & TABLES === */
    .stDataFrame {
        border: 1px solid var(--border);
        border-radius: var(--radius);
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }

    .stDataFrame [data-testid="stDataFrameResizable"] {
        border-radius: var(--radius);
    }

    .stDataFrame thead {
        background: var(--panel-hover) !important;
    }

    .stDataFrame thead th {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 1rem !important;
        border-bottom: 2px solid var(--border) !important;
    }

    .stDataFrame tbody td {
        color: var(--text-secondary) !important;
        padding: 0.875rem 1rem !important;
        border-bottom: 1px solid var(--border) !important;
    }

    .stDataFrame tbody tr:hover {
        background: var(--panel-hover) !important;
    }

    /* === CHARTS === */
    .stPlotlyChart, .stAltairChart, .stPyplot {
        background: var(--panel);
        padding: 1.5rem;
        border-radius: var(--radius);
        border: 1px solid var(--border);
        box-shadow: var(--shadow-sm);
        margin: 1rem 0;
    }

    /* === EXPANDERS === */
    .streamlit-expanderHeader {
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        padding: 1rem 1.25rem !important;
        transition: var(--transition) !important;
    }

    .streamlit-expanderHeader:hover {
        background: var(--panel-hover) !important;
        border-color: var(--accent) !important;
    }

    .streamlit-expanderContent {
        border: 1px solid var(--border) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius) var(--radius) !important;
        background: var(--panel) !important;
    }

    /* === ALERTS === */
    .stAlert {
        border-radius: var(--radius) !important;
        border: 1px solid !important;
        padding: 1rem 1.25rem !important;
    }

    .stSuccess {
        background: var(--success-light) !important;
        border-color: var(--success) !important;
        color: #065F46 !important;
    }

    .stWarning {
        background: var(--warning-light) !important;
        border-color: var(--warning) !important;
        color: #92400E !important;
    }

    .stError {
        background: #FEE2E2 !important;
        border-color: var(--danger) !important;
        color: #991B1B !important;
    }

    .stInfo {
        background: var(--accent-light) !important;
        border-color: var(--accent) !important;
        color: #1E40AF !important;
    }

    /* === PROGRESS BAR === */
    .stProgress > div > div > div {
        background: var(--accent) !important;
        border-radius: 9999px !important;
    }

    .stProgress > div > div {
        background: var(--border) !important;
        border-radius: 9999px !important;
    }

    /* === LINKS === */
    a {
        color: var(--accent) !important;
        text-decoration: none;
        border-bottom: 1px solid transparent;
        transition: var(--transition);
    }

    a:hover {
        border-bottom-color: var(--accent);
    }

    /* === DIVIDER === */
    hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 2rem 0;
    }

    /* Sidebar toggle removed (custom toggle was removed per user request) */

    /* === RESPONSIVE === */
    @media (max-width: 768px) {
        .main > div {
            padding: 1rem;
            margin-left: 0; /* remove left gutter on small screens */
        }
        
        .big-title {
            font-size: 1.75rem;
        }
        
        .card {
            padding: 1rem;
        }
        
        [data-testid="stSidebar"] {
            padding: 1rem 0.75rem !important;
        }
    }

    /* === ANIMATIONS === */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .card, [data-testid="metric-container"], .stPlotlyChart, .stAltairChart {
        animation: fadeIn 0.4s ease-out;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- App session defaults & data controls ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

with st.sidebar.expander('Data settings', expanded=False):
    use_full = st.checkbox('Use full dataset (may be slow)', value=False)
    sample_pct = st.slider('Sample percent when not using full', min_value=1, max_value=100, value=20)
    max_rows = st.number_input('Max rows to read for sampling', value=200000, min_value=1000, step=1000)
    st.session_state['use_full_data'] = use_full
    st.session_state['sample_frac'] = sample_pct / 100.0
    st.session_state['max_rows'] = int(max_rows)


# --- Data & model loading (cached) ---
@st.cache_data(show_spinner=False)
def load_processed_data(use_full: bool, sample_frac: float, max_rows: int):
    proc_dir = repo_root / 'data' / 'processed'
    files = sorted(proc_dir.glob('processed_shots_*.csv'), key=lambda p: p.stat().st_mtime) if proc_dir.exists() else []
    if not files:
        return None, None
    data_path = files[-1]
    try:
        fsize = data_path.stat().st_size
    except Exception:
        fsize = 0

    # If file is large, read a bounded number of rows and sample
    try:
        if not use_full and fsize > 50 * 1024 * 1024:
            df_tmp = pd.read_csv(data_path, nrows=max_rows)
            if sample_frac and 0 < sample_frac < 1.0:
                df = df_tmp.sample(frac=sample_frac, random_state=42)
            else:
                if len(df_tmp) > 100000:
                    df = df_tmp.sample(n=100000, random_state=42)
                else:
                    df = df_tmp
        else:
            df = pd.read_csv(data_path)
    except Exception:
        # fallback to full read
        df = pd.read_csv(data_path)

    return df, data_path


@st.cache_resource(show_spinner=False)
def load_models():
    model_files = {
        'logistic': repo_root / 'results' / 'metrics' / 'model_logistic_calibrated.joblib',
        'random_forest': repo_root / 'results' / 'metrics' / 'model_random_forest_calibrated.joblib',
        'xgboost': repo_root / 'results' / 'metrics' / 'model_xgboost_calibrated.joblib',
        'neural_network': repo_root / 'results' / 'metrics' / 'model_neural_network_calibrated.joblib',
    }
    models_local = {}
    for name, p in model_files.items():
        if p.exists():
            try:
                models_local[name] = joblib.load(p)
            except Exception:
                continue
    return models_local


# Load once (cached)
df, data_path = load_processed_data(st.session_state.get('use_full_data', False), st.session_state.get('sample_frac', 0.2), st.session_state.get('max_rows', 200000))
if df is None:
    st.error('No processed_shots_*.csv files found in data/processed/')
    st.stop()

if 'outcome' not in df.columns:
    st.error('Processed data missing outcome column')
    st.stop()

y = df['outcome'].astype(str).str.lower().eq('goal').astype(int)

# --- Features ---
numeric_feats = [c for c in ['distance', 'minute_num'] if c in df.columns]
binary_feats = [c for c in ['body_head', 'body_foot', 'body_other', 'big_chance', 'half'] if c in df.columns]
cat_feats = [c for c in ['shot_type', 'assist_type'] if c in df.columns]
feature_cols = numeric_feats + binary_feats + cat_feats

# --- Load models ---
model_files = {
    'logistic': repo_root / 'results' / 'metrics' / 'model_logistic_calibrated.joblib',
    'random_forest': repo_root / 'results' / 'metrics' / 'model_random_forest_calibrated.joblib',
    'xgboost': repo_root / 'results' / 'metrics' / 'model_xgboost_calibrated.joblib',
    'neural_network': repo_root / 'results' / 'metrics' / 'model_neural_network_calibrated.joblib',
}

models = {}
for name, p in model_files.items():
    if p.exists():
        try:
            models[name] = joblib.load(p)
        except Exception:
            pass

# --- Navigation ---
page = st.sidebar.radio(
    'Page',
    ['Home', 'xG Predictor', 'Model Performance', 'Player Analysis', 'Team Analysis'],
    key='page'
)

# --- Simple xG heuristic ---
def simple_xg_heuristic(distance, angle, body_part, shot_type):
    """Basic heuristic for expected goals."""
    d = float(distance)
    base = max(0.001, np.exp(-d / 18.0))
    angle_factor = 1.0 - (float(angle) / 180.0) * 0.35
    body_factor = 1.0
    if str(body_part).lower().startswith('head'):
        body_factor = 0.8
    elif str(body_part).lower().startswith('other'):
        body_factor = 0.6
    shot_type_factor = 1.0
    if str(shot_type).lower().startswith('pen'):
        shot_type_factor = 3.0
    elif str(shot_type).lower().startswith('corner'):
        shot_type_factor = 0.5
    return float(np.clip(base * angle_factor * body_factor * shot_type_factor, 0.0, 0.999))

# --- Unwrap model ---
def unwrap_model(model):
    """Extracts the underlying estimator from pipelines or calibrated classifiers."""
    m = model
    try:
        if hasattr(m, 'named_steps'):
            last_step = list(m.named_steps.values())[-1]
            m = last_step
    except Exception:
        pass
    try:
        if hasattr(m, 'calibrated_classifiers_') and getattr(m, 'calibrated_classifiers_'):
            first = m.calibrated_classifiers_[0]
            if hasattr(first, 'estimator'):
                return first.estimator
            return first
    except Exception:
        pass
    for attr in ('base_estimator_', 'base_estimator', 'estimator_', 'estimator'):
        try:
            if hasattr(m, attr):
                candidate = getattr(m, attr)
                if candidate is not None:
                    return candidate
        except Exception:
            continue
    return m

# --- Home page ---
if page == 'Home':
    st.markdown(f"<div class='big-title'>xG Prediction Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>Interactive expected goals (xG) explorer & predictor</div>", unsafe_allow_html=True)

    # Key stats cards
    total_shots = len(df)
    models_loaded = list(models.keys())
    best_model_acc = None
    metrics_p = repo_root / 'results' / 'metrics' / 'metrics_summary.json'
    if metrics_p.exists():
        try:
            import json
            mdata = json.load(open(metrics_p))
            accs = [float(v.get('classification_report', {}).get('accuracy', 0)) for v in mdata.values()]
            best_model_acc = max(accs) if accs else None
        except Exception:
            best_model_acc = None

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown(f"<div class='card'><h3>{total_shots:,}</h3><div>Total shots analyzed</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='card'><h3>{len(models_loaded)}</h3><div>Models loaded</div></div>", unsafe_allow_html=True)
    with c3:
        acc_text = f"{best_model_acc:.3f}" if best_model_acc else 'N/A'
        st.markdown(f"<div class='card'><h3>{acc_text}</h3><div>Best model accuracy</div></div>", unsafe_allow_html=True)

    # Quick navigation buttons
    st.markdown('---')
    st.subheader('Quick Navigation')
    nav_cols = st.columns(3)
    pages = ['xG Predictor', 'Model Performance', 'Player Analysis']
    for col, p in zip(nav_cols, pages):
        with col:
            if st.button(p):
                st.session_state['page'] = p
                rerun_fn = getattr(st, 'experimental_rerun', None)
                if callable(rerun_fn):
                    rerun_fn()

    # xG explainer
    st.markdown('---')
    st.subheader('What is xG?')
    with st.expander('Read more about expected goals (xG)'):
        st.write(
            'Expected Goals (xG) is a probabilistic measure that assigns a value between 0 and 1 to a shot, '
            'representing the likelihood that the shot will result in a goal. Values close to 1 indicate very '
            'high-quality chances (e.g., penalties), while values near 0 indicate low-probability attempts.'
        )
        st.write('Factors often used in xG models include: distance to goal, shot angle, body part, shot type, and game context.')

    # Footer
    st.markdown('---')
    st.markdown(
        "<div style='font-size:0.9em; color:#64748B;'>Contact & Source: <a href='https://github.com/ramo-23/xg-prediction-model'>GitHub</a> • manyeliramokhele23@gmail.com/div>",
        unsafe_allow_html=True
    )


if page == 'xG Predictor':
    st.header('xG Predictor Tool')
    with st.form('predictor'):
        col1, col2 = st.columns(2)
        with col1:
            distance = st.slider('Distance from goal (m)', 0.0, 40.0, 12.0)
            angle = st.slider('Shot angle (degrees)', 0.0, 180.0, 45.0)
            body_part = st.selectbox('Body part', ['Foot', 'Head', 'Other'])
            shot_type = st.selectbox('Shot type', ['Open Play', 'Free Kick', 'Corner', 'Penalty'])
        with col2:
            game_state = st.selectbox('Game state', ['Winning', 'Drawing', 'Losing'])
            minute = st.slider('Match minute', 0, 120, 45)
            home_away = st.radio('Home/Away', ['Home', 'Away'])
            model_choice = st.selectbox('Model (optional)', [''] + list(models.keys()) if models else [''])
        submit = st.form_submit_button('Get xG')

    if submit:
        feat_row = {}
        if 'distance' in numeric_feats:
            feat_row['distance'] = float(distance)
        if 'minute_num' in numeric_feats:
            feat_row['minute_num'] = int(minute)
        for b in binary_feats:
            if b == 'body_head':
                feat_row[b] = 1 if body_part.lower().startswith('head') else 0
            elif b == 'body_foot':
                feat_row[b] = 1 if body_part.lower().startswith('foot') else 0
            elif b == 'body_other':
                feat_row[b] = 1 if body_part.lower().startswith('other') else 0
            else:
                feat_row[b] = 0
        if 'shot_type' in cat_feats:
            feat_row['shot_type'] = shot_type

        feat_row['is_home'] = 1 if home_away == 'Home' else 0
        feat_row['home_away'] = home_away
        _ord = {'Winning': 2, 'Drawing': 1, 'Losing': 0}
        feat_row['game_state'] = _ord.get(game_state, 1)
        for gs in ['Winning', 'Drawing', 'Losing']:
            feat_row[f'game_state_{gs}'] = 1 if game_state == gs else 0

        pred_xg = None
        used_model = None
        if model_choice and model_choice in models:
            used_model = models[model_choice]
            row_model = pd.DataFrame([
                {col: feat_row.get(col,
                    (0 if col in numeric_feats + binary_feats else (df[col].mode().iloc[0] if col in df.columns else 'missing')))
                 for col in feature_cols}
            ]) if feature_cols else pd.DataFrame([feat_row])
            try:
                pred_xg = float(used_model.predict_proba(row_model)[:, 1][0])
            except Exception:
                pred_xg = None

        if pred_xg is None:
            pred_xg = simple_xg_heuristic(distance, angle, body_part, shot_type)

        out_col, gauge_col = st.columns([2, 1])
        with out_col:
            st.subheader('Predicted xG')
            st.metric(label='Predicted xG', value=f'{pred_xg:.3f}', delta=f'{pred_xg*100:.1f}%')
            if distance <= 6:
                compare = 'Similar to a shot from the penalty spot edge'
            elif distance <= 12:
                compare = 'Similar to a close-range shot inside the box'
            elif distance <= 20:
                compare = 'Similar to a long-range inside-the-box attempt'
            else:
                compare = 'Similar to a long-range effort'
            st.write(compare)
            if pred_xg >= 0.5:
                st.success('High quality chance!')
            elif pred_xg >= 0.15:
                st.info('Medium quality chance')
            else:
                st.warning('Low probability shot')
        with gauge_col:
            prog = float(np.clip(pred_xg, 0.0, 1.0))
            st.subheader('Probability')
            st.progress(prog)

        if shap_available and used_model is not None and feature_cols:
            try:
                row_for_model = pd.DataFrame([
                    {col: feat_row.get(col,
                        (0 if col in numeric_feats + binary_feats else (df[col].mode().iloc[0] if col in df.columns else 'missing')))
                     for col in feature_cols}
                ])
                explainer = shap.Explainer(used_model, row_for_model)
                expl_vals = explainer(row_for_model)
                vals = getattr(expl_vals, 'values', None)
                if vals is None:
                    vals = np.array(expl_vals)
                if isinstance(vals, list):
                    vals = vals[-1]
                vals = np.asarray(vals)
                if vals.ndim == 3:
                    vals = vals[:, -1, :]
                contribs = vals[0]
                df_shap = pd.DataFrame({'feature': row_for_model.columns, 'contribution': contribs})
                df_shap = df_shap.sort_values('contribution', key=lambda s: s.abs(), ascending=False)
                st.subheader('Feature contributions (SHAP)')
                st.dataframe(df_shap.reset_index(drop=True))
                st.bar_chart(df_shap.set_index('feature')['contribution'])
            except Exception as e:
                st.info('SHAP explanation not available for this model: %s' % e)
        else:
            st.subheader('Feature contributions (approx.)')
            base = simple_xg_heuristic(20.0, 90.0, 'Foot', 'Open Play')
            dist_contrib = simple_xg_heuristic(distance, 90.0, 'Foot', 'Open Play') - base
            angle_contrib = simple_xg_heuristic(distance, angle, 'Foot', 'Open Play') - simple_xg_heuristic(distance, 90.0, 'Foot', 'Open Play')
            body_contrib = simple_xg_heuristic(distance, angle, body_part, 'Open Play') - simple_xg_heuristic(distance, angle, 'Foot', 'Open Play')
            st.write(f'- Distance effect: {dist_contrib:+.3f}')
            st.write(f'- Angle effect: {angle_contrib:+.3f}')
            st.write(f'- Body part effect: {body_contrib:+.3f}')


if page == 'Model Performance':
    st.header('Model Performance')
    metrics_p = repo_root / 'results' / 'metrics' / 'metrics_summary.json'
    metrics = {}
    if metrics_p.exists():
        try:
            metrics = pd.read_json(metrics_p)
        except Exception:
            try:
                import json
                metrics = json.load(open(metrics_p))
            except Exception:
                metrics = {}
    
    def _is_empty_metrics(obj):
        try:
            if obj is None:
                return True
            if isinstance(obj, pd.DataFrame):
                return obj.empty
            if isinstance(obj, dict):
                return len(obj) == 0
            return len(obj) == 0
        except Exception:
            return True
    
    if _is_empty_metrics(metrics):
        st.warning('No metrics found. Run training first (scripts/train_models.py).')
    else:
        rows = []
        for m in ['logistic', 'random_forest', 'xgboost']:
            if m in metrics:
                entry = metrics[m]
                acc = entry.get('classification_report', {}).get('accuracy') if isinstance(entry.get('classification_report'), dict) else None
                brier = entry.get('brier_score')
                roc = entry.get('roc_auc')
                ttime = entry.get('train_time_seconds', None)
                rows.append({'model': m, 'brier_score': brier, 'roc_auc': roc, 'accuracy': acc, 'train_time_s': ttime})
        df_comp = pd.DataFrame(rows).set_index('model')
        best = df_comp['roc_auc'].idxmax() if 'roc_auc' in df_comp.columns and not df_comp['roc_auc'].isnull().all() else None
        st.subheader('Model comparison')
        st.dataframe(df_comp.style.format({k: '{:.3f}' for k in ['brier_score', 'roc_auc', 'accuracy'] if k in df_comp.columns}))

        if feature_cols:
            with st.expander('Show detailed ROC / Calibration / Confusion (heavy)', expanded=False):
                X = df[feature_cols]
                X_train_full, X_hold, y_train_full, y_hold = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
                X_val, X_test, y_val, y_test = train_test_split(X_hold, y_hold, test_size=0.5, stratify=y_hold, random_state=42)
                fig, axs = plt.subplots(1, 2, figsize=(14, 6))
                ax, ax2 = axs
                for name, mdl in models.items():
                    try:
                        proba = mdl.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, proba)
                        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
                        prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)
                        ax2.plot(prob_pred, prob_true, marker='o', label=f"{name} (Brier={brier_score_loss(y_test, proba):.3f})")
                    except Exception:
                        continue
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
                ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC curves (test set)'); ax.legend(loc='lower right'); ax.grid(True)
                ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
                ax2.set_xlabel('Mean predicted probability'); ax2.set_ylabel('Fraction of positives'); ax2.set_title('Calibration (reliability diagram)'); ax2.legend(); ax2.grid(True)
                st.pyplot(fig)

                if best and best in models:
                    bm = models[best]
                    try:
                        y_pred = (bm.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
                        cm = confusion_matrix(y_test, y_pred)
                        st.subheader(f'Confusion matrix — {best}')
                        st.write(cm)
                        st.write('Precision:', precision_score(y_test, y_pred, zero_division=0))
                        st.write('Recall:', recall_score(y_test, y_pred, zero_division=0))
                        st.write('Accuracy:', accuracy_score(y_test, y_pred))
                    except Exception:
                        st.info('Could not compute confusion matrix for best model')

        else:
            st.info('No feature columns detected; cannot compute ROC/Calibration/Confusion matrix.')


if page == 'Player Analysis':
    st.header('Player Analysis — xG vs Actual Goals')
    player_col = next((c for c in ['player', 'player_name', 'shooter', 'playerId'] if c in df.columns), None)
    team_col = next((c for c in ['team', 'team_name', 'teamId'] if c in df.columns), None)

    pref = next((m for m in ['logistic', 'random_forest', 'xgboost', 'neural_network'] if m in models), None)
    if pref:
        mapper = models[pref]
        try:
            proba_all = mapper.predict_proba(df[feature_cols])[:, 1] if feature_cols else mapper.predict_proba(df)[:, 1]
        except Exception:
            try:
                proba_all = mapper.predict_proba(df)[:, 1]
            except Exception:
                proba_all = np.full(len(df), np.nan)
    else:
        proba_all = np.full(len(df), np.nan)

    df['_pred_xg'] = proba_all
    df['_is_goal'] = y

    if not player_col:
        st.info('No player column found in data — Player Analysis requires a player identifier column.')
    else:
        grp = df.groupby(player_col)
        shots = grp.size().rename('shots')
        xg_sum = grp['_pred_xg'].sum().rename('xg')
        goals = grp['_is_goal'].sum().rename('goals')
        if team_col:
            team_first = grp[team_col].first().rename('team')
        else:
            team_first = pd.Series(index=shots.index, data=[None]*len(shots), name='team')

        player_stats = pd.concat([team_first, shots, goals, xg_sum], axis=1).fillna(0)
        player_stats['difference'] = player_stats['goals'] - player_stats['xg']
        player_stats['conversion_rate'] = player_stats['goals'] / player_stats['shots'].replace(0, np.nan)
        player_stats = player_stats.reset_index().rename(columns={player_col: 'player'})

        st.subheader('xG vs Actual Goals')
        if not player_stats.empty:
            xmin = float(player_stats['xg'].min())
            xmax = float(player_stats['xg'].max())
            line_df = pd.DataFrame({'x': [xmin, xmax], 'y': [xmin, xmax]})
            base_chart = alt.Chart(player_stats).mark_circle(size=60).encode(
                x=alt.X('xg:Q', title='Total xG'),
                y=alt.Y('goals:Q', title='Actual Goals'),
                color=alt.Color('team:N') if team_col else alt.value('steelblue'),
                tooltip=['player:N', 'team:N', 'shots:Q', 'goals:Q', 'xg:Q', 'difference:Q']
            )
            line = alt.Chart(line_df).mark_line(color='black', strokeDash=[4,4]).encode(x='x:Q', y='y:Q')
            st.altair_chart((base_chart + line).interactive(), use_container_width=True)

            st.subheader('Top Overperformers')
            over = player_stats.sort_values('difference', ascending=False).head(20)
            st.dataframe(over[['player', 'team', 'shots', 'goals', 'xg', 'difference']].reset_index(drop=True))

            st.subheader('Top Underperformers')
            under = player_stats.sort_values('difference', ascending=True).head(20)
            st.dataframe(under[['player', 'team', 'shots', 'goals', 'xg', 'difference']].reset_index(drop=True))

            st.subheader('Player Search')
            query = st.text_input('Search player (exact name match)')
            if query:
                sel = player_stats[player_stats['player'].str.contains(query, case=False, na=False)]
                if sel.empty:
                    st.info('No matching player found')
                else:
                    st.write(sel[['player', 'team', 'shots', 'goals', 'xg', 'difference', 'conversion_rate']])
                    pname = sel.iloc[0]['player']
                    shots_df = df[df[player_col].str.contains(pname, na=False)]
                    if not shots_df.empty:
                        st.write('Shot distribution by distance')
                        st.bar_chart(shots_df['distance'].fillna(0).value_counts().sort_index())
                        if 'body_head' in df.columns or 'body_foot' in df.columns or 'body_other' in df.columns:
                            if 'body_part' in shots_df.columns:
                                st.write('By body part')
                                st.bar_chart(shots_df['body_part'].fillna('Unknown').value_counts())
                            else:
                                bcols = [c for c in ['body_head','body_foot','body_other'] if c in shots_df.columns]
                                if bcols:
                                    bp = shots_df[bcols].idxmax(axis=1)
                                    st.write('By body part (inferred)')
                                    st.bar_chart(bp.value_counts())
                        if 'shot_type' in shots_df.columns:
                            st.write('By shot type')
                            st.bar_chart(shots_df['shot_type'].fillna('Unknown').value_counts())

            st.subheader('Conversion Rate vs Avg xG per Shot')
            player_stats['avg_xg_per_shot'] = player_stats['xg'] / player_stats['shots'].replace(0, np.nan)
            conv_df = player_stats.dropna(subset=['conversion_rate','avg_xg_per_shot'])
            if not conv_df.empty:
                conv_chart = alt.Chart(conv_df).mark_circle(size=60).encode(
                    x=alt.X('avg_xg_per_shot:Q', title='Avg xG per Shot'),
                    y=alt.Y('conversion_rate:Q', title='Conversion Rate'),
                    tooltip=['player:N','team:N','shots:Q','goals:Q','avg_xg_per_shot:Q','conversion_rate:Q']
                )
                st.altair_chart(conv_chart.interactive(), use_container_width=True)
        else:
            st.info('No player stats available to plot')


if page == 'Team Analysis':
    st.header('Team Analysis')
    team_col = next((c for c in ['team', 'team_name', 'teamId'] if c in df.columns), None)
    league_col = next((c for c in ['league', 'competition', 'league_name'] if c in df.columns), None)

    pref = next((m for m in ['logistic', 'random_forest', 'xgboost', 'neural_network'] if m in models), None)
    if pref:
        mapper = models[pref]
        try:
            proba_all = mapper.predict_proba(df[feature_cols])[:, 1] if feature_cols else mapper.predict_proba(df)[:, 1]
        except Exception:
            try:
                proba_all = mapper.predict_proba(df)[:, 1]
            except Exception:
                proba_all = np.full(len(df), np.nan)
    else:
        proba_all = np.full(len(df), np.nan)

    df['_pred_xg'] = proba_all
    df['_is_goal'] = y

    if not team_col:
        st.info('No team column found in data — Team Analysis requires a team identifier column.')
    else:
        grp = df.groupby(team_col)
        shots = grp.size().rename('shots')
        xg_sum = grp['_pred_xg'].sum().rename('xg')
        goals = grp['_is_goal'].sum().rename('goals')
        team_stats = pd.concat([shots, goals, xg_sum], axis=1).fillna(0)
        team_stats['difference'] = team_stats['goals'] - team_stats['xg']
        team_stats = team_stats.reset_index().rename(columns={team_col: 'team'})

        st.subheader('Team xG vs Goals (Top 20 by Shots)')
        top = team_stats.sort_values('shots', ascending=False).head(20)
        if not top.empty:
            melt = top.melt(id_vars=['team'], value_vars=['xg','goals'], var_name='metric', value_name='value')
            chart = alt.Chart(melt).mark_bar().encode(
                x=alt.X('team:N', sort='-y', title='Team'),
                y=alt.Y('value:Q', title='Count'),
                color='metric:N',
                tooltip=['team:N','metric:N','value:Q']
            ).properties(width=800)
            st.altair_chart(chart.configure_axisX(labelAngle=-45), use_container_width=True)

        st.subheader('Team Selector')
        teams = team_stats['team'].tolist()
        sel_team = st.selectbox('Select team', options=[''] + teams)
        if sel_team:
            tdf = df[df[team_col] == sel_team]
            st.write('Summary')
            st.write(pd.DataFrame({
                'shots': [len(tdf)],
                'goals': [tdf['_is_goal'].sum()],
                'xg': [tdf['_pred_xg'].sum()],
                'conversion_rate': [tdf['_is_goal'].sum() / max(1, len(tdf))]
            }, index=[sel_team]))
            if 'season' in tdf.columns:
                ag = pd.DataFrame({
                    'xg': tdf.groupby('season')['_pred_xg'].sum(),
                    'goals': tdf.groupby('season')['_is_goal'].sum(),
                }).reset_index()
                line = alt.Chart(ag).transform_fold(['xg','goals'], as_=['metric','value']).mark_line(point=True).encode(x='season:N', y='value:Q', color='metric:N', tooltip=['season','metric','value'])
                st.altair_chart(line, use_container_width=True)
            elif 'date' in tdf.columns:
                try:
                    tdf['date_dt'] = pd.to_datetime(tdf['date'])
                    ag = pd.DataFrame({
                        'xg': tdf.resample('M', on='date_dt')['_pred_xg'].sum(),
                        'goals': tdf.resample('M', on='date_dt')['_is_goal'].sum(),
                    }).reset_index()
                    agm = ag.melt(id_vars=['date_dt'], value_vars=['xg','goals'], var_name='metric', value_name='value')
                    ch = alt.Chart(agm).mark_line(point=True).encode(x='date_dt:T', y='value:Q', color='metric:N', tooltip=['date_dt','metric','value'])
                    st.altair_chart(ch, use_container_width=True)
                except Exception:
                    st.info('Date parsing for team timeline failed')

        st.subheader('Shot Type Distribution by Team')
        if 'shot_type' in df.columns:
            topteams = top['team'].tolist() if not top.empty else team_stats['team'].tolist()[:10]
            dist = df[df[team_col].isin(topteams)].groupby([team_col,'shot_type']).size().rename('count').reset_index()
            dist_tot = dist.groupby(team_col)['count'].transform('sum')
            dist['pct'] = dist['count'] / dist_tot
            st.altair_chart(alt.Chart(dist).mark_bar().encode(x=alt.X('team:N', sort='-y'), y=alt.Y('pct:Q', title='Proportion'), color='shot_type:N', tooltip=['team','shot_type','pct']), use_container_width=True)
        else:
            st.info('No shot_type column available')