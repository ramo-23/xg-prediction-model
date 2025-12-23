"""Interactive Streamlit dashboard for xG visualizations and prediction.

Features:
- Load processed data and trained models from `results/metrics/`.
- Model comparison (ROC + calibration), feature importances, player/team xG aggregations.
- Simple prediction UI to enter shot features and get predicted xG.
"""
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve


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
st.title('xG Dashboard')
st.write('Interactive model comparison, feature importances, player/team xG, and prediction UI.')

# load data
proc_dir = repo_root / 'data' / 'processed'
files = sorted(proc_dir.glob('processed_shots_*.csv'), key=lambda p: p.stat().st_mtime) if proc_dir.exists() else []
if not files:
	st.error('No processed_shots_*.csv files found in data/processed/')
	st.stop()
data_path = files[-1]
df = pd.read_csv(data_path)
if 'outcome' not in df.columns:
	st.error('processed data missing outcome column')
	st.stop()
y = df['outcome'].astype(str).str.lower().eq('goal').astype(int)

# infer features
numeric_feats = [c for c in ['distance', 'minute_num'] if c in df.columns]
binary_feats = [c for c in ['body_head', 'body_foot', 'body_other', 'big_chance', 'half'] if c in df.columns]
cat_feats = [c for c in ['shot_type', 'assist_type'] if c in df.columns]
feature_cols = numeric_feats + binary_feats + cat_feats

# load models
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

st.sidebar.header('Options')
selected_model = st.sidebar.selectbox('Model for single-shot prediction', options=[''] + list(models.keys()))
show_roc = st.sidebar.checkbox('Show model comparison (ROC & calibration)', value=True)
show_imp = st.sidebar.checkbox('Show feature importances', value=True)
show_xg = st.sidebar.checkbox('Show player/team xG analysis', value=True)

fig_dir = repo_root / 'results' / 'metrics' / 'figures'
fig_dir.mkdir(parents=True, exist_ok=True)


if show_roc and models:
	st.header('Model comparison — ROC & Calibration')
	X_train_full, X_hold, y_train_full, y_hold = train_test_split(df[feature_cols] if feature_cols else df, y, test_size=0.4, stratify=y, random_state=42)
	X_val, X_test, y_val, y_test = train_test_split(X_hold, y_hold, test_size=0.5, stratify=y_hold, random_state=42)
	fig, axs = plt.subplots(1, 2, figsize=(14, 6))
	ax = axs[0]
	ax2 = axs[1]
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
	savep = fig_dir / 'streamlit_model_comparison.png'
	fig.savefig(savep, dpi=150, bbox_inches='tight')
	st.write('Saved comparison figure to', str(savep))


if show_imp and models:
	st.header('Feature importance (if available)')
	def unwrap_model(m):
		try:
			if hasattr(m, 'named_steps'):
				clf = m.named_steps.get('clf') or m.named_steps.get('classifier') or list(m.named_steps.values())[-1]
				m = clf
		except Exception:
			pass
		if hasattr(m, 'base_estimator'):
			return m.base_estimator
		if hasattr(m, 'estimator'):
			return m.estimator
		return m

	feat_names = feature_cols
	for name, mdl in models.items():
		try:
			base = unwrap_model(mdl)
			if hasattr(base, 'coef_'):
				coefs = np.ravel(base.coef_)
				if len(coefs) == len(feat_names):
					df_imp = pd.DataFrame({'feature': feat_names, 'importance': np.abs(coefs)}).sort_values('importance', ascending=False).head(20)
					st.subheader(name)
					st.dataframe(df_imp)
					fig = plt.figure(figsize=(6, 4))
					sns.barplot(data=df_imp, x='importance', y='feature')
					st.pyplot(fig)
			elif hasattr(base, 'feature_importances_'):
				fi = base.feature_importances_
				if len(fi) == len(feat_names):
					df_imp = pd.DataFrame({'feature': feat_names, 'importance': fi}).sort_values('importance', ascending=False).head(20)
					st.subheader(name)
					st.dataframe(df_imp)
					fig = plt.figure(figsize=(6, 4))
					sns.barplot(data=df_imp, x='importance', y='feature')
					st.pyplot(fig)
		except Exception:
			continue


if show_xg:
	st.header('Player / Team xG analysis')
	player_cols = [c for c in ['player', 'player_name', 'shooter', 'playerId'] if c in df.columns]
	team_cols = [c for c in ['team', 'team_name', 'teamId'] if c in df.columns]
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
	if player_cols:
		grp = df.groupby(player_cols[0]).agg(xg=('_pred_xg', 'sum'), goals=('_is_goal', 'sum'), attempts=('_is_goal', 'count'))
		grp = grp.sort_values('xg', ascending=False)
		st.subheader('Top xG by player')
		st.dataframe(grp.head(20))
		csvp = repo_root / 'results' / 'metrics' / 'top_xg_by_player.csv'
		grp.to_csv(csvp)
		st.write('Saved', str(csvp))
	else:
		st.info('No player column found in data')
	if team_cols:
		grt = df.groupby(team_cols[0]).agg(xg=('_pred_xg', 'sum'), goals=('_is_goal', 'sum'), attempts=('_is_goal', 'count'))
		grt = grt.sort_values('xg', ascending=False)
		st.subheader('Top xG by team')
		st.dataframe(grt.head(20))
		csvp = repo_root / 'results' / 'metrics' / 'top_xg_by_team.csv'
		grt.to_csv(csvp)
		st.write('Saved', str(csvp))
	else:
		st.info('No team column found in data')


st.header('Single-shot prediction')
if not feature_cols:
	st.warning('No feature columns inferred from processed data; cannot build prediction UI.')
else:
	with st.form('predict_form'):
		inputs = {}
		for f in numeric_feats:
			inputs[f] = st.number_input(f, value=float(df[f].median()) if f in df.columns else 0.0)
		for f in binary_feats:
			opts = [0, 1]
			default = int(df[f].mode().iloc[0]) if f in df.columns else 0
			inputs[f] = st.selectbox(f, options=opts, index=opts.index(default))
		for f in cat_feats:
			opts = sorted(df[f].dropna().unique().tolist()) if f in df.columns else ['missing']
			inputs[f] = st.selectbox(f, options=opts)
		chosen = st.selectbox('Model', options=list(models.keys()) if models else [])
		submitted = st.form_submit_button('Predict')
		if submitted:
			if not chosen:
				st.error('No model available for prediction')
			else:
				row = pd.DataFrame([inputs])
				mdl = models[chosen]
				try:
					p = mdl.predict_proba(row)[:, 1][0]
				except Exception:
					try:
						# fallback: try passing array or full dataframe
						p = mdl.predict_proba(row[feature_cols])[:, 1][0]
					except Exception as e:
						st.error('Prediction failed: %s' % e)
						p = None
				if p is not None:
					st.success(f'Predicted xG: {p:.3f}')

