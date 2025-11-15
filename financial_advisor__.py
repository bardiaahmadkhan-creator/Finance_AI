"""

Personal Finance Advisor — Production-Ready Streamlit App

"""
from __future__ import annotations
import io
import json
import math
import html
import streamlit.components.v1 as components
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
import numpy as np
import pandas as pd
import joblib, os
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
plt.style.use("seaborn-v0_8-whitegrid")

# --- Global Matplotlib Theme ---

mpl.rcParams.update({
	"axes.edgecolor": "#D3D3D3",
	"axes.linewidth": 1.2,
	"axes.facecolor": "#FAFAFA",
	"axes.titlesize": 14,
	"axes.titleweight": "bold",
	"axes.labelsize": 12,
	"axes.labelcolor": "#0A1F44",
	"grid.color": "#EAEAEA",
	"grid.linestyle": "-",
	"grid.alpha": 0.6,
	"xtick.color": "#0A1F44",
	"ytick.color": "#0A1F44",
	"font.family": "Segoe UI",
	"figure.facecolor": "#F7F9FC",
	"legend.frameon": False,
	"legend.fontsize": 11,
})
warnings.filterwarnings("ignore")

# --- Page Config ---

st.set_page_config(page_title="Personal Finance Advisor", layout="wide")

# --- Global CSS Theme ---

st.markdown("""
<style>
/* Global background */
.main {
	background-color:#F7F9FC;
}

/* Finance-style headers */
h1, h2, h3 {
	color:#0A1F44;
	font-family: 'Segoe UI', sans-serif;
}

/* Sidebar background */
[data-testid="stSidebar"] {
	background-color: #FFFFFF;
}

/* Card-like metric container */
.metric-card {
	padding: 20px;
	background-color: #FFFFFF;
	border-radius: 12px;
	box-shadow: 0 2px 10px rgba(0,0,0,0.07);
}

/* Cleaner buttons */
.stButton>button {
	background-color:#0A1F44;
	color:white;
	border-radius:8px;
	padding:0.6rem 1.2rem;
	font-size:16px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* --- Risk Tab Enhancements --- */
.risk-card {
	background-color: #FFFFFF;
	border-radius: 14px;
	padding: 20px;
	margin-bottom: 20px;
	box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.metric-header {
	background: linear-gradient(135deg, #0A1F44 0%, #193C6D 100%);
	color: white;
	border-radius: 12px;
	padding: 18px 25px;
	margin-bottom: 20px;
}
.metric-header h2 {
	margin: 0;
	color: #FFFFFF;
}
.metric-header p {
	margin: 0;
	color: #EAEAEA;
	font-size: 15px;
}
.subtle-title {
	color: #0A1F44;
	font-size: 17px;
	font-weight: bold;
	margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.debt-card {
	background-color: #FFFFFF;
	border-radius: 14px;
	padding: 18px 20px;
	margin-bottom: 20px;
	box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}
.debt-header {
	background: linear-gradient(135deg, #0A1F44 0%, #193C6D 100%);
	color: white;
	padding: 18px 25px;
	border-radius: 12px;
	margin-bottom: 20px;
}
.debt-header h2 {
	margin: 0;
	color: #FFFFFF;
}
.debt-header p {
	color: #EAEAEA;
	font-size: 15px;
	margin: 0;
}
.debt-table {
	border-radius: 12px;
	overflow: hidden;
	box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* --- Enhanced Debt Table Styling --- */
.debt-table-container {
	background-color: #FFFFFF;
	border-radius: 12px;
	padding: 10px 15px;
	margin-top: 10px;
	box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}

.debt-table-header {
	background: linear-gradient(135deg, #0A1F44 0%, #193C6D 100%);
	color: white;
	font-weight: 600;
	text-align: center;
	border-radius: 8px 8px 0 0;
	padding: 6px;
	margin-bottom: 5px;
}

.debt-table .stDataFrame {
	border: none !important;
	border-radius: 0 0 12px 12px;
}

/* --- AI Allocation Dashboard --- */
.ai-card {
	background: #FFFFFF;
	border-radius: 14px;
	box-shadow: 0 2px 10px rgba(0,0,0,0.08);
	padding: 18px 22px;
	margin-bottom: 20px;
}
.ai-header {
	font-weight: 600;
	font-size: 17px;
	color: #0A1F44;
}
.ai-metric {
	font-size: 28px;
	font-weight: bold;
	color: #193C6D;
}
.ai-subtext {
	font-size: 14px;
	color: #52616B;
}
.ai-grid {
	display: flex;
	justify-content: space-around;
	gap: 10px;
	margin-top: 15px;
}
.ai-box {
	flex: 1;
	text-align: center;
	padding: 12px;
	background-color: #F8FAFC;
	border-radius: 10px;
}
.ai-box h4 {
	margin: 0;
	color: #0A1F44;
	font-size: 15px;
}
.ai-box p {
	margin: 0;
	color: #2980B9;
	font-size: 18px;
	font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --- Custom Header ---

st.markdown("""
<div style="padding: 25px; background-color:#FFFFFF; border-radius:12px; margin-bottom:20px;
			box-shadow:0 2px 10px rgba(0,0,0,0.07);">
	<h1 style="margin-bottom:5px;">Personal Finance Advisor</h1>
	<p style="color:#52616B; font-size:18px;">
		Budget diagnostics • Debt strategies • Risk & Portfolio • Retirement Monte Carlo
	</p>
</div>
""", unsafe_allow_html=True)

# --- Utilities ---

@dataclass
class Debt:
	name: str
	balance: float
	apr: float
	min_payment: float


def _fmt_currency(x: float, prefix: str = "$", decimals: int = 2) -> str:
	try:
		return f"{prefix}{x:,.{decimals}f}"
	except Exception:
		return f"{prefix}{x}"

def _html_safe_currency(x: float) -> str:
	"""Format as currency and escape HTML-sensitive characters."""
	return html.escape(_fmt_currency(x))

# --- Core Finance Logic ---

def budget_analysis(
	monthly_income: float,
	expenses: Dict[str, float],
	essential_keys: List[str],
	irregular_expenses: Dict[str, float] = None
) -> Dict:
	total_expenses = max(0.0, sum(float(v) for v in expenses.values()))
 
# --- Add Irregular Expenses ---

	annual_irregular_total = 0.0
	if irregular_expenses:
		annual_irregular_total = sum(float(v) for v in irregular_expenses.values())
	monthly_irregular = annual_irregular_total / 12.0
	total_expenses += monthly_irregular

	income = max(0.0, float(monthly_income))
	savings = income - total_expenses
	savings_rate = (savings / income) if income > 0 else 0.0
	essentials_set = {e.lower() for e in essential_keys}
	essential = sum(float(v) for k, v in expenses.items() if k.lower() in essentials_set)
	essential = min(essential, total_expenses)
	discretionary = total_expenses - essential
	return {
		"income": income,
		"total_expenses": total_expenses,
		"savings": savings,
		"savings_rate": savings_rate,
		"essential_expenses": essential,
		"discretionary_expenses": discretionary,
		"irregular_monthly": monthly_irregular,
		"annual_irregular_total": annual_irregular_total,
	}


def _active_debts(debts: List[Debt]) -> List[Debt]:
	return [d for d in debts if d.balance > 0.01]

def calculate_minimums_this_month(debts: List[Debt]) -> float:
	"""Return sum of minimum payments for the current month, respecting remaining balances."""
	return float(sum(min(d.min_payment, d.balance) for d in debts if d.balance > 0.01))

def debt_payoff_schedule(
	debts_input: List[Debt],
	monthly_income: float,
	monthly_expenses: float,
	method: str = "avalanche",
	cap_months: int = 600,
) -> Tuple[pd.DataFrame, float, int]:

# --- Deep Copy ---

	debts = [Debt(d.name, float(d.balance), float(d.apr), float(d.min_payment)) for d in debts_input]
	records: List[Dict] = []
	month = 0
	total_interest = 0.0

	def order_key(d: Debt) -> float:
		return -d.apr if method == "avalanche" else d.balance

	while _active_debts(debts) and month < cap_months:
		month += 1
  
# --- Calculate Dynamic Extra Each Month ---

		min_this_month = calculate_minimums_this_month(debts)
		available_after_expenses = max(0.0, monthly_income - monthly_expenses)
		dynamic_extra = max(0.0, available_after_expenses - min_this_month)

# --- Total Pool Available For This Month ---

		total_budget = min_this_month + dynamic_extra
		pool = total_budget

# --- Accrue Interest ---

		for d in _active_debts(debts):
			interest = d.balance * (d.apr / 12.0)
			total_interest += interest
			d.balance = d.balance + interest

# --- Figure Out How Much The Contractual Minimums Would Be This Month ---

		actives = sorted(_active_debts(debts), key=order_key)
		mins = {d.name: min(d.min_payment, d.balance) for d in actives}

# --- Pay Minimums First ---

		for d in actives:
			if pool <= 0:
				break
			pay = min(mins[d.name], d.balance, pool)
			d.balance -= pay
			pool -= pay
			records.append({
				"month": month, "debt": d.name, "stage": "min",
				"payment": round(pay, 2), "interest": None,
				"balance": round(max(d.balance, 0.0), 2),
			})

		actives = sorted(_active_debts(debts), key=order_key)
		for d in actives:
			if pool <= 0:
				break
			extra_pay = min(pool, d.balance)
			d.balance -= extra_pay
			pool -= extra_pay
			records.append({
				"month": month, "debt": d.name, "stage": "extra",
				"payment": round(extra_pay, 2), "interest": None,
				"balance": round(max(d.balance, 0.0), 2),
			})

# --- Clean Up Small Negative Residuals ---

	for d in debts:
		if d.balance < 0.01:
			d.balance = 0.0
	df = pd.DataFrame(records)
	total_interest = round(total_interest, 2)
	return df, total_interest, month

def debt_payoff_ai(
	debts_input: List[Debt],
	monthly_income: float,
	monthly_expenses: float,
	ai_weights: Dict[str, float],
	cap_months: int = 600,
) -> Tuple[pd.DataFrame, float, int]:
	"""AI-driven payoff schedule that uses machine-learning-generated weights."""
	debts = [Debt(d.name, float(d.balance), float(d.apr), float(d.min_payment)) for d in debts_input]
	records = []
	month = 0
	total_interest = 0.0

	while _active_debts(debts) and month < cap_months:
		month += 1

# --- Calculate Available Pool ---

		min_this_month = calculate_minimums_this_month(debts)
		available_after_expenses = max(0.0, monthly_income - monthly_expenses)
		dynamic_extra = max(0.0, available_after_expenses - min_this_month)
		total_pool = min_this_month + dynamic_extra
		pool = total_pool

# --- Accrue Interest ---

		for d in _active_debts(debts):
			interest = d.balance * (d.apr / 12.0)
			total_interest += interest
			d.balance += interest

# --- Pay Minimums First ---
  
		for d in debts:
			if pool <= 0:
				break
			pay = min(d.min_payment, d.balance, pool)
			d.balance -= pay
			pool -= pay
			records.append({
				"month": month,
				"debt": d.name,
				"stage": "min",
				"payment": round(pay, 2),
				"balance": round(d.balance, 2),
			})

# --- Use AI Weights For Extra Payments ---

		if dynamic_extra > 0 and pool > 0:
			weight_sum = sum(ai_weights.get(d.name, 0) for d in debts if d.balance > 0)
			if weight_sum == 0:
				weight_sum = 1.0
			for d in debts:
				if pool <= 0 or d.balance <= 0:
					continue
				share = ai_weights.get(d.name, 0) / weight_sum
				extra = min(pool * share, d.balance)
				d.balance -= extra
				pool -= extra
				records.append({
					"month": month,
					"debt": d.name,
					"stage": "extra",
					"payment": round(extra, 2),
					"balance": round(d.balance, 2),
				})

# --- Cleanup ---

		for d in debts:
			if d.balance < 0.01:
				d.balance = 0.0

	df = pd.DataFrame(records)
	total_interest = round(total_interest, 2)
	return df, total_interest, month

# ======================================================
# AI-Driven Debt Optimization
# ======================================================

@st.cache_resource(show_spinner=False)
def train_dynamic_debt_model(n_samples: int = 6000, random_state: int = 42):
	"""
	Improved trainer:
	  - Better targets via Dirichlet search + coordinate refinement
	  - Group-aware split (by portfolio)
	  - HistGradientBoostingRegressor with early stopping
	  - Validation metric computed AFTER per-portfolio normalization
	"""
 
# --- Try To Load Pre-Trained Model If It Exists ---

	model_path = os.path.join("models", "trained_debt_ai_model.pkl")
	scaler_path = os.path.join("models", "trained_debt_ai_scaler.pkl")
	if os.path.exists(model_path) and os.path.exists(scaler_path):
		st.sidebar.info("✅ Loaded pre-trained AI model (skipping retraining).")
		return {"model": joblib.load(model_path), "scaler": joblib.load(scaler_path)}

	rng = np.random.default_rng(random_state)
	results_X, results_y, results_group = [], [], []

	group_id = 0

	def _dirichlet_weights(k: int):
     
# --- Smooth Dirichlet ---

		alpha = np.ones(k) * 0.8  # allows sharper (less even) weights
		w = rng.dirichlet(alpha)
		return w

	def _coordinate_refine(w0, eval_fn, steps=6, step_size=0.15):
		w = w0.copy()
		best = eval_fn(w)
		for _ in range(steps):
			for j in range(len(w)):
				for sgn in (+1, -1):
					w_try = w.copy()
					w_try[j] = max(0.0, w_try[j] + sgn * step_size)
					s = w_try.sum()
					if s <= 0:
						continue
					w_try /= s
					val = eval_fn(w_try)
					if val < best:
						best, w = val, w_try
		return w, best

	for _ in range(n_samples):
     
# --- Random portfolio (2–5 debts) ---

		n_debts = int(rng.integers(2, 6))
		debts = [
			Debt(
				f"Debt {i+1}",
				float(rng.uniform(1000, 25000)),
				float(rng.uniform(0.01, 0.30)),
				float(rng.uniform(40, 350)),
			)
			for i in range(n_debts)
		]

# --- User Context ---

		monthly_income = float(rng.uniform(2500, 9000))
		monthly_expenses = float(rng.uniform(1200, monthly_income * 0.95))
		volatility = float(rng.uniform(0.05, 0.3))

# --- Income Shocks ---

		if rng.random() < 0.15:
			monthly_income *= float(rng.uniform(0.6, 0.9))
		if rng.random() < 0.10:
			debts[rng.integers(0, n_debts)].balance *= float(rng.uniform(0.7, 0.9))

		total_balance = sum(d.balance for d in debts)
		max_apr = max(d.apr for d in debts) if debts else 0.0
		debt_to_income = total_balance / max(monthly_income, 1.0)

		def score_from_weights(w_vec):
			ai_weights = {d.name: float(w) for d, w in zip(debts, w_vec)}
			_, total_interest, months = debt_payoff_ai(
				debts, monthly_income, monthly_expenses, ai_weights, cap_months=360
			)
			alpha = float(rng.uniform(0.35, 0.65))
			return alpha * (total_interest / 1000.0) + (1.0 - alpha) * months

		best_w = _dirichlet_weights(n_debts)
		best_score = score_from_weights(best_w)

# --- Wider Random Exploration ---

		for _try in range(280):
			w0 = _dirichlet_weights(n_debts)
			sc_raw = score_from_weights(w0)
			sc = sc_raw - 0.1 * np.std(w0)     # reward concentrated (non-even) weights
			if sc < best_score:
				best_w, best_score = w0, sc


# --- Local Coordinate Refinement ---

		best_w, best_score_raw = _coordinate_refine(best_w, score_from_weights, steps=8, step_size=0.12)
		best_score = best_score_raw - 0.1 * np.std(best_w)


# --- Feature Engineering ---

		monthly_free_cash = (monthly_income - monthly_expenses)
		interest_proxy = sum(d.balance * d.apr for d in debts)

		for rank, d in enumerate(sorted(debts, key=lambda x: -x.apr), start=1):
			rel_apr = d.apr / max_apr if max_apr > 0 else 0.0
			bal_share = d.balance / max(total_balance, 1.0)
			minpay_ratio = d.min_payment / max(d.balance, 1.0)
			pay_burden = d.min_payment / max(monthly_income, 1.0)
			amort_ratio = (d.min_payment * 12.0) / max(d.balance, 1.0)
			liquidity = monthly_free_cash / max(monthly_income, 1.0)
			int_density = (interest_proxy / (total_balance + 1e-9)) / 100.0
			rel_rank = rank / n_debts

			feat = [
				d.apr,
				rel_apr,
				d.balance,
				bal_share,
				minpay_ratio,
				pay_burden,
				amort_ratio,
				liquidity,
				volatility,
				debt_to_income,
				int_density,
				n_debts,
				rel_rank,
				d.apr * bal_share,                 
				d.balance * minpay_ratio,          
				math.log1p(d.balance),              
				math.log1p(total_balance),       
				monthly_free_cash,                  
				d.apr ** 2,                
				(d.apr * bal_share) * 10.0  

			]
			results_X.append(feat)
			results_y.append(best_w[rank - 1])     
			results_group.append(group_id)
		group_id += 1

	X = np.array(results_X, dtype=float)
	y = np.array(results_y, dtype=float)
	groups = np.array(results_group)
	scaler = StandardScaler()
	X = scaler.fit_transform(X)

# --- Group-Aware Split ---

	gss = GroupShuffleSplit(test_size=0.15, n_splits=1, random_state=random_state)
	(train_idx, val_idx) = next(gss.split(X, y, groups))

	X_train, X_val = X[train_idx], X[val_idx]
	y_train, y_val = y[train_idx], y[val_idx]
	g_train, g_val = groups[train_idx], groups[val_idx]

	model = HistGradientBoostingRegressor(
		loss="absolute_error",      
		learning_rate=0.06,
		max_depth=None,
		max_iter=900,
		l2_regularization=5e-4,
		max_bins=255,
		early_stopping=True,
		validation_fraction=0.1,
		random_state=random_state,
	)

	model.fit(X_train, y_train)

# ---- Group-Normalized Validation MSE ----
 
	def _normalize_by_group(arr, grp):
		out = arr.copy()
		for g in np.unique(grp):
			idx = (grp == g)
			s = out[idx].sum()
			if s > 0:
				out[idx] = out[idx] / s
		return out

	val_raw = np.clip(model.predict(X_val), 0, None)
	y_val_n = _normalize_by_group(y_val, g_val)
	pred_val_n = _normalize_by_group(val_raw, g_val)
	val_mse = float(np.mean((y_val_n - pred_val_n) ** 2))

# --- Save Trained Model and Scaler For Future Use ---

	os.makedirs("models", exist_ok=True)
	model_path = os.path.join("models", "trained_debt_ai_model.pkl")
	scaler_path = os.path.join("models", "trained_debt_ai_scaler.pkl")

	joblib.dump(model, model_path)
	joblib.dump(scaler, scaler_path)

	return {"model": model, "scaler": scaler}


def ai_debt_optimizer(debts: List[Debt], income_vol: float = 0.1) -> Dict[str, float]:
	"""Predict optimal extra-payment allocation using the trained advanced model."""
	if not debts:
		return {}

	trained = train_dynamic_debt_model()
	model, scaler = trained["model"], trained["scaler"]

# --- Recreate The Same Context Vars Used In Training ---
 
	monthly_income = sum(d.min_payment for d in debts) * 3.5
	total_balance = sum(d.balance for d in debts)
	max_apr = max(d.apr for d in debts) if debts else 0.0
	n_debts = len(debts)
	debt_to_income = total_balance / max(monthly_income, 1.0)
	interest_proxy = sum(d.balance * d.apr for d in debts)
	monthly_min_total = sum(d.min_payment for d in debts)
	monthly_free_cash = (monthly_income - monthly_min_total)

	features = []
 
	# Sort by APR (descending) to match the exact training order
 
	for rank, d in enumerate(sorted(debts, key=lambda x: -x.apr), start=1):
		rel_apr = d.apr / max_apr if max_apr > 0 else 0.0
		bal_share = d.balance / max(total_balance, 1.0)
		minpay_ratio = d.min_payment / max(d.balance, 1.0)
		pay_burden = d.min_payment / max(monthly_income, 1.0)
		amort_ratio = (d.min_payment * 12.0) / max(d.balance, 1.0)
		liquidity = monthly_free_cash / max(monthly_income, 1.0)
		int_density = (interest_proxy / (total_balance + 1e-9)) / 100.0
		rel_rank = rank / n_debts
  
		feat = [
			d.apr,                      
			rel_apr,                   
			d.balance,                 
			bal_share,                 
			minpay_ratio,              
			pay_burden,                
			amort_ratio,               
			liquidity,                 
			income_vol,                
			debt_to_income,            
			int_density,               
			n_debts,                   
			rel_rank,                  
			d.apr * bal_share,         
			d.balance * minpay_ratio,  
			math.log1p(d.balance),     
			math.log1p(total_balance), 
			monthly_free_cash,         
			d.apr ** 2,                    
			(d.apr * bal_share) * 10.0   
		]
		features.append(feat)
	X = scaler.transform(np.array(features, dtype=float))
	preds = np.clip(model.predict(X), 0, None)
	if preds.sum() == 0:
		preds += 1e-6
		apr_bias = np.array([d.apr for d in sorted(debts, key=lambda x: -x.apr)], dtype=float)
		preds = 0.8 * preds + 0.2 * apr_bias
	weights = preds / preds.sum()

	return {d.name: float(w) for d, w in zip(sorted(debts, key=lambda x: -x.apr), weights)}


# --- Risk Profiling --- 

@st.cache_resource(show_spinner=False)
def build_risk_model(k: int = 3):
	"""Builds a cached KMeans model on synthetic but correlated population."""
	rng = np.random.default_rng(42)
	N = 1200
	ages = rng.integers(20, 70, size=N)
	log_income = rng.normal(10.8, 0.6, size=N)
	incomes = np.clip(np.exp(log_income), 15000, 450000)
	base_srate = np.clip(0.05 + 0.000003 * (incomes - incomes.mean()), 0.01, 0.6)
	savings_rate = np.clip(base_srate + rng.normal(0, 0.05, size=N), 0.01, 0.7)
	debt_ratio = np.clip(0.35 - 0.0000015 * (incomes - incomes.mean()) + rng.normal(0, 0.08, N), 0.0, 0.95)
	experience = np.clip(((ages - 20) / 5.0) + rng.normal(0, 1.0, N), 0, 10)

	data = pd.DataFrame(
		{
			"age": ages,
			"income": incomes,
			"savings_rate": savings_rate,
			"debt_ratio": debt_ratio,
			"investment_experience": experience,
		}
	)
	scaler = StandardScaler()
	X = scaler.fit_transform(data[["age", "income", "savings_rate", "debt_ratio", "investment_experience"]])
	kmeans = KMeans(n_clusters=k, random_state=1, n_init=10).fit(X)

# Label clusters by risk appetite score

	centroids = scaler.inverse_transform(kmeans.cluster_centers_)
	centroid_scores = []
	for c in centroids:
		age, income, srate, dratio, xp = c
		score = (srate * 2.2) + (xp * 0.25) - (dratio * 1.3) + (0.000002 * income)
		centroid_scores.append(score)
	order = np.argsort(centroid_scores)
	labels = ["Conservative", "Balanced", "Aggressive"][:k]
	label_map = {cluster: labels[i] for i, cluster in enumerate(order)}

	return {
		"scaler": scaler,
		"kmeans": kmeans,
		"label_map": label_map,
	}


def risk_profile(user_features: Dict, model) -> Tuple[str, int]:
	u = np.array(
		[
			[
				float(user_features["age"]),
				float(user_features["income"]),
				float(user_features["savings_rate"]),
				float(user_features["debt_ratio"]),
				float(user_features["investment_experience"]),
			]
		]
	)
	scaler, kmeans, label_map = model["scaler"], model["kmeans"], model["label_map"]
	u_scaled = scaler.transform(u)
	cluster = int(kmeans.predict(u_scaled)[0])
	label = label_map.get(cluster, "Balanced")

# Rule overrides for extreme cases

	dr, sr, xp, age = (
		float(user_features["debt_ratio"]),
		float(user_features["savings_rate"]),
		float(user_features["investment_experience"]),
		float(user_features["age"]),
	)
	if dr > 0.6 and sr < 0.1:
		label = "Conservative"
	elif sr > 0.25 and xp >= 6 and dr < 0.3:
		label = "Aggressive"
	elif age < 25 and dr > 0.4 and sr < 0.15:
		label = "Conservative"

	return label, cluster

def explain_risk_profile(label: str, user_features: Dict) -> str:
	"""Return a natural-language explanation for the assigned risk profile."""
	age = user_features["age"]
	income = user_features["income"]
	srate = user_features["savings_rate"]
	dr = user_features["debt_ratio"]
	xp = user_features["investment_experience"]

	reasons = []

# --- Savings rate ---

	if srate < 0.10:
		reasons.append(f"your savings rate is low ({srate*100:.1f}%), which limits your risk capacity")
	elif srate < 0.20:
		reasons.append(f"your savings rate is moderate ({srate*100:.1f}%)")
	else:
		reasons.append(f"your strong savings rate ({srate*100:.1f}%) supports taking more risk")

# --- Debt ratio ---

	if dr > 0.50:
		reasons.append(f"your debt level is high relative to income ({dr*100:.1f}%)")
	elif dr > 0.25:
		reasons.append(f"your debt load is manageable ({dr*100:.1f}%)")
	else:
		reasons.append(f"your low debt ratio ({dr*100:.1f}%) gives you flexibility")

# --- Age effect ---

	if age < 25:
		reasons.append("your young age allows for long-term growth but cash flow matters")
	elif age < 45:
		reasons.append("your age supports a growth-focused strategy")
	else:
		reasons.append("your age shifts focus toward stability and capital preservation")

# --- Experience effect ---

	if xp <= 2:
		reasons.append("you have limited investing experience")
	elif xp <= 5:
		reasons.append("you have moderate investment experience")
	else:
		reasons.append("your strong investment experience supports handling market volatility")

# --- Assemble natural language summary ---

	base = f"You're classified as {label} because "
	explanation = base + ", and ".join(reasons) + "."

	return explanation

def risk_to_allocation(risk_label: str) -> Dict[str, float]:
	templates = {
		"Conservative": {"cash": 0.10, "bonds": 0.60, "equities": 0.25, "alternatives": 0.05},
		"Balanced": {"cash": 0.05, "bonds": 0.40, "equities": 0.50, "alternatives": 0.05},
		"Aggressive": {"cash": 0.02, "bonds": 0.20, "equities": 0.72, "alternatives": 0.06},
	}
	return templates.get(risk_label, templates["Balanced"]) 


# Portfolio Optimization

def sharpe_optimize(expected_returns: np.ndarray, cov: np.ndarray, risk_free: float = 0.02):
	n = len(expected_returns)

	def neg_sharpe(w):
		ret = float(w @ expected_returns)
		vol = math.sqrt(float(w @ cov @ w))
		if vol <= 1e-12:
			return 1e6
		return -(ret - risk_free) / vol

	bounds = [(0.0, 1.0)] * n
	cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

	try:
		from scipy.optimize import minimize

		res = minimize(neg_sharpe, np.ones(n) / n, bounds=bounds, constraints=cons, method="SLSQP")
		if res.success and np.isfinite(res.fun):
			w = np.clip(res.x, 0, 1)
			w = w / w.sum()
			ret = float(w @ expected_returns)
			vol = math.sqrt(float(w @ cov @ w))
			sharpe = (ret - risk_free) / vol if vol > 0 else -1e9
			return {"weights": w, "return": ret, "vol": vol, "sharpe": sharpe, "solver": "scipy"}
	except Exception:
		pass

# Fallback random search

	rng = np.random.default_rng(7)
	best = None
	for _ in range(40000):
		w = rng.random(n)
		w /= w.sum()
		ret = float(w @ expected_returns)
		vol = math.sqrt(float(w @ cov @ w))
		if vol <= 1e-12:
			continue
		sharpe = (ret - risk_free) / vol
		if (best is None) or (sharpe > best["sharpe"]):
			best = {"weights": w, "return": ret, "vol": vol, "sharpe": sharpe, "solver": "random"}
	return best


# Monte Carlo (GBM)

def monte_carlo_gbm(
	current_savings: float,
	annual_contribution: float,
	years: int,
	n_sims: int = 5000,
	mu: float = 0.06,
	sigma: float = 0.12,
	inflation: float = 0.02,
) -> Tuple[np.ndarray, Dict]:
	"""Monte Carlo using Geometric Brownian Motion; returns in nominal and real terms."""
	rng = np.random.default_rng(42)
	dt = 1.0
	real_factor = (1 + inflation) ** years

	results = np.zeros(n_sims)
	for i in range(n_sims):
		value = float(current_savings)
		for _ in range(years):
			z = rng.normal(0, 1)
			growth = math.exp((mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z)
			value = value * growth + annual_contribution
		results[i] = value

	summary = {
		"median_nominal": float(np.median(results)),
		"p10_nominal": float(np.percentile(results, 10)),
		"p90_nominal": float(np.percentile(results, 90)),
		"mean_nominal": float(np.mean(results)),
		"median_real": float(np.median(results) / real_factor),
		"p10_real": float(np.percentile(results, 10) / real_factor),
		"p90_real": float(np.percentile(results, 90) / real_factor),
	}
	return results, summary


# Savings & Emergency

def emergency_fund_recommendation(monthly_expenses: float, months: int = 6) -> float:
	return float(monthly_expenses) * int(months)


def months_to_goal(
	goal_amount: float,
	current_savings: float,
	monthly_saving: float,
	annual_return: float = 0.04
) -> Tuple[int, float]:
	"""
	Overflow-safe solver for months to reach a savings goal with monthly contributions and compound growth.
	Uses a closed-form (log) solution when possible; otherwise falls back to a capped, log-space search.
	Returns (months_needed, projected_balance_at_that_time).
	"""
	PV = float(current_savings)
	FV = float(goal_amount)
	C  = float(monthly_saving)
	r  = max(0.0, float(annual_return)) / 12.0  # monthly

	# Early exits
	if PV >= FV:
		return 0, PV
	if C <= 0 and r <= 0:
		return 10_000, PV
	if r == 0.0:
		# Pure arithmetic progression
		if C <= 0:
			return 10_000, PV
		n = max(0, math.ceil((FV - PV) / max(C, 1e-12)))
		return int(n), float(PV + C * n)

	if C < 0:
		return 10_000, PV

	denom = PV + (C / r)
	num   = FV + (C / r)

	if denom > 0 and num > 0 and num >= denom:
		A = num / denom
		if A <= 1.0:
			return 0, PV
		n_real = math.log(A) / math.log(1.0 + r)
		n = int(math.ceil(max(0.0, n_real)))
		return n, float(FV)

	def f_of_n(n: float) -> float:
		t = n * math.log1p(r) 
		if t > 700.0:
			return 1.0
		A = math.exp(t)
		return PV * A + C * ((A - 1.0) / r) - FV

	low, high = 0.0, 2400.0 
	if f_of_n(high) < 0:
		return 10_000, PV  # treat as practically unreachable under given parameters

	for _ in range(80):  # enough for sub-month precision
		mid = 0.5 * (low + high)
		if f_of_n(mid) >= 0:
			high = mid
		else:
			low = mid
	n = int(math.ceil(high))

	# Avoid overflow computing the balance
	return n, float(FV)

# Sidebar Inputs

@st.cache_resource(show_spinner=False)
def train_financial_health_model(n_samples: int = 5000, random_state: int = 42):
	"""Train a synthetic ML model to predict overall financial health (0–100) safely."""
	rng = np.random.default_rng(random_state)

	# --- Generate synthetic dataset ---
	ages = rng.integers(18, 75, n_samples)
	incomes = rng.normal(55000, 25000, n_samples).clip(15000, 250000)
	expenses = incomes * rng.uniform(0.4, 0.95, n_samples)
	savings_rate = np.clip((incomes - expenses) / np.maximum(incomes, 1), -1, 1)
	debts = rng.normal(15000, 12000, n_samples).clip(0, 100000)
	debt_ratio = np.clip(debts / np.maximum(incomes, 1), 0, 5.0)  # cap at 5× income
	credit_utilization = rng.uniform(0, 1, n_samples)
	investment_xp = rng.integers(0, 10, n_samples)
	risk_score = rng.uniform(0, 1, n_samples)
	emergency_months = rng.uniform(0, 12, n_samples)

	# --- Build features matrix ---
	X = np.column_stack([
		ages, incomes, expenses, savings_rate, debts,
		debt_ratio, credit_utilization, investment_xp,
		risk_score, emergency_months
	])

	flexibility = (
		1.15 if np.mean(ages) < 30 or np.mean(incomes) < 50000 else
		0.9 if np.mean(incomes) > 120000 or np.mean(investment_xp) > 5 else
		1.0
	)
# --- Income-driven scoring setup

	income_norm = np.clip((incomes - np.percentile(incomes, 10)) /
						  max(np.percentile(incomes, 90) - np.percentile(incomes, 10), 1e-6), 0, 1)
	income_component = 45 * (income_norm ** 1.0)

# --- Behavior-driven bonuses ---

	saving_bonus = np.clip(((np.maximum(savings_rate, 0) * 100) ** 1.05) / 1.0, 0, 60)
	emergency_bonus = np.clip(((emergency_months / 6) * 25), 0, 30)

# --- Debt-free and spending dynamics ---

	debt_penalty = np.clip((debt_ratio ** 0.8) * 10, 0, 30)
	debt_free_bonus = np.where(debt_ratio < 0.05, 10, 0)
	overspend_balance = (incomes - expenses) / np.maximum(incomes, 1)
	overspend_bonus = np.clip(overspend_balance * 30, 0, 20)

# --- Composite financial health ---

	y = (
		income_component
		+ saving_bonus
		+ emergency_bonus * 0.9
		+ overspend_bonus
		+ debt_free_bonus
		+ 4.0 * risk_score
		+ 3.0 * (investment_xp / 10)
		- 0.8 * debt_penalty
		- 2.0 * credit_utilization
		+ rng.normal(0, 2, n_samples)
	)

# Inject a bottom tail so the model learns to assign truly low scores
	bad_idx = rng.choice(n_samples, size=int(n_samples * 0.20), replace=False)
	y[bad_idx] -= rng.uniform(10, 40, size=bad_idx.shape[0])
	y = np.clip(y, 0, 100)

# --- Defensive normalization & strict clipping ---
	X = np.nan_to_num(X, nan=0.0, posinf=1e4, neginf=0.0)
	X = np.clip(X, -1e4, 1e4)

	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	# --- Safer ML model ---
	model = HistGradientBoostingRegressor(
		loss="absolute_error",        # robust to outliers
		learning_rate=0.05,
		max_iter=400,                 # fewer iterations to reduce internal overflow
		max_depth=8,
		l2_regularization=1e-3,
		random_state=random_state,
	)

	model.fit(X_scaled, y)
	return {"model": model, "scaler": scaler}


def predict_financial_health(user_data: Dict) -> float:
	"""Predicts user’s financial health score (0–100)."""
	model_info = train_financial_health_model()
	model, scaler = model_info["model"], model_info["scaler"]

	X_user = np.array([[
		user_data["age"],
		user_data["income"],
		user_data["expenses"],
		user_data["savings_rate"],
		user_data["debt_balance"],
		user_data["debt_balance"] / max(user_data["income"], 1),
		user_data["credit_utilization"],
		user_data["investment_experience"],
		user_data["risk_score"],
		user_data["emergency_fund_months"],
	]])

	X_user_scaled = scaler.transform(X_user)
	pred = float(np.clip(model.predict(X_user_scaled)[0], 0, 100))
	return round(pred, 2)


with st.sidebar:
	st.header("User profile")
	age = st.number_input("Age", min_value=16, max_value=100, value=19, step=1)
	annual_income = st.number_input("Annual income (CAD)", min_value=0.0, value=42000.0, step=1000.0)
	monthly_income = annual_income / 12.0
	st.write(f"Monthly income: **{_fmt_currency(monthly_income)}**")

	st.subheader("Monthly expenses")
	expense_inputs = {
		"Rent/Housing": st.number_input("Rent / Housing", value=800.0),
		"Groceries": st.number_input("Groceries", value=250.0),
		"Transportation": st.number_input("Transportation", value=100.0),
		"Utilities": st.number_input("Utilities", value=80.0),
		"Insurance": st.number_input("Insurance", value=0.0),
		"Phone": st.number_input("Phone", value=40.0),
		"Entertainment": st.number_input("Entertainment", value=150.0),
		"Other": st.number_input("Other", value=0.0),
	}
	st.subheader("Irregular expenses (non-monthly)")
	irregular_expenses = {
	"Car insurance (annual)": st.number_input("Car insurance (annual)", value=0.0),
	"Tuition (annual)": st.number_input("Tuition (annual)", value=0.0),
	"Travel/Vacations (annual)": st.number_input("Travel / Vacations (annual)", value=0.0),
	"Gifts / Holidays (annual)": st.number_input("Gifts / Holidays (annual)", value=0.0),
	"Subscriptions (yearly)": st.number_input("Subscriptions (yearly)", value=0.0),
	}

	default_essentials = ["Rent/Housing", "Groceries", "Transportation", "Utilities", "Insurance"]
	essential_keys = st.multiselect(
		"Which of these are essential for your situation?",
		options=list(expense_inputs.keys()),
		default=default_essentials,
	)

	st.subheader("Savings & Debts")
	current_savings = st.number_input("Current savings (CAD)", value=5000.0)
	monthly_saving = st.number_input("Monthly amount available for saving/debt (CAD)", value=500.0)

	st.markdown("**Debts (up to 4)**")
	d1 = Debt("Debt 1", st.number_input("Debt 1 balance", value=2500.0), st.number_input("Debt 1 APR (decimal)", value=0.1999), st.number_input("Debt 1 min payment", value=75.0))
	d2 = Debt("Debt 2", st.number_input("Debt 2 balance", value=8000.0), st.number_input("Debt 2 APR (decimal)", value=0.045), st.number_input("Debt 2 min payment", value=120.0))
	d3 = Debt("Debt 3", st.number_input("Debt 3 balance (optional)", value=0.0), st.number_input("Debt 3 APR (decimal)", value=0.0), st.number_input("Debt 3 min payment", value=0.0))
	d4 = Debt("Debt 4", st.number_input("Debt 4 balance (optional)", value=0.0), st.number_input("Debt 4 APR (decimal)", value=0.0), st.number_input("Debt 4 min payment", value=0.0))

	st.subheader("Investing & Retirement")
	experience = st.slider("Investment experience (0 = none → 10 = expert)", 0, 10, 2)
	retirement_years = st.number_input("Years to retirement", min_value=1, max_value=60, value=40)
	mc_sims = st.number_input("Monte Carlo simulations", min_value=1000, max_value=20000, value=5000, step=1000)
	exp_return = st.number_input("Expected annual return (μ)", min_value=0.0, max_value=0.30, value=0.06, step=0.005)
	exp_vol = st.number_input("Annual volatility (σ)", min_value=0.01, max_value=0.80, value=0.12, step=0.01)
	inflation = st.number_input("Inflation (for real $)", min_value=0.0, max_value=0.10, value=0.02, step=0.005)

	st.markdown("---")
	run_button = st.button("Run analysis")

# Prepare debts

all_debts = [d for d in [d1, d2, d3, d4] if d.balance > 0.0 and d.apr >= 0.0 and d.min_payment >= 0.0]

# Main Tabs

tab_budget, tab_debt, tab_portfolio, tab_retirement, tab_health = st.tabs([
	"Budget & Savings",
	"Debt Payoff",
	"Risk & Portfolio",
	"Retirement & Reports",
	"Financial Health",
])

if run_button:
	t0 = time.time()

# Compute shared data once

	budget_for_savings = budget_analysis(
		monthly_income,
		expense_inputs,
		essential_keys,
		irregular_expenses
	)
	total_debt_balance = float(sum(d.balance for d in all_debts)) if all_debts else 0.0
	model = build_risk_model(k=3)

# Reusable dynamic metrics

	dyn_income = budget_for_savings["income"]
	dyn_expenses = budget_for_savings["total_expenses"]
	dyn_savings = budget_for_savings["savings"]
	dyn_rate = budget_for_savings["savings_rate"] * 100

# Budget & Savings

	with tab_budget:
		st.subheader("Budget analysis")
		budget = budget_for_savings
		c1, c2, c3 = st.columns(3)
		c1.metric("Monthly income", _fmt_currency(budget["income"]))
		c2.metric("Monthly expenses", _fmt_currency(budget["total_expenses"]))
		c3.metric("Monthly savings", _fmt_currency(budget["savings"]), delta=f"Rate: {budget['savings_rate']*100:.1f}%")
		st.info(f"Irregular expenses add {_fmt_currency(budget['irregular_monthly'])} per month.")


		if budget["savings"] < 0:
			st.error("Warning: You are spending more than you earn this month.")
		elif budget["savings_rate"] < 0.10:
			st.warning("Savings rate below 10%. Consider trimming discretionary spending.")
		else:
			st.success("Healthy savings rate. Nice work!")

# --- Cleaned Budget Expense Pie Chart ---

		fig, ax = plt.subplots(figsize=(5.8, 5.8))
		labels = list(expense_inputs.keys())
		sizes = [expense_inputs[k] for k in labels]

		if sum(sizes) > 0:
			colors = plt.cm.Paired(np.linspace(0.1, 0.9, len(labels)))
			wedges, texts, autotexts = ax.pie(
				sizes,
				labels=None, 
				autopct=lambda p: f"{p:.0f}%" if p > 4 else "",
				startangle=140,
				colors=colors,
				textprops={"color": "white", "fontsize": 10},
				wedgeprops={"edgecolor": "white", "linewidth": 1.2}
			)
			plt.setp(autotexts, size=10, weight="bold")
			ax.set_title("Expense Breakdown", fontsize=15, weight="bold", color="#0A1F44")
			ax.legend(
				labels,
				title="Expense Categories",
				loc="upper center",
				bbox_to_anchor=(0.5, -0.08),
				ncol=3,
				fontsize=10,
				title_fontsize=11,
				frameon=False
			)

			st.pyplot(fig)
		else:
			st.info("No expenses entered to visualize.")

# Savings goal example to $20k

		st.markdown("\n**Savings goal example**")
		months_needed, projected = months_to_goal(20000.0, current_savings, budget["savings"])
		st.markdown(
			f"Months to reach \\$20,000: **{months_needed}** "
			f"(Projected balance at that time: {_fmt_currency(projected).replace('$', '\\$')})"
		)

		efund = emergency_fund_recommendation(budget_for_savings["total_expenses"], months=6)
		st.write(f"Recommended emergency fund (6 months): **{_fmt_currency(efund)}**")

# Debt Payoff 

	with tab_debt:
     
	# --- Header ---
 
		st.markdown("""
		<div class="debt-header">
			<h2>Debt Payoff & AI Optimization</h2>
			<p>Compare Avalanche, Snowball, and AI-driven payoff strategies in a clean visual dashboard.</p>
		</div>
		""", unsafe_allow_html=True)

		if not all_debts:
			st.info("No debts entered — add debts in the sidebar to simulate.")
		else:
      
		# --- Core Schedules ---
  
			sched_av, interest_av, months_av = debt_payoff_schedule(
				all_debts, monthly_income, budget_for_savings["total_expenses"], "avalanche"
			)
			sched_sn, interest_sn, months_sn = debt_payoff_schedule(
				all_debts, monthly_income, budget_for_savings["total_expenses"], "snowball"
			)
			ai_alloc = ai_debt_optimizer(all_debts)
			sched_ai, interest_ai, months_ai = debt_payoff_ai(
				all_debts, monthly_income, budget_for_savings["total_expenses"], ai_alloc
			)

		# --- Top Metrics ---
  
			c1, c2, c3 = st.columns(3)
			c1.markdown(f"""
			<div class="debt-card" style="text-align:center;">
				<h3 style="color:#0A1F44;">Avalanche</h3>
				<h2 style="color:#16A085;">{_fmt_currency(interest_av)}</h2>
				<p style="color:#52616B;">Total interest • {months_av} months</p>
			</div>
			""", unsafe_allow_html=True)
			c2.markdown(f"""
			<div class="debt-card" style="text-align:center;">
				<h3 style="color:#0A1F44;">Snowball</h3>
				<h2 style="color:#C70039;">{_fmt_currency(interest_sn)}</h2>
				<p style="color:#52616B;">Total interest • {months_sn} months</p>
			</div>
			""", unsafe_allow_html=True)
			best = min(interest_av, interest_sn, interest_ai)
			best_label = (
				"AI-Optimized" if best == interest_ai else
				"Avalanche" if best == interest_av else "Snowball"
			)
			c3.markdown(f"""
			<div class="debt-card" style="text-align:center;">
				<h3 style="color:#0A1F44;">Best Strategy</h3>
				<h2 style="color:#2980B9;">{best_label}</h2>
				<p style="color:#52616B;">Lowest total interest cost</p>
			</div>
			""", unsafe_allow_html=True)

		# --- Line Chart ---
  
			def total_balance_by_month(df: pd.DataFrame) -> pd.DataFrame:
				last_rows = (
					df.sort_values(["month","debt"])
					  .groupby(["month","debt"],as_index=False)
					  .tail(1)
				)
				return last_rows.groupby("month",as_index=False)["balance"].sum()

			av_agg, sn_agg, ai_agg = map(total_balance_by_month, [sched_av, sched_sn, sched_ai])
			fig2, ax2 = plt.subplots(figsize=(7.5,4))
			ax2.plot(av_agg["month"], av_agg["balance"], lw=2.5, color="#16A085", label="Avalanche")
			ax2.plot(sn_agg["month"], sn_agg["balance"], lw=2.5, linestyle="--", color="#C70039", label="Snowball")
			ax2.plot(ai_agg["month"], ai_agg["balance"], lw=2.5, linestyle=":", color="#0A1F44", label="AI-Optimized")
			ax2.set_title("Debt Balance Over Time", fontsize=14, weight="bold", color="#0A1F44")
			ax2.set_xlabel("Month")
			ax2.set_ylabel("Total Balance (CAD)")
			ax2.grid(alpha=0.25)
			ax2.legend()
			st.markdown("<div class='debt-card'>", unsafe_allow_html=True)
			st.pyplot(fig2)
			st.markdown("</div>", unsafe_allow_html=True)

		# --- Polished Tables Section ---
  
			st.markdown("### Detailed Schedules")
			tabA, tabB, tabC = st.tabs([" Avalanche", " Snowball", "🤖 AI-Optimized"])
			with tabA:
				st.markdown("<div class='debt-table'>", unsafe_allow_html=True)
				st.dataframe(sched_av.head(20), use_container_width=True, height=350)
				st.markdown("</div>", unsafe_allow_html=True)
			with tabB:
				st.markdown("<div class='debt-table'>", unsafe_allow_html=True)
				st.dataframe(sched_sn.head(20), use_container_width=True, height=350)
				st.markdown("</div>", unsafe_allow_html=True)
			with tabC:
				st.markdown("<div class='debt-table'>", unsafe_allow_html=True)
				st.dataframe(sched_ai.head(20), use_container_width=True, height=350)
				st.markdown("</div>", unsafe_allow_html=True)

		# --- AI Explanation Box ---
			st.markdown("""
			<div class="debt-card" style="background-color:#f8f9fc; border-left:4px solid #0A1F44;">
			<b>AI-Optimized Strategy Insight:</b><br>
			Avalanche and Snowball are rule-based and usually minimize interest strictly.<br>
			The <b>AI model</b> uses dynamic weighting to distribute payments for more balanced progress 
			across debts — often improving real-world completion rates.<br><br>
			<i>Use AI mode if you value smoother payoff consistency over strict mathematical minimal interest.</i>
			</div>
			 """, unsafe_allow_html=True)
   


# --- AI Allocation Dashboard ---

			st.markdown("### AI-Optimized Strategy Summary")

# Calculate totals for comparison

			best_interest = min(interest_ai, interest_av, interest_sn)
			savings_vs_av = (interest_av - interest_ai)
			savings_vs_sn = (interest_sn - interest_ai)

			st.markdown("""
			<div class="ai-card">
				<div class="ai-header">AI Strategy Performance Overview</div>
				<div class="ai-grid">
					<div class="ai-box">
						<h4>Total Interest Paid</h4>
						<p>${:,.2f}</p>
					</div>
					<div class="ai-box">
						<h4>Months to Payoff</h4>
						<p>{}</p>
					</div>
					<div class="ai-box">
						<h4>Savings vs Avalanche</h4>
						<p style="color:#16A085;">${:,.2f}</p>
					</div>
					<div class="ai-box">
						<h4>Savings vs Snowball</h4>
						<p style="color:#C70039;">${:,.2f}</p>
					</div>
				</div>
			</div>
			""".format(interest_ai, months_ai, savings_vs_av, savings_vs_sn), unsafe_allow_html=True)

# --- Allocation Breakdown in styled format ---

			st.markdown("""
			<div class="ai-card">
				<div class="ai-header">AI-Recommended Allocation Weights</div>
			""", unsafe_allow_html=True)

			for debt_name, weight in ai_alloc.items():
				st.markdown(f"""
				<div style='display:flex; justify-content:space-between; 
							padding:8px 0; border-bottom:1px solid #eee;'>
					<span style='color:#0A1F44; font-weight:600;'>{debt_name}</span>
					<span style='color:#2980B9; font-weight:bold;'>{weight*100:.1f}%</span>
				</div>
				""", unsafe_allow_html=True)

			st.markdown("</div>", unsafe_allow_html=True)


# Risk & Portfolio

	with tab_portfolio:
     
	# --- Top Section Header ---
 
		st.markdown("""
		<div class="metric-header">
			<h2>Risk Profiling & Portfolio Intelligence</h2>
			<p>AI-driven insights for optimal asset allocation and strategy calibration.</p>
		</div>
		""", unsafe_allow_html=True)

		total_debt_balance = float(sum(d.balance for d in all_debts)) if all_debts else 0.0
		user_features = {
			"age": float(age),
			"income": float(annual_income),
			"savings_rate": float(budget_for_savings["savings_rate"]),
			"debt_ratio": float(total_debt_balance / max(annual_income, 1.0)),
			"investment_experience": float(experience),
		}

		model = build_risk_model(k=3)
		label, cluster = risk_profile(user_features, model)
		alloc = risk_to_allocation(label)
		risk_explanation = explain_risk_profile(label, user_features)

	# --- Risk Profile Overview Cards ---
 
		col1, col2, col3 = st.columns(3)
		col1.markdown(f"""
		<div class="risk-card" style="text-align:center;">
			<h3 style="color:#0A1F44;">Profile Type</h3>
			<h2 style="color:#16A085;">{label}</h2>
			<p style="color:#52616B;">Based on your debt, savings, and experience</p>
		</div>
		""", unsafe_allow_html=True)

		col2.markdown(f"""
		<div class="risk-card" style="text-align:center;">
			<h3 style="color:#0A1F44;">Investment Experience</h3>
			<h2 style="color:#F39C12;">{experience}/10</h2>
			<p style="color:#52616B;">Influences risk tolerance and volatility handling</p>
		</div>
		""", unsafe_allow_html=True)

		col3.markdown(f"""
		<div class="risk-card" style="text-align:center;">
			<h3 style="color:#0A1F44;">Savings Rate</h3>
			<h2 style="color:#2980B9;">{budget_for_savings['savings_rate']*100:.1f}%</h2>
			<p style="color:#52616B;">Higher rates indicate stronger risk capacity</p>
		</div>
		""", unsafe_allow_html=True)

	# --- Explanation & Allocation Side by Side ---
 
		left, right = st.columns([0.55, 0.45])

		with left:
			st.markdown(f"""
			<div class="risk-card">
				<div class="subtle-title">Why You Received This Profile</div>
				<p style="font-size:16px; line-height:1.6;">{risk_explanation}</p>
			</div>
			""", unsafe_allow_html=True)

		with right:
			st.markdown(f"""
			<div class="risk-card">
				<div class="subtle-title">Recommended Asset Allocation</div>
				<p>Cash: {alloc['cash']*100:.0f}%<br>
				   Bonds: {alloc['bonds']*100:.0f}%<br>
				   Equities: {alloc['equities']*100:.0f}%<br>
				   Alternatives: {alloc['alternatives']*100:.0f}%</p>
			</div>
			""", unsafe_allow_html=True)

	# --- Insight Box at Bottom ---
 
		st.markdown(f"""
		<div class="risk-card" style="background-color:#f7f9fc; border-left:4px solid #0A1F44;">
			<b>Insight:</b><br>
			Your {label.lower()} profile indicates a {alloc['equities']*100:.0f}% equity allocation —
			this balances long-term growth potential with acceptable volatility for your income level.
			Review quarterly to ensure alignment with evolving goals.
		</div>
		""", unsafe_allow_html=True)


# --- Sharpe-optimal Portfolio ---

		st.markdown("### Sharpe-Optimal Portfolio")
		classes = ["cash", "bonds", "equities", "alternatives"]
		expected_returns = np.array([0.01, 0.035, 0.08, 0.05])
		cov = np.array(
			[
				[0.0001, 0.0002, 0.0003, 0.0001],
				[0.0002, 0.0025, 0.0012, 0.0003],
				[0.0003, 0.0012, 0.06, 0.002],
				[0.0001, 0.0003, 0.002, 0.01],
			]
		)

		opt = sharpe_optimize(expected_returns, cov, risk_free=0.02)
		if opt:
			weights = opt["weights"]
			weights_pct = [float(w) * 100 for w in weights]
			roles = [
				"Liquidity / Stability",
				"Income / Lower Volatility",
				"Growth Engine",
				"Diversification / Non-Correlation",
			]

			readable_table = pd.DataFrame({
				"Asset Class": classes,
				"Optimized Weight": [f"{w:.1f}%" for w in weights_pct],
				"Role in Portfolio": roles
			})

			st.dataframe(readable_table, use_container_width=True)

# --- Portfolio Summary Card ---

			st.markdown("""
			<style>
			.portfolio-summary-card {
				background-color: #FFFFFF;
				border-radius: 14px;
				padding: 22px 25px;
				margin-top: 10px;
				margin-bottom: 25px;
				box-shadow: 0 2px 12px rgba(0,0,0,0.07);
				font-family: 'Segoe UI', sans-serif;
			}

			.portfolio-summary-header {
				background: linear-gradient(135deg, #0A1F44 0%, #193C6D 100%);
				padding: 16px 22px;
				border-radius: 12px;
				margin-bottom: 18px;
				color: white;
			}

			.portfolio-summary-header h2 {
				margin: 0;
				font-weight: 600;
			}

			.ps-grid {
				display: grid;
				grid-template-columns: repeat(3, 1fr);
				gap: 18px;
				margin-bottom: 15px;
			}

			.ps-box {
				background-color: #F8FAFC;
				padding: 15px;
				border-radius: 12px;
				text-align: center;
			}

			.ps-metric {
				font-size: 26px;
				font-weight: bold;
				color: #193C6D;
			}

			.ps-label {
				font-size: 14px;
				color: #52616B;
			}

			.ps-text {
				font-size: 15px;
				line-height: 1.65;
				color: #394B59;
				padding-top: 8px;
			}
			</style>
			""", unsafe_allow_html=True)


			st.markdown(f"""
			<div class="portfolio-summary-card"> 
				<div class="portfolio-summary-header">
					<h2>Sharpe-Optimal Portfolio Summary</h2>
				</div>
				<div class="ps-grid">
					<div class="ps-box">
						<div class="ps-metric">{opt['return']:.4f}</div>
						<div class="ps-label">Expected Return</div>
					</div>
					<div class="ps-box">
						<div class="ps-metric">{opt['vol']:.4f}</div>
						<div class="ps-label">Volatility</div>
					</div>
					<div class="ps-box">
						<div class="ps-metric">{opt['sharpe']:.4f}</div>
						<div class="ps-label">Sharpe Ratio</div>
					</div>
				</div>
				<div class="ps-text">
					The optimizer selected a mix that maximizes return per unit of risk.  
					In your case, the model favors a higher allocation to 
					<b>{classes[np.argmax(weights)].capitalize()}</b>, reflecting its role as the most efficient 
					long-term growth driver under the assumptions provided.
					Bonds and alternatives complement this by controlling volatility while maintaining diversification.
				</div>
			</div>
			""", unsafe_allow_html=True)

	# ---- Comparison with Template Allocation ----
 
			st.markdown("### Compare With Your Risk-Based Template")
			comp_df = pd.DataFrame({
				"Asset Class": classes,
				"Template Allocation": [f"{alloc[c]*100:.0f}%" for c in classes],
				"Optimized Allocation": [f"{float(w)*100:.0f}%" for w in weights],
			})
			st.dataframe(comp_df, use_container_width=True)
			fig_pie, pie_ax = plt.subplots(figsize=(6.2, 6.2))
			colors = ["#0A1F44", "#1E88E5", "#16A085", "#F39C12"]
			wedges, texts, autotexts = pie_ax.pie(
				weights_pct,
				labels=None, 
				autopct=lambda p: f"{p:.0f}%" if p > 3 else "",
				startangle=90,
				colors=colors,
				wedgeprops={"edgecolor": "white", "linewidth": 1.4},
				textprops={"fontsize": 11, "color": "white"}
			)
			plt.setp(autotexts, size=11, weight="bold")
			pie_ax.set_title("Optimized Portfolio Allocation", fontsize=15, weight="bold", color="#0A1F44")
			
			pie_ax.legend(
				[f"{cls.capitalize()} ({w:.1f}%)" for cls, w in zip(classes, weights_pct)],
				loc="upper center",
				bbox_to_anchor=(0.5, -0.08),
				ncol=4,             
				fontsize=10,
				frameon=False,
				title="Asset Classes",
				title_fontsize=11
			)

			st.pyplot(fig_pie)

# ---- Bar Chart ----

			fig_bar, bar_ax = plt.subplots(figsize=(7.5, 3.8))
			bar_colors = ["#0A1F44", "#1E88E5", "#16A085", "#F39C12"]
			bars = bar_ax.bar(classes, weights_pct, color=bar_colors, alpha=0.9, zorder=3)
			bar_ax.set_ylim(0, 100)
			bar_ax.set_ylabel("Weight (%)", fontsize=12)
			bar_ax.set_title("Optimized Portfolio Allocation (Bar View)", fontsize=15, weight="bold", color="#0A1F44")
			bar_ax.grid(alpha=0.25, zorder=0)

# Annotate values above bars

			for bar, val in zip(bars, weights_pct):
				bar_ax.text(bar.get_x() + bar.get_width()/2, val + 2, f"{val:.1f}%", 
					ha="center", color="#0A1F44", fontweight="bold", fontsize=11)

			st.pyplot(fig_bar)

# ---- Interpretation Box ----

			st.markdown(f"""
			<div style="
				background-color:#f7f9fc;
				padding: 15px;
				border-radius:10px;
				border-left: 4px solid #0A1F44;
				font-size:16px;">
				<b>Interpretation:</b><br>
				This optimized mix suggests {weights_pct[2]:.0f}% equities, indicating the model sees growth 
				as the most efficient return source under your assumptions.  
				Bonds and alternatives smooth volatility, while cash is kept minimal to avoid performance drag.
			</div>
			""", unsafe_allow_html=True)

		else:
			st.warning("Optimizer did not return a solution; using template allocation instead.")
			st.json(alloc)

	with tab_retirement:
		st.subheader("Monte Carlo retirement projection (GBM)")
		mc_results, mc_summary = monte_carlo_gbm(
			current_savings=current_savings,
			annual_contribution = 0.0,
			years=int(retirement_years),
			n_sims=int(mc_sims),
			mu=float(exp_return),
			sigma=float(exp_vol),
			inflation=float(inflation),
		)
  
# --- Dynamic unified metric display ---
  
		st.markdown(f"""
		<div style="
			display:flex; 
			justify-content:space-between; 
			gap:20px; 
			margin-bottom:20px;">
			<div style="
				flex:1; 
				background-color:#FFFFFF; 
				border-radius:12px; 
				box-shadow:0 2px 10px rgba(0,0,0,0.08);
				padding:20px; 
				text-align:center;">
				<h4 style="color:#52616B;">Monthly Income</h4>
				<h2 style="color:#0A1F44;">{_html_safe_currency(dyn_income)}</h2>
			</div>
			<div style="
				flex:1; 
				background-color:#FFFFFF; 
				border-radius:12px; 
				box-shadow:0 2px 10px rgba(0,0,0,0.08);
				padding:20px; 
				text-align:center;">
				<h4 style="color:#52616B;">Total Expenses</h4>
				<h2 style="color:#0A1F44;">{_html_safe_currency(dyn_expenses)}</h2>
			</div>
			<div style="
				flex:1; 
				background-color:#FFFFFF; 
				border-radius:12px; 
				box-shadow:0 2px 10px rgba(0,0,0,0.08);
				padding:20px; 
				text-align:center;">
				<h4 style="color:#52616B;">Monthly Savings</h4>
				<h2 style="color:#0A1F44;">{_html_safe_currency(dyn_savings)}</h2>
				<p style="color:#28B463; font-weight:bold;">▲ {dyn_rate:.1f}%</p>
			</div>
		</div>
		""", unsafe_allow_html=True)

		fig4, ax4 = plt.subplots(figsize=(8, 4))
		n, bins, patches = ax4.hist(mc_results, bins=60, color="#0A1F44", alpha=0.8, edgecolor="white", linewidth=0.5)

# Add median line and annotation

		median_val = np.median(mc_results)
		ax4.axvline(median_val, color="#F39C12", lw=2.5, linestyle="--", label=f"Median: {_fmt_currency(median_val)}")
		ax4.legend(frameon=False, fontsize=10)
		ax4.set_title(f"Distribution of Terminal Portfolio Values — {int(retirement_years)} Years",
					  fontsize=15, weight="bold", color="#0A1F44")
		ax4.set_xlabel("Portfolio Value at Retirement (CAD)", fontsize=12)
		ax4.set_ylabel("Simulation Frequency", fontsize=12)
		ax4.grid(alpha=0.25)

		st.pyplot(fig4)

# --- Visual summary of outcomes ---

		median = mc_summary["median_real"]
		p10 = mc_summary["p10_real"]
		p90 = mc_summary["p90_real"]  
		fig5, ax5 = plt.subplots(figsize=(8, 4))
		labels = ["Best Case (90th%)", "Median Case", "Worst Case (10th%)"]
		values = [p90, median, p10]
		colors = ["#2ECC71", "#3498DB", "#E74C3C"]

		bars = ax5.barh(labels, values, color=colors, alpha=0.9)
		ax5.set_title("Expected Retirement Outcomes (Real $)", fontsize=15, weight="bold", color="#0A1F44")
		ax5.set_xlabel("Real Portfolio Value (Inflation-Adjusted CAD)", fontsize=12)
		ax5.grid(alpha=0.25, zorder=0)

# Annotate values beside each bar

		for bar, val in zip(bars, values):
			ax5.text(val + (0.02 * val), bar.get_y() + bar.get_height()/2,
					 _fmt_currency(val),
					 va="center", color="#0A1F44", fontweight="bold", fontsize=11)

# Add subtle axis formatting

		ax5.spines["top"].set_visible(False)
		ax5.spines["right"].set_visible(False)
		st.pyplot(fig5)  
  
# --- Natural-language interpretation ---

		interpretation_html = f"""
		<div style='background-color:#f8f9fc; padding:15px; border-left:4px solid #0A1F44;
					border-radius:8px; line-height:1.6; font-size:16px; font-family:Segoe UI, sans-serif;'>
		<b>Interpretation:</b><br>
		If your investments earn an average of <b>{exp_return*100:.1f}%</b> annually with about 
		<b>{exp_vol*100:.1f}%</b> volatility, your retirement portfolio (in today's dollars) 
		is most likely to reach around <b>{_fmt_currency(median)}</b>.<br><br>
		In an optimistic scenario (strong markets), you could retire with approximately 
		<b>{_fmt_currency(p90)}</b>.<br>
		In a conservative scenario (weaker markets), you may have closer to 
		<b>{_fmt_currency(p10)}</b>.<br><br>
		This projection highlights how consistency, time in the market, and diversification 
		shape long-term outcomes.
		</div>
		"""

		components.html(interpretation_html, height=230)

# Recommendations

		st.markdown("**Actionable next steps**")
		efund = emergency_fund_recommendation(budget_for_savings["total_expenses"], months=6)
		alloc_text = ", ".join([f"**{v*100:.0f}% {k.capitalize()}**" for k, v in alloc.items()])
		st.markdown(
			f"- Maintain an emergency fund of **{_fmt_currency(efund)}** (6 months of expenses).\n"
			f"- For **{label}** risk, start with template allocation — {alloc_text} — and use the optimized weights as a validation check, not an automatic prescription.\n"
			f"- Prefer **avalanche** if minimizing interest is the goal; prefer **snowball** if you need behavioral wins (faster small payoffs)."
		)
		
# JSON report download

		report = {
			"timestamp": time.time(),
			"user": {
				"age": age,
				"annual_income": annual_income,
				"monthly_income": monthly_income,
				"expenses": expense_inputs,
				"current_savings": current_savings,
				"monthly_saving": monthly_saving,
				"debts": [{"name": d.name, "balance": d.balance, "apr": d.apr, "min_payment": d.min_payment} for d in all_debts],
				"experience": experience,
			},
			"budget": budget_for_savings,
			"risk_label": label,
			"allocation_template": alloc,
			"monte_carlo_summary": mc_summary,
		}
		report_bytes = json.dumps(report, indent=2).encode("utf-8")
		st.download_button("Download JSON report", data=report_bytes, file_name="pfa_report.json", mime="application/json")
  
# ======================================================
# Financial Health Tab
# ======================================================

	with tab_health:
		st.subheader("Overall Financial Health (AI-Based)")	
		total_debt_balance = float(sum(d.balance for d in all_debts)) if all_debts else 0.0
		efund_target = emergency_fund_recommendation(budget_for_savings["total_expenses"], months=6)
		emergency_ratio = min(current_savings / max(efund_target, 1), 2.0)
	
	# Prepare user feature dictionary
		health_data = {
			"age": age,
			"income": annual_income,
			"expenses": budget_for_savings["total_expenses"] * 12,
			"savings_rate": budget_for_savings["savings_rate"],
			"debt_balance": total_debt_balance,
			"credit_utilization": min(total_debt_balance / max(annual_income, 1), 1.0),
			"investment_experience": experience,
			"risk_score": 1.0 if label == "Aggressive" else (0.6 if label == "Balanced" else 0.3),
			"emergency_fund_months": emergency_ratio * 6,
		}  

		score = predict_financial_health(health_data)
		if score >= 80:
			grade = "Excellent"
		elif score >= 65:
			grade = "Good"
		elif score >= 45:
			grade = "Fair"
		else:
			grade = "Poor"

		st.metric("Your Financial Health Score", f"{score}/100", delta_color="off")
		st.progress(score / 100.0)
		st.markdown(f"**Grade:** {grade}")

	# Explain key contributors
 
		st.markdown("### Breakdown & Insights")
		if score >= 85:
			st.success("Your finances are strong — balanced debt, good savings, and stable cash flow. Keep investing regularly.")
		elif score >= 70:
			st.info("Solid position. Try to reduce debts slightly and build your emergency fund to at least 6 months.")
		elif score >= 55:
			st.warning("Moderate risk. You’re probably spending a bit too much relative to income or under-saving.")
		else:
			st.error("Your financial structure is unstable. Focus on budgeting, paying down high-APR debt, and saving consistently.")

	# Visualization of key metrics
 
		categories = ["Savings Rate", "Debt Ratio", "Emergency Fund", "Experience"]
		values = [
			budget_for_savings["savings_rate"] * 100,
			100 * (total_debt_balance / max(annual_income, 1)),
			min(emergency_ratio * 100, 200),
			experience * 10,
		]
		fig_health, ax_health = plt.subplots(figsize=(6.5, 4))
		bars = ax_health.bar(categories, values, color=["#16A085", "#E74C3C", "#3498DB", "#9B59B6"])
		ax_health.set_ylim(0, 200)
		ax_health.set_ylabel("Scaled Value")
		ax_health.set_title("Key Drivers of Financial Health", fontsize=14, weight="bold", color="#0A1F44")
		for bar, val in zip(bars, values):
			ax_health.text(bar.get_x() + bar.get_width()/2, val + 3, f"{val:.1f}", ha="center", fontsize=10, color="#0A1F44")
		st.pyplot(fig_health)

	st.success(f"Analysis completed in {time.time() - t0:.2f} seconds.")

# Footer / Dev Notes

st.markdown("---")

st.markdown("""
<div style='text-align:center; padding:20px 0; margin-top:40px; color:#0A1F44; font-family: "Segoe UI", sans-serif;'>
    <div style='font-size:16px; font-weight:600;'>
        Personal Finance Advisor
    </div>
    <div style='font-size:14px; color:#52616B;'>
        A financial intelligence platform designed by <b>Bardia Ahmadkhan</b>
    </div>
    <hr style='margin-top:18px; opacity:0.2;'>
    <div style='font-size:13px; color:#7D8A97; max-width:700px; margin:auto; line-height:1.55;'>
        <b>Disclaimer:</b> This application is an educational tool intended to help users 
        understand budgeting, debt strategies, investment concepts, and long-term financial 
        planning. It does <b>not</b> provide personalized financial, legal, tax, or investment 
        advice, and should not be used as an independent financial advisory service —
        especially by individuals who are not licensed professionals.<br><br>
        The platform may support licensed financial advisors by automating calculations 
        and offering modern AI-enhanced insights, but all decisions should be validated 
        by a qualified professional. Use this tool for learning, exploration, and 
        high-level planning — not as a substitute for certified financial advice.
    </div>
    <div style='margin-top:15px; font-size:13px; color:#7D8A97;'>
        © """ + time.strftime("%Y") + """ All rights reserved.
    </div>
</div>
""",
    unsafe_allow_html=True
)

#streamlit run financial_advisor__.py