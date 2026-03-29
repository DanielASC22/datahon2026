"""
March Madness Predictive Model
==============================
Architecture Roadmap:
  1. Data Loading & Matchup Construction
  2. Feature Engineering (Four Factors, Efficiency, Elo, Momentum)
  3. External Data Integration (KenPom/BartTorvik, Massey-style ordinals, 538)
  4. Symmetric Matchup Pipeline
  5. Season-based GroupKFold Validation
  6. Ensemble Modeling (Logistic Regression + Gradient Boosting)
  7. Calibration & Probability Capping
  8. 2026 Bracket Predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("/Users/heidynaranjo/Downloads/dataset")

# ============================================================
# 1. DATA LOADING
# ============================================================

def load_data():
    """Load all relevant CSVs."""
    data = {}
    data['kenpom'] = pd.read_csv(DATA_DIR / "KenPom Barttorvik.csv")
    data['matchups'] = pd.read_csv(DATA_DIR / "Tournament Matchups.csv")
    data['resumes'] = pd.read_csv(DATA_DIR / "Resumes.csv")
    data['ratings_538'] = pd.read_csv(DATA_DIR / "538 Ratings.csv")
    data['evan_miya'] = pd.read_csv(DATA_DIR / "EvanMiya.csv")
    data['shooting'] = pd.read_csv(DATA_DIR / "Shooting Splits.csv")
    data['kenpom_pre'] = pd.read_csv(DATA_DIR / "KenPom Preseason.csv")
    data['teamsheet'] = pd.read_csv(DATA_DIR / "Teamsheet Ranks.csv")
    data['public_picks'] = pd.read_csv(DATA_DIR / "Public Picks.csv")
    data['heat_check'] = pd.read_csv(DATA_DIR / "Heat Check Ratings.csv")
    data['z_rating'] = pd.read_csv(DATA_DIR / "Z Rating Teams.csv")
    data['tournament_sim'] = pd.read_csv(DATA_DIR / "Tournament Simulation.csv")
    data['seed_results'] = pd.read_csv(DATA_DIR / "Seed Results.csv")
    data['team_results'] = pd.read_csv(DATA_DIR / "Team Results.csv")
    data['upset_seed'] = pd.read_csv(DATA_DIR / "Upset Seed Info.csv")
    data['conf_stats'] = pd.read_csv(DATA_DIR / "Conference Stats.csv")
    data['heat_idx'] = pd.read_csv(DATA_DIR / "Heat Check Tournament Index.csv")
    data['ap_poll'] = pd.read_csv(DATA_DIR / "AP Poll Data.csv")
    return data


# ============================================================
# 2. MATCHUP CONSTRUCTION
# ============================================================

def build_matchups(matchups_df):
    """
    Pair consecutive rows in Tournament Matchups to form games.
    Rows are paired by BY YEAR NO (descending, odd/even pairs).
    The team with the lower ROUND number went further = winner.
    """
    df = matchups_df.copy()
    # Only use years with completed scores
    df = df[df['SCORE'].notna() & (df['SCORE'] != '')].copy()
    df['SCORE'] = df['SCORE'].astype(int)

    games = []
    # Sort by YEAR, then BY YEAR NO descending — pairs are consecutive
    df = df.sort_values(['YEAR', 'BY YEAR NO'], ascending=[True, False]).reset_index(drop=True)

    for i in range(0, len(df) - 1, 2):
        r1 = df.iloc[i]
        r2 = df.iloc[i + 1]
        if r1['YEAR'] != r2['YEAR']:
            continue
        # The team with lower ROUND went further (Round 1 > Round 64)
        # ROUND=1 means champion, ROUND=64 means lost in first round
        if r1['ROUND'] <= r2['ROUND']:
            winner, loser = r1, r2
        else:
            winner, loser = r2, r1

        games.append({
            'YEAR': int(r1['YEAR']),
            'CURRENT ROUND': int(r1['CURRENT ROUND']),
            'W_TEAM NO': int(winner['TEAM NO']),
            'W_TEAM': winner['TEAM'],
            'W_SEED': int(winner['SEED']),
            'W_SCORE': int(winner['SCORE']),
            'L_TEAM NO': int(loser['TEAM NO']),
            'L_TEAM': loser['TEAM'],
            'L_SEED': int(loser['SEED']),
            'L_SCORE': int(loser['SCORE']),
        })

    return pd.DataFrame(games)


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def build_team_features(data):
    """
    Build per-team-per-year feature vectors from KenPom/BartTorvik + extras.
    Includes the Four Factors, adjusted efficiency, and rating composites.
    """
    kp = data['kenpom'].copy()

    # --- Four Factors (already in KenPom Barttorvik) ---
    # EFG%  = Effective FG%
    # TOV%  = Turnover %
    # OREB% = Offensive Rebound %
    # FTR   = Free Throw Rate (FTA/FGA)
    # We also have defensive versions: EFG%D, TOV%D, DREB%, FTRD

    features = kp[['YEAR', 'TEAM NO', 'TEAM', 'SEED', 'CONF']].copy()

    # Four Factors - Offense
    features['EFG'] = kp['EFG%']
    features['TOV'] = kp['TOV%']
    features['OREB'] = kp['OREB%']
    features['FTR'] = kp['FTR']

    # Four Factors - Defense
    features['EFG_D'] = kp['EFG%D']
    features['TOV_D'] = kp['TOV%D']
    features['DREB'] = kp['DREB%']
    features['FTR_D'] = kp['FTRD']

    # Adjusted Efficiency
    features['KADJ_O'] = kp['KADJ O']
    features['KADJ_D'] = kp['KADJ D']
    features['KADJ_EM'] = kp['KADJ EM']
    features['BADJ_EM'] = kp['BADJ EM']
    features['BADJ_O'] = kp['BADJ O']
    features['BADJ_D'] = kp['BADJ D']
    features['BARTHAG'] = kp['BARTHAG']

    # Tempo
    features['KADJ_T'] = kp['KADJ T']

    # Shooting profile
    features['TWO_PT'] = kp['2PT%']
    features['TWO_PT_D'] = kp['2PT%D']
    features['THREE_PT'] = kp['3PT%']
    features['THREE_PT_D'] = kp['3PT%D']
    features['BLK'] = kp['BLK%']
    features['AST'] = kp['AST%']

    # Record
    features['WIN_PCT'] = kp['WIN%']
    features['W'] = kp['W']
    features['L'] = kp['L']

    # Experience & Talent
    features['EXP'] = kp['EXP']
    features['TALENT'] = kp['TALENT']
    features['AVG_HGT'] = kp['AVG HGT']
    features['EFF_HGT'] = kp['EFF HGT']

    # Strength of Schedule
    features['ELITE_SOS'] = kp['ELITE SOS']
    features['WAB'] = kp['WAB']

    # Points per possession
    features['PPPO'] = kp['PPPO']
    features['PPPD'] = kp['PPPD']

    # FT%
    features['FT_PCT'] = kp['FT%']

    # --- Merge KenPom Preseason (momentum = actual - preseason) ---
    pre = data['kenpom_pre'][['YEAR', 'TEAM NO', 'PRESEASON KADJ EM', 'PRESEASON KADJ EM RANK',
                               'KADJ EM RANK CHANGE', 'KADJ EM CHANGE']].copy()
    features = features.merge(pre, on=['YEAR', 'TEAM NO'], how='left')

    # Momentum: improvement from preseason
    features['MOMENTUM'] = features['KADJ EM CHANGE'].fillna(0)

    # --- Merge Resumes (Quad records, NET, ELO, etc.) ---
    res = data['resumes'][['YEAR', 'TEAM NO', 'NET RPI', 'RESUME', 'WAB RANK',
                            'ELO', 'B POWER', 'Q1 W', 'Q2 W', 'Q1 PLUS Q2 W',
                            'Q3 Q4 L', 'R SCORE']].copy()
    features = features.merge(res, on=['YEAR', 'TEAM NO'], how='left')

    # --- Merge 538 Ratings ---
    r538 = data['ratings_538'][['YEAR', 'TEAM NO', 'POWER RATING', 'POWER RATING RANK']].copy()
    r538.columns = ['YEAR', 'TEAM NO', 'RATING_538', 'RANK_538']
    features = features.merge(r538, on=['YEAR', 'TEAM NO'], how='left')

    # --- Merge EvanMiya ---
    em = data['evan_miya'][['YEAR', 'TEAM NO', 'O RATE', 'D RATE', 'RELATIVE RATING',
                             'KILLSHOTS PER GAME', 'KILLSHOTS MARGIN']].copy()
    em.columns = ['YEAR', 'TEAM NO', 'EM_O_RATE', 'EM_D_RATE', 'EM_RELATIVE',
                  'EM_KILLSHOTS', 'EM_KILL_MARGIN']
    features = features.merge(em, on=['YEAR', 'TEAM NO'], how='left')

    # --- Merge Teamsheet Ranks (composite quality metric) ---
    ts = data['teamsheet'][['YEAR', 'TEAM NO', 'RESUME AVG', 'RESUME AVG RANK',
                             'QUALITY AVG', 'QUALITY AVG RANK']].copy()
    features = features.merge(ts, on=['YEAR', 'TEAM NO'], how='left')

    # --- Merge Z Rating ---
    zr = data['z_rating'][['YEAR', 'TEAM NO', 'Z RATING']].copy()
    features = features.merge(zr, on=['YEAR', 'TEAM NO'], how='left')

    # --- Merge Heat Check Tournament Index (power/path) ---
    hci = data['heat_idx']
    hci_cols = ['YEAR', 'TEAM NO']
    for c in ['POWER', 'PATH', 'POOL VALUE']:
        if c in hci.columns:
            hci_cols.append(c)
    if len(hci_cols) > 2:
        hci_sub = hci[hci_cols].copy()
        rename_map = {c: f'SIM_{c.replace(" ", "_")}' for c in hci_cols if c not in ['YEAR', 'TEAM NO']}
        hci_sub = hci_sub.rename(columns=rename_map)
        features = features.merge(hci_sub, on=['YEAR', 'TEAM NO'], how='left')

    # --- Merge Public Picks (contrarian signal) ---
    pp = data['public_picks'][['YEAR', 'TEAM NO', 'R64', 'S16', 'F4']].copy()
    pp.columns = ['YEAR', 'TEAM NO', 'PUB_R64', 'PUB_S16', 'PUB_F4']
    # Convert percentage strings to floats
    for col in ['PUB_R64', 'PUB_S16', 'PUB_F4']:
        pp[col] = pp[col].astype(str).str.rstrip('%').astype(float) / 100.0
    features = features.merge(pp, on=['YEAR', 'TEAM NO'], how='left')

    # --- Merge Heat Check Ratings (draw categories) ---
    hc = data['heat_check']
    if 'TEAM NO' in hc.columns:
        hc_cols = ['YEAR', 'TEAM NO']
        for c in ['EASY DRAW', 'TOUGH DRAW', 'DARK HORSE', 'UPSET ALERT', 'CINDERELLA']:
            if c in hc.columns:
                hc_cols.append(c)
        hc = hc[hc_cols].copy()
        for c in ['EASY DRAW', 'TOUGH DRAW', 'DARK HORSE', 'UPSET ALERT', 'CINDERELLA']:
            if c in hc.columns:
                hc[c] = hc[c].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0}).fillna(0).astype(int)
        features = features.merge(hc, on=['YEAR', 'TEAM NO'], how='left')

    return features


# ============================================================
# 4. SYMMETRIC MATCHUP TRAINING DATA
# ============================================================

def build_training_data(games_df, team_features):
    """
    Create symmetric training pairs: for each game, create two rows
    (TeamA vs TeamB and TeamB vs TeamA) with label = 1 if TeamA won.
    This removes home/ordering bias.
    """
    # Identify numeric feature columns (everything except identifiers)
    id_cols = ['YEAR', 'TEAM NO', 'TEAM', 'SEED', 'CONF']
    feat_cols = [c for c in team_features.columns if c not in id_cols]

    rows = []
    for _, g in games_df.iterrows():
        year = g['YEAR']
        w_no = g['W_TEAM NO']
        l_no = g['L_TEAM NO']

        w_feats = team_features[(team_features['YEAR'] == year) & (team_features['TEAM NO'] == w_no)]
        l_feats = team_features[(team_features['YEAR'] == year) & (team_features['TEAM NO'] == l_no)]

        if len(w_feats) == 0 or len(l_feats) == 0:
            continue

        w_feats = w_feats.iloc[0]
        l_feats = l_feats.iloc[0]

        # Perspective 1: Winner as Team A (label=1)
        row1 = {'YEAR': year, 'CURRENT_ROUND': g['CURRENT ROUND'], 'LABEL': 1}
        for fc in feat_cols:
            row1[f'A_{fc}'] = w_feats[fc]
            row1[f'B_{fc}'] = l_feats[fc]
            # Difference feature
            try:
                row1[f'DIFF_{fc}'] = float(w_feats[fc]) - float(l_feats[fc])
            except (ValueError, TypeError):
                row1[f'DIFF_{fc}'] = 0

        # Seed difference (key feature)
        row1['SEED_DIFF'] = int(g['L_SEED']) - int(g['W_SEED'])  # positive = A is better seed
        rows.append(row1)

        # Perspective 2: Loser as Team A (label=0) — mirror
        row2 = {'YEAR': year, 'CURRENT_ROUND': g['CURRENT ROUND'], 'LABEL': 0}
        for fc in feat_cols:
            row2[f'A_{fc}'] = l_feats[fc]
            row2[f'B_{fc}'] = w_feats[fc]
            try:
                row2[f'DIFF_{fc}'] = float(l_feats[fc]) - float(w_feats[fc])
            except (ValueError, TypeError):
                row2[f'DIFF_{fc}'] = 0

        row2['SEED_DIFF'] = int(g['W_SEED']) - int(g['L_SEED'])
        rows.append(row2)

    return pd.DataFrame(rows)


# ============================================================
# 5. FEATURE SELECTION
# ============================================================

def get_model_features():
    """Return the list of DIFF features to use in modeling."""
    core_features = [
        # Four Factors differentials
        'DIFF_EFG', 'DIFF_TOV', 'DIFF_OREB', 'DIFF_FTR',
        'DIFF_EFG_D', 'DIFF_TOV_D', 'DIFF_DREB', 'DIFF_FTR_D',

        # Adjusted Efficiency differentials
        'DIFF_KADJ_O', 'DIFF_KADJ_D', 'DIFF_KADJ_EM',
        'DIFF_BADJ_EM', 'DIFF_BARTHAG',

        # Shooting
        'DIFF_TWO_PT', 'DIFF_THREE_PT', 'DIFF_TWO_PT_D', 'DIFF_THREE_PT_D',

        # Record & experience
        'DIFF_WIN_PCT', 'DIFF_EXP', 'DIFF_TALENT',

        # Strength of schedule
        'DIFF_ELITE_SOS', 'DIFF_WAB',

        # Efficiency
        'DIFF_PPPO', 'DIFF_PPPD',

        # Momentum
        'DIFF_MOMENTUM',

        # Ratings
        'DIFF_ELO', 'DIFF_RATING_538',

        # Resume
        'DIFF_Q1 PLUS Q2 W', 'DIFF_R SCORE',

        # EvanMiya
        'DIFF_EM_RELATIVE', 'DIFF_EM_KILLSHOTS',

        # Composite
        'DIFF_QUALITY AVG', 'DIFF_Z RATING',

        # Heat Check Index
        'DIFF_SIM_POWER',

        # Seed
        'SEED_DIFF',
    ]
    return core_features


# ============================================================
# 6. VALIDATION — Season-Based GroupKFold
# ============================================================

def validate_model(train_df, feature_cols):
    """
    Season-based GroupKFold cross-validation.
    Each fold holds out one full season to prevent leakage.
    """
    X = train_df[feature_cols].fillna(0).values
    y = train_df['LABEL'].values
    groups = train_df['YEAR'].values

    gkf = GroupKFold(n_splits=min(len(train_df['YEAR'].unique()), 10))

    log_losses = []
    brier_scores = []
    all_probs = []
    all_labels = []
    all_years = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        val_years = groups[val_idx]

        # Ensemble: Logistic Regression + Gradient Boosting
        lr = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs'))
        ])
        gb = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=10, random_state=42
        )

        lr.fit(X_train, y_train)
        gb.fit(X_train, y_train)

        # Blend predictions (0.4 LR + 0.6 GB)
        p_lr = lr.predict_proba(X_val)[:, 1]
        p_gb = gb.predict_proba(X_val)[:, 1]
        p_blend = 0.4 * p_lr + 0.6 * p_gb

        # Probability capping
        p_blend = np.clip(p_blend, 0.02, 0.98)

        ll = log_loss(y_val, p_blend)
        bs = brier_score_loss(y_val, p_blend)
        log_losses.append(ll)
        brier_scores.append(bs)

        all_probs.extend(p_blend)
        all_labels.extend(y_val)
        all_years.extend(val_years)

        held_out_year = np.unique(val_years)
        print(f"  Fold {fold+1} | Year(s) {held_out_year} | LogLoss={ll:.4f} | Brier={bs:.4f}")

    print(f"\n  Mean LogLoss: {np.mean(log_losses):.4f} (+/- {np.std(log_losses):.4f})")
    print(f"  Mean Brier:   {np.mean(brier_scores):.4f} (+/- {np.std(brier_scores):.4f})")

    return np.array(all_probs), np.array(all_labels), np.array(all_years)


# ============================================================
# 7. CALIBRATION DIAGNOSTICS
# ============================================================

def plot_calibration(probs, labels, title="Calibration Curve"):
    """Plot reliability diagram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Calibration curve
    fraction_pos, mean_predicted = calibration_curve(labels, probs, n_bins=10, strategy='uniform')
    axes[0].plot(mean_predicted, fraction_pos, 's-', label='Model')
    axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect')
    axes[0].set_xlabel('Mean Predicted Probability')
    axes[0].set_ylabel('Fraction of Positives')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Prediction distribution
    axes[1].hist(probs, bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Predicted Probability')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Prediction Distribution')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/heidynaranjo/Datathon/calibration_curve.png', dpi=150)
    plt.show()
    print("Saved: calibration_curve.png")


def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance from Gradient Boosting."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices], align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Features (Gradient Boosting)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/Users/heidynaranjo/Datathon/feature_importance.png', dpi=150)
    plt.show()
    print("Saved: feature_importance.png")


# ============================================================
# 8. FINAL MODEL TRAINING + 2026 PREDICTIONS
# ============================================================

def train_final_model(train_df, feature_cols):
    """Train on all historical data with Isotonic calibration."""
    X = train_df[feature_cols].fillna(0).values
    y = train_df['LABEL'].values

    # Logistic Regression (calibrated)
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs'))
    ])
    lr_cal = CalibratedClassifierCV(lr, method='isotonic', cv=5)
    lr_cal.fit(X, y)

    # Gradient Boosting (calibrated)
    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42
    )
    gb_cal = CalibratedClassifierCV(gb, method='isotonic', cv=5)
    gb_cal.fit(X, y)

    # Also train raw GB for feature importance
    gb_raw = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42
    )
    gb_raw.fit(X, y)

    return lr_cal, gb_cal, gb_raw


def predict_2026_matchups(data, team_features, lr_model, gb_model, feature_cols):
    """Generate predictions for 2026 tournament matchups."""
    matchups = data['matchups']
    m2026 = matchups[matchups['YEAR'] == 2026].copy()
    m2026 = m2026.sort_values('BY YEAR NO', ascending=False).reset_index(drop=True)

    predictions = []
    for i in range(0, len(m2026) - 1, 2):
        t1 = m2026.iloc[i]
        t2 = m2026.iloc[i + 1]

        t1_no = int(t1['TEAM NO'])
        t2_no = int(t2['TEAM NO'])

        t1_feats = team_features[(team_features['YEAR'] == 2026) & (team_features['TEAM NO'] == t1_no)]
        t2_feats = team_features[(team_features['YEAR'] == 2026) & (team_features['TEAM NO'] == t2_no)]

        if len(t1_feats) == 0 or len(t2_feats) == 0:
            continue

        t1_feats = t1_feats.iloc[0]
        t2_feats = t2_feats.iloc[0]

        id_cols = ['YEAR', 'TEAM NO', 'TEAM', 'SEED', 'CONF']
        feat_cols_team = [c for c in team_features.columns if c not in id_cols]

        row = {}
        for fc in feat_cols_team:
            try:
                row[f'DIFF_{fc}'] = float(t1_feats[fc]) - float(t2_feats[fc])
            except (ValueError, TypeError):
                row[f'DIFF_{fc}'] = 0
        row['SEED_DIFF'] = int(t2['SEED']) - int(t1['SEED'])

        X_pred = pd.DataFrame([row])[feature_cols].fillna(0).values

        p_lr = lr_model.predict_proba(X_pred)[:, 1][0]
        p_gb = gb_model.predict_proba(X_pred)[:, 1][0]
        p_blend = 0.4 * p_lr + 0.6 * p_gb

        # Probability capping
        p_blend = np.clip(p_blend, 0.02, 0.98)

        predictions.append({
            'ROUND': int(t1['CURRENT ROUND']),
            'Team_A': t1['TEAM'],
            'Seed_A': int(t1['SEED']),
            'Team_B': t2['TEAM'],
            'Seed_B': int(t2['SEED']),
            'P(A wins)': round(p_blend, 4),
            'P(B wins)': round(1 - p_blend, 4),
            'Predicted Winner': t1['TEAM'] if p_blend > 0.5 else t2['TEAM'],
        })

    return pd.DataFrame(predictions)


# ============================================================
# 9. BRIER SCORE — 33.3% OPTIMAL RISK STRATEGY
# ============================================================
"""
Mathematical Logic for the 33.3% Optimal Risk Strategy
=======================================================

Brier Score = (1/N) * Σ (forecast_i - outcome_i)²

For an underdog with true win probability p:
  - If you predict p:     Expected Brier = p*(1-p)² + (1-p)*p²  = p(1-p)
  - If you predict 0.50:  Expected Brier = p*(0.5)² + (1-p)*(0.5)² = 0.25

The OPTIMAL RISK threshold: deviate from the favorite only when the
underdog's true probability exceeds ~33.3%.

Why 33.3%? Consider a 12-vs-5 matchup where the 12 seed has a ~35% chance:
  - Predicting the favorite (p=0.65): Brier = 0.65*0.35 = 0.2275
  - Predicting 50/50 (p=0.50):        Brier = 0.25
  - Predicting the underdog (p=0.35):  Brier = 0.2275 (same by symmetry)

At exactly p=0.333, the cost of being wrong on the upset equals the gain
from being right. Below 33.3%, always lean toward the favorite.

PRACTICAL IMPLEMENTATION:
  - For games where P(upset) < 0.33: push probability TOWARD the favorite
    (but don't exceed 0.98)
  - For games where P(upset) >= 0.33: keep the model's probability —
    these are high-value upset picks
  - Never predict 0.0 or 1.0 — the log loss penalty is infinite
"""

def apply_brier_strategy(predictions_df, aggressive=False):
    """
    Apply the 33.3% risk strategy to predictions.
    If aggressive=True, slightly boost upset probabilities above threshold.
    """
    df = predictions_df.copy()

    for idx, row in df.iterrows():
        p_fav = max(row['P(A wins)'], row['P(B wins)'])
        p_dog = min(row['P(A wins)'], row['P(B wins)'])

        if p_dog < 0.333 and not aggressive:
            # Push slightly more toward favorite
            boost = 0.02
            if row['P(A wins)'] > row['P(B wins)']:
                df.at[idx, 'P(A wins)'] = min(row['P(A wins)'] + boost, 0.98)
                df.at[idx, 'P(B wins)'] = max(row['P(B wins)'] - boost, 0.02)
            else:
                df.at[idx, 'P(B wins)'] = min(row['P(B wins)'] + boost, 0.98)
                df.at[idx, 'P(A wins)'] = max(row['P(A wins)'] - boost, 0.02)

    return df


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 60)
    print("MARCH MADNESS PREDICTIVE MODEL")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/7] Loading data...")
    data = load_data()
    print(f"  Loaded {len(data)} datasets")

    # Step 2: Build matchups from historical tournaments
    print("\n[2/7] Building historical matchups...")
    games = build_matchups(data['matchups'])
    print(f"  Built {len(games)} games across {games['YEAR'].nunique()} seasons")
    print(f"  Seasons: {sorted(games['YEAR'].unique())}")

    # Step 3: Build team features
    print("\n[3/7] Engineering features...")
    team_features = build_team_features(data)
    print(f"  {len(team_features)} team-seasons with {len(team_features.columns)} columns")

    # Step 4: Build symmetric training data
    print("\n[4/7] Building symmetric training pairs...")
    train_df = build_training_data(games, team_features)
    print(f"  {len(train_df)} training rows ({len(train_df)//2} games × 2 perspectives)")

    # Step 5: Feature selection & validation
    feature_cols = get_model_features()
    # Filter to features that actually exist in training data
    feature_cols = [f for f in feature_cols if f in train_df.columns]
    print(f"\n  Using {len(feature_cols)} features")

    print("\n[5/7] Cross-validation (Season-based GroupKFold)...")
    cv_probs, cv_labels, cv_years = validate_model(train_df, feature_cols)

    # Step 6: Calibration diagnostics
    print("\n[6/7] Plotting diagnostics...")
    plot_calibration(cv_probs, cv_labels)

    # Step 7: Train final model & predict 2026
    print("\n[7/7] Training final model on all historical data...")
    lr_model, gb_model, gb_raw = train_final_model(train_df, feature_cols)

    plot_feature_importance(gb_raw, feature_cols)

    # Generate 2026 predictions
    print("\n" + "=" * 60)
    print("2026 TOURNAMENT PREDICTIONS")
    print("=" * 60)
    preds = predict_2026_matchups(data, team_features, lr_model, gb_model, feature_cols)

    if len(preds) > 0:
        preds_adjusted = apply_brier_strategy(preds)

        # Display Round of 64
        r64 = preds_adjusted[preds_adjusted['ROUND'] == 64]
        if len(r64) > 0:
            print("\nRound of 64:")
            print("-" * 80)
            for _, row in r64.iterrows():
                winner_marker = ">>>" if row['Predicted Winner'] == row['Team_A'] else "   "
                loser_marker = "   " if row['Predicted Winner'] == row['Team_A'] else ">>>"
                print(f"  {winner_marker} ({row['Seed_A']:>2}) {row['Team_A']:<20} {row['P(A wins)']:.1%}")
                print(f"  {loser_marker} ({row['Seed_B']:>2}) {row['Team_B']:<20} {row['P(B wins)']:.1%}")
                print()

        # Save all predictions
        preds_adjusted.to_csv('/Users/heidynaranjo/Datathon/predictions_2026.csv', index=False)
        print(f"\nSaved predictions to predictions_2026.csv")

        # Summary stats
        upsets = preds_adjusted[
            ((preds_adjusted['Seed_A'] > preds_adjusted['Seed_B']) &
             (preds_adjusted['Predicted Winner'] == preds_adjusted['Team_A'])) |
            ((preds_adjusted['Seed_B'] > preds_adjusted['Seed_A']) &
             (preds_adjusted['Predicted Winner'] == preds_adjusted['Team_B']))
        ]
        print(f"\n  Predicted upsets: {len(upsets)} / {len(preds_adjusted)}")
    else:
        print("  No 2026 matchups found to predict.")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
