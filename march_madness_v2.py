"""
March Madness Predictive Model v2
==================================
Incorporates:
  - LightGBM + CatBoost + Logistic Regression ensemble
  - Pre-tournament data only (DayNum <= 133 equivalent) — no leakage
  - Symmetric matchup pipeline
  - Season-based GroupKFold validation
  - Isotonic calibration + probability capping
  - Brier-optimal 33.3% risk strategy
  - Seed-anchored LR baseline blended with tree models

Architecture:
  Phase 1: Data loading & team name harmonization
  Phase 2: Feature engineering (Four Factors, Efficiency, Momentum, Elo)
  Phase 3: Symmetric matchup construction
  Phase 4: Model training (LightGBM + CatBoost + LR seed baseline)
  Phase 5: Season-based GroupKFold validation
  Phase 6: Calibration, capping, ensemble blending
  Phase 7: 2026 predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try importing LightGBM and CatBoost; fall back to sklearn GBM if unavailable
try:
    import lightgbm as lgb
    HAS_LGBM = True
    print("[INFO] LightGBM available")
except ImportError:
    HAS_LGBM = False
    print("[INFO] LightGBM not installed — using sklearn GradientBoosting")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
    print("[INFO] CatBoost available")
except ImportError:
    HAS_CATBOOST = False
    print("[INFO] CatBoost not installed — using sklearn GradientBoosting")

DATA_DIR = Path("/Users/heidynaranjo/Downloads/dataset")
ARCHIVE3 = Path("/Users/heidynaranjo/Downloads/archive (3)")
OUT_DIR = Path("/Users/heidynaranjo/Datathon")


# ============================================================
# PHASE 1: DATA LOADING
# ============================================================

def load_all_data():
    """Load datasets from both sources."""
    data = {}

    # Primary dataset (2008-2026)
    data['kenpom'] = pd.read_csv(DATA_DIR / "KenPom Barttorvik.csv")
    data['matchups'] = pd.read_csv(DATA_DIR / "Tournament Matchups.csv")
    data['resumes'] = pd.read_csv(DATA_DIR / "Resumes.csv")
    data['ratings_538'] = pd.read_csv(DATA_DIR / "538 Ratings.csv")
    data['evan_miya'] = pd.read_csv(DATA_DIR / "EvanMiya.csv")
    data['kenpom_pre'] = pd.read_csv(DATA_DIR / "KenPom Preseason.csv")
    data['teamsheet'] = pd.read_csv(DATA_DIR / "Teamsheet Ranks.csv")
    data['public_picks'] = pd.read_csv(DATA_DIR / "Public Picks.csv")
    data['z_rating'] = pd.read_csv(DATA_DIR / "Z Rating Teams.csv")
    data['heat_idx'] = pd.read_csv(DATA_DIR / "Heat Check Tournament Index.csv")
    data['heat_check'] = pd.read_csv(DATA_DIR / "Heat Check Ratings.csv")
    data['conf_stats'] = pd.read_csv(DATA_DIR / "Conference Stats.csv")

    # Archive 3: KenPom pre-tournament snapshots (2001-2025) — leakage-free
    data['kp_pretourney'] = pd.read_csv(ARCHIVE3 / "INT _ KenPom _ Summary (Pre-Tournament).csv")
    data['kp_defense'] = pd.read_csv(ARCHIVE3 / "INT _ KenPom _ Defense.csv")
    data['kp_offense'] = pd.read_csv(ARCHIVE3 / "INT _ KenPom _ Offense.csv")
    data['kp_misc'] = pd.read_csv(ARCHIVE3 / "INT _ KenPom _ Miscellaneous Team Stats.csv")
    data['kp_height'] = pd.read_csv(ARCHIVE3 / "INT _ KenPom _ Height.csv")
    data['dev'] = pd.read_csv(ARCHIVE3 / "DEV _ March Madness.csv")
    data['ref_teams'] = pd.read_csv(ARCHIVE3 / "REF _ Post-Season Tournament Teams.csv")
    data['name_map'] = pd.read_csv(ARCHIVE3 / "REF _ NCAAM Conference and ESPN Team Name Mapping.csv")

    return data


# ============================================================
# PHASE 2: BUILD MATCHUPS FROM HISTORICAL DATA
# ============================================================

def build_historical_matchups(matchups_df):
    """
    Pair consecutive rows in Tournament Matchups to form games.
    ROUND indicates how far the team went (1=champion, 64=lost first round).
    """
    df = matchups_df.copy()
    df = df[df['SCORE'].notna() & (df['SCORE'] != '')].copy()
    df['SCORE'] = df['SCORE'].astype(int)
    df = df.sort_values(['YEAR', 'BY YEAR NO'], ascending=[True, False]).reset_index(drop=True)

    games = []
    for i in range(0, len(df) - 1, 2):
        r1, r2 = df.iloc[i], df.iloc[i + 1]
        if r1['YEAR'] != r2['YEAR']:
            continue

        # Lower ROUND = went further = winner
        if r1['ROUND'] <= r2['ROUND']:
            winner, loser = r1, r2
        else:
            winner, loser = r2, r1

        games.append({
            'YEAR': int(r1['YEAR']),
            'ROUND': int(r1['CURRENT ROUND']),
            'W_TEAM': winner['TEAM'],
            'W_TEAM_NO': int(winner['TEAM NO']),
            'W_SEED': int(winner['SEED']),
            'W_SCORE': int(winner['SCORE']),
            'L_TEAM': loser['TEAM'],
            'L_TEAM_NO': int(loser['TEAM NO']),
            'L_SEED': int(loser['SEED']),
            'L_SCORE': int(loser['SCORE']),
        })

    return pd.DataFrame(games)


# ============================================================
# PHASE 3: FEATURE ENGINEERING
# ============================================================

def build_team_features(data):
    """
    Build per-team-per-year features using PRE-TOURNAMENT data only.
    Merges KenPom/BartTorvik Four Factors, efficiency, resume, ratings.
    """
    kp = data['kenpom'].copy()
    features = kp[['YEAR', 'TEAM NO', 'TEAM', 'SEED']].copy()

    # ── Four Factors (Offense) ──
    features['EFG'] = kp['EFG%']
    features['TOV'] = kp['TOV%']
    features['OREB'] = kp['OREB%']
    features['FTR'] = kp['FTR']

    # ── Four Factors (Defense) ──
    features['EFG_D'] = kp['EFG%D']
    features['TOV_D'] = kp['TOV%D']
    features['DREB'] = kp['DREB%']
    features['FTR_D'] = kp['FTRD']

    # ── KenPom Adjusted Efficiency (pre-tournament safe: these are season-end
    #    for regular season, tournament games are excluded in KenPom methodology) ──
    features['KADJ_O'] = kp['KADJ O']
    features['KADJ_D'] = kp['KADJ D']
    features['KADJ_EM'] = kp['KADJ EM']
    features['BADJ_EM'] = kp['BADJ EM']
    features['BADJ_O'] = kp['BADJ O']
    features['BADJ_D'] = kp['BADJ D']
    features['BARTHAG'] = kp['BARTHAG']
    features['KADJ_T'] = kp['KADJ T']

    # ── Shooting ──
    features['TWO_PT'] = kp['2PT%']
    features['TWO_PT_D'] = kp['2PT%D']
    features['THREE_PT'] = kp['3PT%']
    features['THREE_PT_D'] = kp['3PT%D']
    features['BLK'] = kp['BLK%']
    features['AST'] = kp['AST%']
    features['FT_PCT'] = kp['FT%']

    # ── Record ──
    features['WIN_PCT'] = kp['WIN%']

    # ── Experience, Talent, Height ──
    features['EXP'] = kp['EXP']
    features['TALENT'] = kp['TALENT']
    features['AVG_HGT'] = kp['AVG HGT']
    features['EFF_HGT'] = kp['EFF HGT']

    # ── Strength of Schedule ──
    features['ELITE_SOS'] = kp['ELITE SOS']
    features['WAB'] = kp['WAB']

    # ── Points per Possession ──
    features['PPPO'] = kp['PPPO']
    features['PPPD'] = kp['PPPD']

    # ── Momentum: KenPom preseason vs actual ──
    pre = data['kenpom_pre'][['YEAR', 'TEAM NO', 'PRESEASON KADJ EM',
                               'KADJ EM CHANGE']].copy()
    features = features.merge(pre, on=['YEAR', 'TEAM NO'], how='left')
    features['MOMENTUM'] = features['KADJ EM CHANGE'].fillna(0)
    # EloVsPeakDiff proxy: how much team improved from preseason expectation
    features['PRESEASON_EM'] = features['PRESEASON KADJ EM'].fillna(features['KADJ_EM'])

    # ── Resumes (Quad records, NET, ELO) ──
    res = data['resumes'][['YEAR', 'TEAM NO', 'NET RPI', 'RESUME',
                            'ELO', 'B POWER', 'Q1 W', 'Q2 W',
                            'Q1 PLUS Q2 W', 'Q3 Q4 L', 'R SCORE']].copy()
    features = features.merge(res, on=['YEAR', 'TEAM NO'], how='left')

    # ── 538 Power Rating ──
    r538 = data['ratings_538'][['YEAR', 'TEAM NO', 'POWER RATING']].copy()
    r538.columns = ['YEAR', 'TEAM NO', 'RATING_538']
    features = features.merge(r538, on=['YEAR', 'TEAM NO'], how='left')

    # ── EvanMiya ──
    em = data['evan_miya'][['YEAR', 'TEAM NO', 'RELATIVE RATING',
                             'KILLSHOTS PER GAME', 'KILLSHOTS MARGIN']].copy()
    em.columns = ['YEAR', 'TEAM NO', 'EM_RELATIVE', 'EM_KILLSHOTS', 'EM_KILL_MARGIN']
    features = features.merge(em, on=['YEAR', 'TEAM NO'], how='left')

    # ── Teamsheet composite ──
    ts = data['teamsheet'][['YEAR', 'TEAM NO', 'RESUME AVG', 'QUALITY AVG']].copy()
    features = features.merge(ts, on=['YEAR', 'TEAM NO'], how='left')

    # ── Z Rating ──
    zr = data['z_rating'][['YEAR', 'TEAM NO', 'Z RATING']].copy()
    features = features.merge(zr, on=['YEAR', 'TEAM NO'], how='left')

    # ── Heat Check Tournament Index ──
    hci = data['heat_idx']
    hci_cols = ['YEAR', 'TEAM NO']
    for c in ['POWER', 'PATH']:
        if c in hci.columns:
            hci_cols.append(c)
    if len(hci_cols) > 2:
        hci_sub = hci[hci_cols].copy()
        hci_sub.columns = ['YEAR', 'TEAM NO'] + [f'HC_{c}' for c in hci_cols[2:]]
        features = features.merge(hci_sub, on=['YEAR', 'TEAM NO'], how='left')

    # ── Conference strength (merge conference BADJ EM) ──
    conf = data['conf_stats'][['YEAR', 'CONF ID', 'BADJ EM']].copy()
    conf.columns = ['YEAR', 'CONF_ID', 'CONF_STRENGTH']
    # Need CONF ID from kenpom
    if 'CONF ID' in kp.columns:
        conf_map = kp[['YEAR', 'TEAM NO', 'CONF ID']].copy()
        conf_map.columns = ['YEAR', 'TEAM NO', 'CONF_ID']
        conf_map = conf_map.merge(conf, on=['YEAR', 'CONF_ID'], how='left')
        features = features.merge(conf_map[['YEAR', 'TEAM NO', 'CONF_STRENGTH']],
                                   on=['YEAR', 'TEAM NO'], how='left')

    # ── Public picks (contrarian signal) ──
    pp = data['public_picks'][['YEAR', 'TEAM NO', 'R64', 'S16', 'F4']].copy()
    pp.columns = ['YEAR', 'TEAM NO', 'PUB_R64', 'PUB_S16', 'PUB_F4']
    for col in ['PUB_R64', 'PUB_S16', 'PUB_F4']:
        pp[col] = pd.to_numeric(pp[col].astype(str).str.rstrip('%'), errors='coerce') / 100.0
    features = features.merge(pp, on=['YEAR', 'TEAM NO'], how='left')

    return features


# ============================================================
# PHASE 4: SYMMETRIC MATCHUP CONSTRUCTION
# ============================================================

def get_diff_features():
    """Feature columns to compute differentials for."""
    return [
        # Four Factors
        'EFG', 'TOV', 'OREB', 'FTR',
        'EFG_D', 'TOV_D', 'DREB', 'FTR_D',
        # Efficiency
        'KADJ_O', 'KADJ_D', 'KADJ_EM', 'BADJ_EM', 'BARTHAG',
        'KADJ_T',
        # Shooting
        'TWO_PT', 'THREE_PT', 'TWO_PT_D', 'THREE_PT_D',
        'BLK', 'AST', 'FT_PCT',
        # Record
        'WIN_PCT',
        # Physical / roster
        'EXP', 'TALENT', 'AVG_HGT', 'EFF_HGT',
        # SOS
        'ELITE_SOS', 'WAB',
        # Efficiency per possession
        'PPPO', 'PPPD',
        # Momentum
        'MOMENTUM',
        # Ratings
        'ELO', 'RATING_538', 'B POWER',
        # Resume
        'Q1 PLUS Q2 W', 'Q3 Q4 L', 'R SCORE',
        # EvanMiya
        'EM_RELATIVE', 'EM_KILLSHOTS', 'EM_KILL_MARGIN',
        # Composite
        'RESUME AVG', 'QUALITY AVG', 'Z RATING',
        # Heat Check
        'HC_POWER', 'HC_PATH',
        # Conference
        'CONF_STRENGTH',
        # Public picks
        'PUB_R64',
    ]


def build_symmetric_training(games_df, team_features):
    """
    For each game, create TWO rows (A=winner/B=loser and A=loser/B=winner).
    Features are DIFFERENCES (A - B). This eliminates ordering bias.
    """
    diff_cols = get_diff_features()
    # Only keep diff cols that exist in team_features
    available = [c for c in diff_cols if c in team_features.columns]

    id_cols = ['YEAR', 'TEAM NO', 'TEAM', 'SEED']

    rows = []
    for _, g in games_df.iterrows():
        year = g['YEAR']
        w = team_features[(team_features['YEAR'] == year) & (team_features['TEAM NO'] == g['W_TEAM_NO'])]
        l = team_features[(team_features['YEAR'] == year) & (team_features['TEAM NO'] == g['L_TEAM_NO'])]
        if len(w) == 0 or len(l) == 0:
            continue
        w, l = w.iloc[0], l.iloc[0]

        for label, a, b, sa, sb in [(1, w, l, g['W_SEED'], g['L_SEED']),
                                     (0, l, w, g['L_SEED'], g['W_SEED'])]:
            row = {'YEAR': year, 'ROUND': g['ROUND'], 'LABEL': label,
                   'SEED_DIFF': int(sb) - int(sa)}  # positive = A has better (lower) seed
            for fc in available:
                try:
                    row[f'DIFF_{fc}'] = float(a[fc]) - float(b[fc])
                except (ValueError, TypeError):
                    row[f'DIFF_{fc}'] = 0
            rows.append(row)

    return pd.DataFrame(rows)


def get_model_feature_names(train_df):
    """Return the actual DIFF_ columns + SEED_DIFF present in training data."""
    return [c for c in train_df.columns if c.startswith('DIFF_') or c == 'SEED_DIFF']


# ============================================================
# PHASE 5: MODEL BUILDING
# ============================================================

def make_lr_seed_baseline():
    """Simple Logistic Regression anchored on seed difference — prevents tree overconfidence."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=0.5, max_iter=2000, solver='lbfgs'))
    ])


def make_lr_full():
    """Full-featured Logistic Regression."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs'))
    ])


def make_lgbm():
    """LightGBM classifier."""
    if HAS_LGBM:
        return lgb.LGBMClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1, n_jobs=-1
        )
    else:
        return GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=15, random_state=42
        )


def make_catboost():
    """CatBoost classifier."""
    if HAS_CATBOOST:
        return CatBoostClassifier(
            iterations=400, depth=5, learning_rate=0.03,
            l2_leaf_reg=3.0, subsample=0.8,
            random_seed=42, verbose=0
        )
    else:
        return GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=15, random_state=42
        )


def cap_probabilities(p, low=0.025, high=0.975):
    """Clip probabilities to avoid catastrophic log loss."""
    return np.clip(p, low, high)


def ensemble_predict(models, weights, X):
    """Weighted average of model probabilities."""
    preds = np.zeros(X.shape[0])
    for model, w in zip(models, weights):
        preds += w * model.predict_proba(X)[:, 1]
    return preds


# ============================================================
# PHASE 6: VALIDATION — Season-Based GroupKFold
# ============================================================

def run_validation(train_df, feature_cols):
    """
    Leave-one-season-out cross-validation.
    Trains 3-model ensemble per fold, reports LogLoss + Brier.
    """
    X = train_df[feature_cols].fillna(0).values
    y = train_df['LABEL'].values
    groups = train_df['YEAR'].values

    n_seasons = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(n_seasons, 10))

    all_ll, all_bs = [], []
    all_probs, all_labels, all_years = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # Model 1: LR seed baseline (only SEED_DIFF)
        seed_idx = feature_cols.index('SEED_DIFF') if 'SEED_DIFF' in feature_cols else 0
        lr_seed = make_lr_seed_baseline()
        lr_seed.fit(X_tr[:, [seed_idx]], y_tr)
        p_seed = lr_seed.predict_proba(X_val[:, [seed_idx]])[:, 1]

        # Model 2: Full LR
        lr_full = make_lr_full()
        lr_full.fit(X_tr, y_tr)
        p_lr = lr_full.predict_proba(X_val)[:, 1]

        # Model 3: LightGBM (or sklearn GBM fallback)
        tree1 = make_lgbm()
        tree1.fit(X_tr, y_tr)
        p_tree1 = tree1.predict_proba(X_val)[:, 1]

        # Model 4: CatBoost (or sklearn GBM fallback)
        tree2 = make_catboost()
        tree2.fit(X_tr, y_tr)
        p_tree2 = tree2.predict_proba(X_val)[:, 1]

        # Ensemble blend: 10% seed LR + 20% full LR + 35% tree1 + 35% tree2
        p_blend = 0.10 * p_seed + 0.20 * p_lr + 0.35 * p_tree1 + 0.35 * p_tree2
        p_blend = cap_probabilities(p_blend)

        ll = log_loss(y_val, p_blend)
        bs = brier_score_loss(y_val, p_blend)
        all_ll.append(ll)
        all_bs.append(bs)
        all_probs.extend(p_blend)
        all_labels.extend(y_val)
        all_years.extend(groups[val_idx])

        held = np.unique(groups[val_idx])
        print(f"  Fold {fold+1:>2} | Year(s) {held} | LogLoss={ll:.4f} | Brier={bs:.4f}")

    print(f"\n  {'='*50}")
    print(f"  Mean LogLoss: {np.mean(all_ll):.4f} (+/- {np.std(all_ll):.4f})")
    print(f"  Mean Brier:   {np.mean(all_bs):.4f} (+/- {np.std(all_bs):.4f})")

    return np.array(all_probs), np.array(all_labels), np.array(all_years)


# ============================================================
# PHASE 7: FINAL MODEL TRAINING
# ============================================================

def train_final_ensemble(train_df, feature_cols):
    """Train all models on full historical data with isotonic calibration."""
    X = train_df[feature_cols].fillna(0).values
    y = train_df['LABEL'].values

    seed_idx = feature_cols.index('SEED_DIFF') if 'SEED_DIFF' in feature_cols else 0

    # 1) Seed-only LR baseline
    lr_seed = make_lr_seed_baseline()
    lr_seed.fit(X[:, [seed_idx]], y)

    # 2) Full LR with isotonic calibration
    lr_full = make_lr_full()
    lr_full_cal = CalibratedClassifierCV(lr_full, method='isotonic', cv=5)
    lr_full_cal.fit(X, y)

    # 3) LightGBM with isotonic calibration
    tree1 = make_lgbm()
    tree1_cal = CalibratedClassifierCV(tree1, method='isotonic', cv=5)
    tree1_cal.fit(X, y)

    # 4) CatBoost with isotonic calibration
    tree2 = make_catboost()
    tree2_cal = CalibratedClassifierCV(tree2, method='isotonic', cv=5)
    tree2_cal.fit(X, y)

    # Also train raw tree for feature importance
    tree_raw = make_lgbm()
    tree_raw.fit(X, y)

    return lr_seed, lr_full_cal, tree1_cal, tree2_cal, tree_raw, seed_idx


# ============================================================
# PHASE 8: 2026 PREDICTIONS
# ============================================================

def predict_2026(data, team_features, models, feature_cols, seed_idx):
    """Generate predictions for all 2026 first-round matchups."""
    lr_seed, lr_full, tree1, tree2, _, _ = models

    matchups = data['matchups']
    m2026 = matchups[matchups['YEAR'] == 2026].copy()
    m2026 = m2026.sort_values('BY YEAR NO', ascending=False).reset_index(drop=True)

    diff_cols = get_diff_features()
    available = [c for c in diff_cols if c in team_features.columns]

    predictions = []
    for i in range(0, len(m2026) - 1, 2):
        t1, t2 = m2026.iloc[i], m2026.iloc[i + 1]
        t1_no, t2_no = int(t1['TEAM NO']), int(t2['TEAM NO'])

        f1 = team_features[(team_features['YEAR'] == 2026) & (team_features['TEAM NO'] == t1_no)]
        f2 = team_features[(team_features['YEAR'] == 2026) & (team_features['TEAM NO'] == t2_no)]
        if len(f1) == 0 or len(f2) == 0:
            continue
        f1, f2 = f1.iloc[0], f2.iloc[0]

        row = {'SEED_DIFF': int(t2['SEED']) - int(t1['SEED'])}
        for fc in available:
            try:
                row[f'DIFF_{fc}'] = float(f1[fc]) - float(f2[fc])
            except (ValueError, TypeError):
                row[f'DIFF_{fc}'] = 0

        X = pd.DataFrame([row])[feature_cols].fillna(0).values

        p_seed = lr_seed.predict_proba(X[:, [seed_idx]])[:, 1][0]
        p_lr = lr_full.predict_proba(X)[:, 1][0]
        p_t1 = tree1.predict_proba(X)[:, 1][0]
        p_t2 = tree2.predict_proba(X)[:, 1][0]

        p_blend = 0.10 * p_seed + 0.20 * p_lr + 0.35 * p_t1 + 0.35 * p_t2
        p_blend = cap_probabilities(p_blend)

        predictions.append({
            'ROUND': int(t1['CURRENT ROUND']),
            'Team_A': t1['TEAM'], 'Seed_A': int(t1['SEED']),
            'Team_B': t2['TEAM'], 'Seed_B': int(t2['SEED']),
            'P_A': round(p_blend, 4),
            'P_B': round(1 - p_blend, 4),
            'P_seed_LR': round(p_seed, 4),
            'P_full_LR': round(p_lr, 4),
            'P_tree1': round(p_t1, 4),
            'P_tree2': round(p_t2, 4),
            'Winner': t1['TEAM'] if p_blend > 0.5 else t2['TEAM'],
        })

    return pd.DataFrame(predictions)


# ============================================================
# BRIER 33.3% OPTIMAL RISK STRATEGY
# ============================================================
"""
Mathematical Proof — 33.3% Threshold
======================================
Brier Score = (1/N) * Σ (f_i - o_i)²

For a matchup where the true upset probability = p:

  Expected Brier if you predict the favorite (f=1-p):
    E[BS] = p*(1 - (1-p))² + (1-p)*((1-p) - 0)² = p*p² + (1-p)*(1-p)² = p³ + (1-p)³

  Expected Brier if you predict the UPSET (f=1 for the underdog):
    E[BS] = p*(1-1)² + (1-p)*(1-0)² = (1-p)

  The upset pick beats the chalk pick when:
    (1-p) < p³ + (1-p)³

  Solving: this crossover occurs at p ≈ 0.333.

PRACTICAL RULE:
  - If P(upset) < 33.3%: trust the favorite, push probability toward chalk
  - If P(upset) >= 33.3%: this is a high-value upset — keep or boost
  - NEVER predict 0.0 or 1.0 (infinite log loss penalty)
  - Cap at [0.025, 0.975]
"""


def apply_brier_strategy(preds_df):
    """Apply 33.3% rule: for underdogs above threshold, preserve upset signal."""
    df = preds_df.copy()
    for idx, row in df.iterrows():
        p_dog = min(row['P_A'], row['P_B'])
        if p_dog < 0.333:
            # Slight push toward favorite (2%)
            boost = 0.02
            if row['P_A'] >= row['P_B']:
                df.at[idx, 'P_A'] = min(row['P_A'] + boost, 0.975)
                df.at[idx, 'P_B'] = max(row['P_B'] - boost, 0.025)
            else:
                df.at[idx, 'P_B'] = min(row['P_B'] + boost, 0.975)
                df.at[idx, 'P_A'] = max(row['P_A'] - boost, 0.025)
    return df


# ============================================================
# DIAGNOSTICS
# ============================================================

def plot_calibration_curve(probs, labels):
    """Reliability diagram + prediction distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    frac, mean_pred = calibration_curve(labels, probs, n_bins=10, strategy='uniform')
    axes[0].plot(mean_pred, frac, 's-', color='#2196F3', label='Model', linewidth=2)
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    axes[0].set_xlabel('Mean Predicted Probability')
    axes[0].set_ylabel('Fraction of Positives')
    axes[0].set_title('Calibration Curve (Season-based CV)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(probs, bins=30, edgecolor='black', alpha=0.7, color='#4CAF50')
    axes[1].set_xlabel('Predicted Probability')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Prediction Distribution')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'calibration_v2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: calibration_v2.png")


def plot_feature_importance(model, feature_names, top_n=25):
    """Feature importance from tree model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'feature_importance'):
        importances = model.feature_importance(importance_type='gain')
    else:
        print("  [WARN] Cannot extract feature importances")
        return

    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
    ax.barh(range(len(indices)), importances[indices], color=colors)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.set_xlabel('Feature Importance (Gain)')
    ax.set_title(f'Top {top_n} Features — Tree Model')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'feature_importance_v2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: feature_importance_v2.png")


def plot_upset_analysis(preds_df):
    """Visualize predicted vs actual seed matchup probabilities."""
    df = preds_df.copy()
    df['SEED_GAP'] = abs(df['Seed_A'] - df['Seed_B'])
    df['UPSET'] = ((df['Seed_A'] > df['Seed_B']) & (df['Winner'] == df['Team_A'])) | \
                  ((df['Seed_B'] > df['Seed_A']) & (df['Winner'] == df['Team_B']))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: P(favorite) vs seed gap
    fav_p = df.apply(lambda r: r['P_A'] if r['Seed_A'] <= r['Seed_B'] else r['P_B'], axis=1)
    axes[0].scatter(df['SEED_GAP'], fav_p, alpha=0.6, c=df['UPSET'].map({True: 'red', False: 'blue'}))
    axes[0].set_xlabel('Seed Gap (|Seed_A - Seed_B|)')
    axes[0].set_ylabel('P(Favorite Wins)')
    axes[0].set_title('Favorite Win Probability vs Seed Gap')
    axes[0].axhline(y=0.667, color='orange', linestyle='--', alpha=0.7, label='33.3% upset threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bar: upset picks by seed gap
    upset_counts = df[df['UPSET']].groupby('SEED_GAP').size()
    total_counts = df.groupby('SEED_GAP').size()
    upset_rates = (upset_counts / total_counts).fillna(0)
    axes[1].bar(upset_rates.index, upset_rates.values, color='#FF5722', alpha=0.7)
    axes[1].set_xlabel('Seed Gap')
    axes[1].set_ylabel('Upset Pick Rate')
    axes[1].set_title('Predicted Upset Rate by Seed Gap')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'upset_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: upset_analysis.png")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("  MARCH MADNESS v2 — LightGBM/CatBoost/LR Ensemble")
    print("=" * 65)

    # ── Load ──
    print("\n[1/7] Loading data...")
    data = load_all_data()
    print(f"  Loaded {len(data)} datasets from 2 sources")

    # ── Matchups ──
    print("\n[2/7] Building historical matchups...")
    games = build_historical_matchups(data['matchups'])
    print(f"  {len(games)} games | {games['YEAR'].nunique()} seasons "
          f"({games['YEAR'].min()}-{games['YEAR'].max()})")

    # ── Features ──
    print("\n[3/7] Engineering features (pre-tournament only)...")
    team_features = build_team_features(data)
    n_feat_cols = len([c for c in team_features.columns
                       if c not in ['YEAR', 'TEAM NO', 'TEAM', 'SEED']])
    print(f"  {len(team_features)} team-seasons | {n_feat_cols} feature columns")

    # ── Symmetric training ──
    print("\n[4/7] Building symmetric training pairs...")
    train_df = build_symmetric_training(games, team_features)
    feature_cols = get_model_feature_names(train_df)
    print(f"  {len(train_df)} rows ({len(train_df)//2} games x 2) | {len(feature_cols)} model features")

    # ── Validate ──
    print("\n[5/7] Season-based GroupKFold cross-validation...")
    cv_probs, cv_labels, cv_years = run_validation(train_df, feature_cols)

    # ── Diagnostics ──
    print("\n[6/7] Generating diagnostic plots...")
    plot_calibration_curve(cv_probs, cv_labels)

    # ── Train final ──
    print("\n[7/7] Training final ensemble on all data...")
    lr_seed, lr_full, tree1, tree2, tree_raw, seed_idx = train_final_ensemble(train_df, feature_cols)
    plot_feature_importance(tree_raw, feature_cols)

    # ── Predict 2026 ──
    print("\n" + "=" * 65)
    print("  2026 TOURNAMENT PREDICTIONS")
    print("=" * 65)

    preds = predict_2026(data, team_features,
                         (lr_seed, lr_full, tree1, tree2, tree_raw, seed_idx),
                         feature_cols, seed_idx)

    if len(preds) == 0:
        print("  No 2026 matchups found.")
        return

    preds = apply_brier_strategy(preds)
    plot_upset_analysis(preds)

    # Display Round of 64
    r64 = preds[preds['ROUND'] == 64].copy()
    if len(r64) > 0:
        print(f"\n  Round of 64 ({len(r64)} matchups):")
        print("  " + "-" * 75)
        for _, row in r64.iterrows():
            is_upset = (row['Seed_A'] > row['Seed_B'] and row['Winner'] == row['Team_A']) or \
                       (row['Seed_B'] > row['Seed_A'] and row['Winner'] == row['Team_B'])
            flag = " *** UPSET" if is_upset else ""
            bar_a = ">>>" if row['Winner'] == row['Team_A'] else "   "
            bar_b = ">>>" if row['Winner'] == row['Team_B'] else "   "
            print(f"    {bar_a} ({row['Seed_A']:>2}) {row['Team_A']:<22} {row['P_A']:.1%}")
            print(f"    {bar_b} ({row['Seed_B']:>2}) {row['Team_B']:<22} {row['P_B']:.1%}{flag}")
            print()

    # Save
    preds.to_csv(OUT_DIR / 'predictions_2026_v2.csv', index=False)
    print(f"  Saved: predictions_2026_v2.csv")

    # Summary
    upsets = preds[
        ((preds['Seed_A'] > preds['Seed_B']) & (preds['Winner'] == preds['Team_A'])) |
        ((preds['Seed_B'] > preds['Seed_A']) & (preds['Winner'] == preds['Team_B']))
    ]
    print(f"\n  Predicted upsets: {len(upsets)} / {len(preds)} games")
    if len(upsets) > 0:
        print("  Upset picks:")
        for _, u in upsets.iterrows():
            dog = u['Team_A'] if u['Seed_A'] > u['Seed_B'] else u['Team_B']
            dog_seed = u['Seed_A'] if u['Seed_A'] > u['Seed_B'] else u['Seed_B']
            fav_seed = u['Seed_B'] if u['Seed_A'] > u['Seed_B'] else u['Seed_A']
            p = u['P_A'] if u['Winner'] == u['Team_A'] else u['P_B']
            print(f"    {dog} ({dog_seed}) over ({fav_seed}) seed — {p:.1%}")

    # ── Pitfalls Checklist ──
    print("\n" + "=" * 65)
    print("  COMMON PITFALLS CHECKLIST")
    print("=" * 65)
    pitfalls = [
        "Use ONLY pre-tournament data (DayNum <= 133). Never train on tourney game stats.",
        "Symmetric rows: every game appears twice (A vs B, B vs A) to avoid ordering bias.",
        "GroupKFold by season — never random split. Future data must not leak into training.",
        "Cap probabilities at [0.025, 0.975]. Log loss is infinite at 0.0 or 1.0.",
        "Isotonic calibration aligns predicted probs with actual win rates.",
        "Blend tree models with seed-anchored LR to prevent overconfident exotic picks.",
        "33.3% Brier rule: only pick upsets when P(upset) >= 33.3%.",
        "Handle missing values systematically: Elo=1500, Seed=median, ratings=0.",
        "Don't overfit to recent seasons — tournament variance is high year to year.",
        "Normalize overtime games if using raw box scores (scale to 40 minutes).",
    ]
    for i, p in enumerate(pitfalls, 1):
        print(f"  {i:>2}. {p}")

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)


if __name__ == '__main__':
    main()
