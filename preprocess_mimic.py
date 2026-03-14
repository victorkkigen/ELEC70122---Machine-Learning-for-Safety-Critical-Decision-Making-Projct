"""
MIMIC-III Demo Preprocessing for Concept-Based OPE
===================================================

Run this script in the same directory as your MIMIC-III Demo CSV files.

Usage:
    python preprocess_mimic.py

Output:
    - mimic_trajectories.npz (processed trajectories for OPE)
    - mimic_summary.txt (dataset statistics)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
import json

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MIMICConfig:
    """Configuration for MIMIC data processing"""
    data_dir: str = "./mimic-iii-clinical-database-demo-1.4"  # MIMIC data folder
    output_dir: str = "."  # Output to current directory
    time_bin_hours: float = 4.0
    max_trajectory_length: int = 20
    min_trajectory_length: int = 3
    
    # Action space bins (as in Komorowski et al.)
    n_fluid_bins: int = 5
    n_vaso_bins: int = 5
    
    # Feature names (15 concepts as in Majumdar et al.)
    feature_names: List[str] = None
    
    # Vital sign normalization ranges
    vital_ranges: Dict = None
    
    def __post_init__(self):
        if self.feature_names is None:
            self.feature_names = [
                'creatinine', 'fio2', 'lactate', 'pao2', 'paco2',
                'urine', 'gcs', 'calcium', 'chloride', 'glucose',
                'hco3', 'magnesium', 'potassium', 'sodium', 'spo2'
            ]
        
        if self.vital_ranges is None:
            self.vital_ranges = {
                'map': (40, 130),
                'hr': (40, 180),
                'spo2': (70, 100),
                'fio2': (21, 100),
                'gcs': (3, 15),
                'urine': (0, 500),
                'temp': (35, 40),
                'rr': (8, 40),
                'creatinine': (0.3, 10),
                'lactate': (0.5, 15),
                'pao2': (50, 500),
                'paco2': (20, 80),
                'calcium': (6, 12),
                'chloride': (90, 120),
                'glucose': (50, 400),
                'hco3': (10, 40),
                'magnesium': (1, 4),
                'potassium': (2.5, 6.5),
                'sodium': (125, 155),
            }


# ============================================================================
# Item IDs
# ============================================================================

# Chart events (vital signs) - MetaVision and CareVue item IDs
CHART_ITEMS = {
    'map': [220052, 220181, 225312, 52, 456, 6702, 443, 224322],
    'hr': [220045, 211],
    'spo2': [220277, 646, 220227],
    'fio2': [223835, 3420, 190, 3422],
    'gcs': [220739, 198],
    'gcs_eye': [220739, 184],
    'gcs_verbal': [223900, 723],
    'gcs_motor': [223901, 454],
    'temp': [223761, 678, 223762, 676],
    'rr': [220210, 618, 615, 224690],
}

# Lab events
LAB_ITEMS = {
    'creatinine': [50912],
    'lactate': [50813],
    'pao2': [50821],
    'paco2': [50818],
    'calcium': [50893],
    'chloride': [50902, 50806],
    'glucose': [50931, 50809],
    'hco3': [50882, 50803],
    'magnesium': [50960],
    'potassium': [50971, 50822],
    'sodium': [50983, 50824],
}

# Output events (urine)
URINE_ITEMS = [40055, 43175, 40069, 40094, 40715, 40473, 40085, 40057, 40056, 
               227488, 227489, 226559, 226560, 226561, 226584, 226563, 226564,
               226565, 226567, 226557, 226558]

# Vasopressors
VASOPRESSOR_ITEMS = {
    'norepinephrine': [221906, 30047, 30120],
    'epinephrine': [221289, 30044, 30119],
    'dopamine': [221662, 30043, 30307],
    'vasopressin': [222315, 30051],
    'phenylephrine': [221749, 30127, 30128],
}

# IV Fluids
IV_FLUID_ITEMS = [225158, 225828, 225159, 220949, 220950, 220952,
                  30008, 30009, 30181, 30021, 30018, 30020, 30015, 30060, 30061]

ALL_CHART_ITEM_IDS = []
for items in CHART_ITEMS.values():
    ALL_CHART_ITEM_IDS.extend(items)
ALL_CHART_ITEM_IDS = list(set(ALL_CHART_ITEM_IDS))


# ============================================================================
# Data Loading
# ============================================================================

def load_tables(data_dir: str, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Load all MIMIC tables."""
    data_dir = Path(data_dir)
    tables = {}
    
    files_to_load = [
        ('icustays', 'ICUSTAYS.csv', True),
        ('admissions', 'ADMISSIONS.csv', True),
        ('patients', 'PATIENTS.csv', False),
        ('labevents', 'LABEVENTS.csv', True),
        ('chartevents', 'CHARTEVENTS.csv', False),
        ('outputevents', 'OUTPUTEVENTS.csv', False),
        ('inputevents_mv', 'INPUTEVENTS_MV.csv', False),
        ('inputevents_cv', 'INPUTEVENTS_CV.csv', False),
        ('d_items', 'D_ITEMS.csv', False),
        ('d_labitems', 'D_LABITEMS.csv', False),
    ]
    
    for name, filename, required in files_to_load:
        path = data_dir / filename
        if path.exists():
            if verbose:
                print(f"Loading {filename}...", end=" ")
            df = pd.read_csv(path)
            tables[name] = df
            if verbose:
                print(f"{len(df):,} rows")
        elif required:
            raise FileNotFoundError(f"Required file not found: {path}")
        else:
            if verbose:
                print(f"[!] Optional file not found: {filename}")
            tables[name] = None
    
    return tables


def filter_chartevents(chartevents: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Filter chartevents to only vital signs we need."""
    if chartevents is None:
        return None
    
    original_len = len(chartevents)
    filtered = chartevents[chartevents['itemid'].isin(ALL_CHART_ITEM_IDS)].copy()
    
    if verbose:
        print(f"Filtered CHARTEVENTS: {original_len:,} -> {len(filtered):,} rows")
    
    return filtered


# ============================================================================
# Trajectory Building
# ============================================================================

def normalize_vital(value: float, vital_name: str, config: MIMICConfig) -> float:
    """Normalize vital sign to [0, 1] range."""
    if vital_name not in config.vital_ranges:
        return value
    vmin, vmax = config.vital_ranges[vital_name]
    normalized = (value - vmin) / (vmax - vmin)
    return np.clip(normalized, 0, 1)


def bin_to_concept(value: float, n_bins: int = 10) -> int:
    """Convert normalized value to discrete concept (0-9)."""
    if np.isnan(value):
        return 5  # Default to middle
    binned = int(value * n_bins)
    return np.clip(binned, 0, n_bins - 1)


def extract_values(events_df: pd.DataFrame, 
                   id_col: str, 
                   id_val: int,
                   item_ids: List[int],
                   time_col: str = 'charttime',
                   value_col: str = 'valuenum') -> pd.DataFrame:
    """Extract values for specific items."""
    if events_df is None:
        return pd.DataFrame()
    
    mask = (events_df[id_col] == id_val) & (events_df['itemid'].isin(item_ids))
    
    cols = [time_col, value_col]
    cols = [c for c in cols if c in events_df.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    
    subset = events_df.loc[mask, cols].copy()
    subset = subset.dropna()
    subset.columns = ['time', 'value']
    
    return subset


def bin_time(df: pd.DataFrame, 
             icu_intime: pd.Timestamp, 
             bin_hours: float) -> pd.DataFrame:
    """Assign time bins relative to ICU admission."""
    if df.empty:
        return df
    
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df['hours'] = (df['time'] - icu_intime).dt.total_seconds() / 3600
    df['time_bin'] = (df['hours'] / bin_hours).astype(int)
    df = df[df['time_bin'] >= 0]
    
    return df


def aggregate_by_bin(df: pd.DataFrame) -> Dict[int, float]:
    """Aggregate values by time bin (median)."""
    if df.empty:
        return {}
    return df.groupby('time_bin')['value'].median().to_dict()


def compute_action(fluid_amount: float, vaso_amount: float, config: MIMICConfig) -> int:
    """Compute discrete action from fluid and vasopressor amounts."""
    # Fluid bins (ml per time bin)
    fluid_bins = [0, 100, 250, 500, 1000, np.inf]
    fluid_bin = np.digitize(fluid_amount, fluid_bins) - 1
    fluid_bin = np.clip(fluid_bin, 0, config.n_fluid_bins - 1)
    
    # Vasopressor bins (normalized dose)
    vaso_bins = [0, 0.05, 0.1, 0.2, 0.5, np.inf]
    vaso_bin = np.digitize(vaso_amount, vaso_bins) - 1
    vaso_bin = np.clip(vaso_bin, 0, config.n_vaso_bins - 1)
    
    return int(fluid_bin * config.n_vaso_bins + vaso_bin)


def process_patient(icustay_id: int, 
                    tables: Dict[str, pd.DataFrame],
                    config: MIMICConfig) -> Optional[Dict]:
    """Process a single ICU stay into a trajectory."""
    
    icustays = tables['icustays']
    admissions = tables['admissions']
    labevents = tables['labevents']
    chartevents = tables['chartevents']
    outputevents = tables['outputevents']
    inputevents_mv = tables['inputevents_mv']
    inputevents_cv = tables['inputevents_cv']
    
    # Get ICU stay info
    stay = icustays[icustays['icustay_id'] == icustay_id]
    if stay.empty:
        return None
    stay = stay.iloc[0]
    
    hadm_id = stay['hadm_id']
    subject_id = stay['subject_id']
    
    # Get admission info for mortality
    adm = admissions[admissions['hadm_id'] == hadm_id]
    if adm.empty:
        return None
    adm = adm.iloc[0]
    died = adm['hospital_expire_flag'] == 1
    
    # Parse times
    icu_intime = pd.to_datetime(stay['intime'])
    icu_outtime = pd.to_datetime(stay['outtime'])
    los_hours = (icu_outtime - icu_intime).total_seconds() / 3600
    
    # Calculate number of time bins
    n_bins = min(
        int(los_hours / config.time_bin_hours) + 1,
        config.max_trajectory_length
    )
    
    if n_bins < config.min_trajectory_length:
        return None
    
    # Initialize arrays
    n_features = len(config.feature_names)
    states = np.full((n_bins, n_features), 0.5)  # Default to middle
    raw_states = {name: {} for name in config.feature_names}
    actions = np.zeros(n_bins, dtype=int)
    rewards = np.zeros(n_bins)
    
    # Feature index mapping
    feat_idx = {name: i for i, name in enumerate(config.feature_names)}
    
    # --- Extract LAB values ---
    lab_features = ['creatinine', 'lactate', 'pao2', 'paco2', 'calcium', 
                    'chloride', 'glucose', 'hco3', 'magnesium', 'potassium', 'sodium']
    
    for feat in lab_features:
        if feat in LAB_ITEMS and feat in feat_idx:
            vals_df = extract_values(labevents, 'hadm_id', hadm_id, 
                                     LAB_ITEMS[feat], 'charttime', 'valuenum')
            if not vals_df.empty:
                vals_df = bin_time(vals_df, icu_intime, config.time_bin_hours)
                binned = aggregate_by_bin(vals_df)
                raw_states[feat] = binned
                for t, val in binned.items():
                    if 0 <= t < n_bins:
                        states[t, feat_idx[feat]] = normalize_vital(val, feat, config)
    
    # --- Extract CHART values ---
    if chartevents is not None:
        chart_features = ['fio2', 'spo2', 'gcs']
        chart_mapping = {
            'fio2': CHART_ITEMS.get('fio2', []),
            'spo2': CHART_ITEMS.get('spo2', []),
            'gcs': CHART_ITEMS.get('gcs', []) + CHART_ITEMS.get('gcs_eye', []) + 
                   CHART_ITEMS.get('gcs_verbal', []) + CHART_ITEMS.get('gcs_motor', []),
        }
        
        for feat in chart_features:
            if feat in feat_idx and feat in chart_mapping:
                vals_df = extract_values(chartevents, 'icustay_id', icustay_id,
                                        chart_mapping[feat], 'charttime', 'valuenum')
                if not vals_df.empty:
                    vals_df = bin_time(vals_df, icu_intime, config.time_bin_hours)
                    binned = aggregate_by_bin(vals_df)
                    raw_states[feat] = binned
                    for t, val in binned.items():
                        if 0 <= t < n_bins:
                            states[t, feat_idx[feat]] = normalize_vital(val, feat, config)
    
    # --- Extract URINE output ---
    if outputevents is not None and 'urine' in feat_idx:
        vals_df = extract_values(outputevents, 'icustay_id', icustay_id,
                                URINE_ITEMS, 'charttime', 'value')
        if not vals_df.empty:
            vals_df = bin_time(vals_df, icu_intime, config.time_bin_hours)
            # Sum urine output per bin (not median)
            binned = vals_df.groupby('time_bin')['value'].sum().to_dict()
            raw_states['urine'] = binned
            for t, val in binned.items():
                if 0 <= t < n_bins:
                    states[t, feat_idx['urine']] = normalize_vital(val, 'urine', config)
    
    # --- Extract TREATMENTS and compute actions ---
    fluid_per_bin = np.zeros(n_bins)
    vaso_per_bin = np.zeros(n_bins)
    
    # MetaVision inputs
    if inputevents_mv is not None:
        mv_subset = inputevents_mv[inputevents_mv['icustay_id'] == icustay_id].copy()
        if not mv_subset.empty:
            mv_subset['starttime'] = pd.to_datetime(mv_subset['starttime'])
            mv_subset['time_bin'] = ((mv_subset['starttime'] - icu_intime).dt.total_seconds() / 3600 / config.time_bin_hours).astype(int)
            
            # IV Fluids
            fluid_mask = mv_subset['itemid'].isin(IV_FLUID_ITEMS)
            for _, row in mv_subset[fluid_mask].iterrows():
                t = row['time_bin']
                if 0 <= t < n_bins:
                    fluid_per_bin[t] += row.get('amount', 0) or 0
            
            # Vasopressors
            all_vaso_items = []
            for items in VASOPRESSOR_ITEMS.values():
                all_vaso_items.extend(items)
            vaso_mask = mv_subset['itemid'].isin(all_vaso_items)
            for _, row in mv_subset[vaso_mask].iterrows():
                t = row['time_bin']
                if 0 <= t < n_bins:
                    vaso_per_bin[t] += row.get('rate', 0) or 0
    
    # CareVue inputs
    if inputevents_cv is not None:
        cv_subset = inputevents_cv[inputevents_cv['icustay_id'] == icustay_id].copy()
        if not cv_subset.empty:
            cv_subset['charttime'] = pd.to_datetime(cv_subset['charttime'])
            cv_subset['time_bin'] = ((cv_subset['charttime'] - icu_intime).dt.total_seconds() / 3600 / config.time_bin_hours).astype(int)
            
            # IV Fluids
            fluid_mask = cv_subset['itemid'].isin(IV_FLUID_ITEMS)
            for _, row in cv_subset[fluid_mask].iterrows():
                t = row['time_bin']
                if 0 <= t < n_bins:
                    fluid_per_bin[t] += row.get('amount', 0) or 0
            
            # Vasopressors
            all_vaso_items = []
            for items in VASOPRESSOR_ITEMS.values():
                all_vaso_items.extend(items)
            vaso_mask = cv_subset['itemid'].isin(all_vaso_items)
            for _, row in cv_subset[vaso_mask].iterrows():
                t = row['time_bin']
                if 0 <= t < n_bins:
                    vaso_per_bin[t] += row.get('rate', 0) or 0
    
    # Compute actions
    for t in range(n_bins):
        actions[t] = compute_action(fluid_per_bin[t], vaso_per_bin[t], config)
    
    # --- Assign rewards ---
    rewards[-1] = -1 if died else +1
    
    # --- Forward-fill missing values ---
    for i in range(n_features):
        last_valid = 0.5
        for t in range(n_bins):
            if states[t, i] == 0.5 and t > 0:
                states[t, i] = last_valid
            else:
                last_valid = states[t, i]
    
    return {
        'icustay_id': int(icustay_id),
        'hadm_id': int(hadm_id),
        'subject_id': int(subject_id),
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'raw_states': raw_states,
        'died': bool(died),
        'los_hours': float(los_hours),
        'n_bins': int(n_bins),
    }


def build_all_trajectories(tables: Dict[str, pd.DataFrame],
                           config: MIMICConfig,
                           verbose: bool = True) -> List[Dict]:
    """Build trajectories for all ICU stays."""
    
    # Parse datetime columns
    tables['icustays']['intime'] = pd.to_datetime(tables['icustays']['intime'])
    tables['icustays']['outtime'] = pd.to_datetime(tables['icustays']['outtime'])
    
    # Filter chartevents to vitals only
    if tables['chartevents'] is not None:
        tables['chartevents'] = filter_chartevents(tables['chartevents'], verbose)
    
    trajectories = []
    icustay_ids = tables['icustays']['icustay_id'].unique()
    n_total = len(icustay_ids)
    
    if verbose:
        print(f"\nProcessing {n_total} ICU stays...")
    
    for i, icustay_id in enumerate(icustay_ids):
        if verbose and (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{n_total}")
        
        try:
            traj = process_patient(icustay_id, tables, config)
            if traj is not None:
                trajectories.append(traj)
        except Exception as e:
            if verbose:
                print(f"  Error processing {icustay_id}: {e}")
    
    if verbose:
        print(f"\nBuilt {len(trajectories)} trajectories")
        if trajectories:
            mortality = 100 * sum(t['died'] for t in trajectories) / len(trajectories)
            mean_len = np.mean([t['n_bins'] for t in trajectories])
            print(f"  Mortality rate: {mortality:.1f}%")
            print(f"  Mean trajectory length: {mean_len:.1f} time bins")
    
    return trajectories


# ============================================================================
# Concept Extraction
# ============================================================================

def extract_hard_concepts(states: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Convert continuous states to hard (discrete) concepts.
    Each feature is binned into n_bins discrete levels (0 to n_bins-1).
    """
    hard_concepts = np.zeros_like(states, dtype=int)
    for i in range(states.shape[1]):
        hard_concepts[:, i] = np.clip((states[:, i] * n_bins).astype(int), 0, n_bins - 1)
    return hard_concepts


def create_concept_index(hard_concepts: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Create a single concept index from multi-dimensional hard concepts.
    Uses first 5 features to create index (to keep cardinality manageable).
    """
    n_features_for_index = min(5, hard_concepts.shape[1])
    indices = np.zeros(hard_concepts.shape[0], dtype=int)
    
    for i in range(n_features_for_index):
        indices += hard_concepts[:, i] * (n_bins ** i)
    
    return indices


# ============================================================================
# Save/Load
# ============================================================================

def save_trajectories(trajectories: List[Dict], 
                      output_path: str,
                      config: MIMICConfig,
                      verbose: bool = True):
    """Save trajectories to NPZ file."""
    
    # Convert to arrays
    all_states = []
    all_actions = []
    all_rewards = []
    all_died = []
    all_lengths = []
    all_icustay_ids = []
    
    for traj in trajectories:
        all_states.append(traj['states'])
        all_actions.append(traj['actions'])
        all_rewards.append(traj['rewards'])
        all_died.append(traj['died'])
        all_lengths.append(traj['n_bins'])
        all_icustay_ids.append(traj['icustay_id'])
    
    # Pad to same length
    max_len = max(all_lengths)
    n_features = all_states[0].shape[1]
    
    states_padded = np.zeros((len(trajectories), max_len, n_features))
    actions_padded = np.zeros((len(trajectories), max_len), dtype=int)
    rewards_padded = np.zeros((len(trajectories), max_len))
    
    for i, (s, a, r, l) in enumerate(zip(all_states, all_actions, all_rewards, all_lengths)):
        states_padded[i, :l] = s
        actions_padded[i, :l] = a
        rewards_padded[i, :l] = r
    
    # Save
    np.savez(
        output_path,
        states=states_padded,
        actions=actions_padded,
        rewards=rewards_padded,
        lengths=np.array(all_lengths),
        died=np.array(all_died),
        icustay_ids=np.array(all_icustay_ids),
        feature_names=np.array(config.feature_names),
        n_actions=config.n_fluid_bins * config.n_vaso_bins,
    )
    
    if verbose:
        print(f"\nSaved to {output_path}")
        print(f"  States shape: {states_padded.shape}")
        print(f"  Actions shape: {actions_padded.shape}")


def save_summary(trajectories: List[Dict], 
                 output_path: str,
                 config: MIMICConfig):
    """Save dataset summary."""
    
    summary = {
        'n_trajectories': len(trajectories),
        'mortality_rate': sum(t['died'] for t in trajectories) / len(trajectories),
        'mean_length': np.mean([t['n_bins'] for t in trajectories]),
        'max_length': max(t['n_bins'] for t in trajectories),
        'min_length': min(t['n_bins'] for t in trajectories),
        'n_features': len(config.feature_names),
        'feature_names': config.feature_names,
        'n_actions': config.n_fluid_bins * config.n_vaso_bins,
        'time_bin_hours': config.time_bin_hours,
    }
    
    # Feature coverage
    feature_coverage = {name: 0 for name in config.feature_names}
    for traj in trajectories:
        for name, vals in traj['raw_states'].items():
            if vals:
                feature_coverage[name] += 1
    
    summary['feature_coverage'] = {
        name: count / len(trajectories) 
        for name, count in feature_coverage.items()
    }
    
    with open(output_path, 'w') as f:
        f.write("MIMIC-III Demo Dataset Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of trajectories: {summary['n_trajectories']}\n")
        f.write(f"Mortality rate: {summary['mortality_rate']*100:.1f}%\n")
        f.write(f"Mean trajectory length: {summary['mean_length']:.1f} time bins\n")
        f.write(f"Time bin size: {summary['time_bin_hours']} hours\n")
        f.write(f"Number of features: {summary['n_features']}\n")
        f.write(f"Number of actions: {summary['n_actions']}\n")
        f.write("\nFeature coverage (% of trajectories with data):\n")
        for name, coverage in summary['feature_coverage'].items():
            f.write(f"  {name}: {coverage*100:.1f}%\n")
    
    print(f"Saved summary to {output_path}")
    
    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("MIMIC-III Demo Preprocessing for Concept-Based OPE")
    print("=" * 60)
    
    config = MIMICConfig()
    
    # Load data
    print("\n[1/4] Loading tables...")
    tables = load_tables(config.data_dir, verbose=True)
    
    # Build trajectories
    print("\n[2/4] Building trajectories...")
    trajectories = build_all_trajectories(tables, config, verbose=True)
    
    if not trajectories:
        print("\nERROR: No trajectories built. Check your data files.")
        return
    
    # Save trajectories
    print("\n[3/4] Saving trajectories...")
    save_trajectories(trajectories, 'mimic_trajectories.npz', config, verbose=True)
    
    # Save summary
    print("\n[4/4] Saving summary...")
    summary = save_summary(trajectories, 'mimic_summary.txt', config)
    
    print("\n" + "=" * 60)
    print("DONE! Files created:")
    print("  - mimic_trajectories.npz (upload this to Claude)")
    print("  - mimic_summary.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()