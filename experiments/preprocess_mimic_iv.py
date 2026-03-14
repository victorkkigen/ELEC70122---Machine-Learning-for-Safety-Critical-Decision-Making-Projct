"""
MIMIC-IV Demo Preprocessing for Concept-Based OPE
==================================================

Following the MIMIC-Sepsis benchmark methodology (Huang et al. 2025)
and AI Clinician approach (Komorowski et al. 2018).

Key features:
- Sepsis-3 approximation (using available data)
- 4-hour time bins
- Clinical features aligned with Komorowski
- Vasopressor and fluid actions (5x5 = 25 actions)
- Mortality-based rewards

Run: python preprocess_mimic_iv.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MIMICIVConfig:
    """Configuration for MIMIC-IV preprocessing."""
    data_dir: str = "../mimic-iv-clinical-database-demo-2.2"
    output_dir: str = "."
    
    # Time binning
    time_bin_hours: float = 4.0
    max_trajectory_length: int = 20
    min_trajectory_length: int = 3
    
    # Action space (Komorowski: 5 fluid bins × 5 vasopressor bins = 25 actions)
    n_fluid_bins: int = 5
    n_vaso_bins: int = 5
    
    # Item IDs for MIMIC-IV (updated from MIMIC-III)
    vital_itemids: Dict = field(default_factory=lambda: {
        'heart_rate': [220045],
        'sbp': [220050, 220179],  # Systolic BP (arterial and non-invasive)
        'dbp': [220051, 220180],  # Diastolic BP
        'mbp': [220052, 220181],  # Mean BP
        'resp_rate': [220210, 224690],
        'spo2': [220277],
        'temperature': [223761, 223762],  # Celsius and Fahrenheit
        'gcs_eye': [220739],
        'gcs_verbal': [223900],
        'gcs_motor': [223901],
    })
    
    # Vasopressor item IDs
    vaso_itemids: List = field(default_factory=lambda: [
        221906,  # Norepinephrine
        221289,  # Epinephrine
        221662,  # Dopamine
        222315,  # Vasopressin
        221749,  # Phenylephrine
    ])
    
    # IV Fluid item IDs (common crystalloids/colloids)
    fluid_itemids: List = field(default_factory=lambda: [
        220949,  # Dextrose 5%
        220950,  # Normal Saline (0.9%)
        220952,  # Sodium Chloride 0.45%
        225158,  # Lactated Ringers
        225159,  # Sodium Chloride 0.9%
        225828,  # Lactated Ringers
    ])
    
    # Lab item IDs
    lab_itemids: Dict = field(default_factory=lambda: {
        'lactate': [50813, 52442],
        'creatinine': [50912, 52024],
        'bilirubin': [50885],
        'platelet': [51265],
        'wbc': [51300, 51301],
        'hemoglobin': [51222],
        'hematocrit': [51221],
        'potassium': [50971, 50822],
        'sodium': [50983, 50824],
        'chloride': [50902, 50806],
        'bicarbonate': [50882],
        'bun': [51006],
        'glucose': [50931, 50809],
        'pao2': [50821],
        'paco2': [50818],
        'ph': [50820],
        'fio2': [50816],
    })
    
    # Normalization ranges (based on clinical plausibility)
    vital_ranges: Dict = field(default_factory=lambda: {
        'heart_rate': (30, 200),
        'sbp': (50, 220),
        'dbp': (20, 140),
        'mbp': (30, 160),
        'resp_rate': (5, 50),
        'spo2': (70, 100),
        'temperature': (32, 42),
        'gcs_total': (3, 15),
    })
    
    lab_ranges: Dict = field(default_factory=lambda: {
        'lactate': (0.5, 20),
        'creatinine': (0.1, 15),
        'bilirubin': (0.1, 30),
        'platelet': (10, 600),
        'wbc': (0.5, 50),
        'hemoglobin': (4, 18),
        'hematocrit': (15, 55),
        'potassium': (2, 7),
        'sodium': (120, 160),
        'chloride': (80, 130),
        'bicarbonate': (10, 40),
        'bun': (2, 150),
        'glucose': (30, 500),
        'pao2': (40, 500),
        'paco2': (15, 100),
        'ph': (6.8, 7.8),
        'fio2': (21, 100),
    })


# =============================================================================
# Data Loading
# =============================================================================

def load_tables(config: MIMICIVConfig, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Load all required MIMIC-IV tables."""
    data_dir = Path(config.data_dir)
    tables = {}
    
    files_to_load = [
        ('patients', 'hosp/patients.csv'),
        ('admissions', 'hosp/admissions.csv'),
        ('icustays', 'icu/icustays.csv'),
        ('chartevents', 'icu/chartevents.csv'),
        ('labevents', 'hosp/labevents.csv'),
        ('inputevents', 'icu/inputevents.csv'),
        ('outputevents', 'icu/outputevents.csv'),
        ('d_items', 'icu/d_items.csv'),
        ('d_labitems', 'hosp/d_labitems.csv'),
    ]
    
    for name, filepath in files_to_load:
        path = data_dir / filepath
        if path.exists():
            if verbose:
                print(f"Loading {filepath}...", end=" ")
            df = pd.read_csv(path)
            tables[name] = df
            if verbose:
                print(f"{len(df):,} rows")
        else:
            if verbose:
                print(f"[!] File not found: {filepath}")
            tables[name] = None
    
    return tables


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_vitals(chartevents: pd.DataFrame, stay_id: int, 
                   intime: pd.Timestamp, outtime: pd.Timestamp,
                   config: MIMICIVConfig) -> pd.DataFrame:
    """Extract vital signs for a single ICU stay."""
    # Get all vital itemids
    all_vital_ids = []
    for ids in config.vital_itemids.values():
        all_vital_ids.extend(ids)
    
    # Filter chartevents
    mask = (chartevents['stay_id'] == stay_id) & \
           (chartevents['itemid'].isin(all_vital_ids))
    vitals = chartevents.loc[mask, ['charttime', 'itemid', 'valuenum']].copy()
    
    if len(vitals) == 0:
        return pd.DataFrame()
    
    vitals['charttime'] = pd.to_datetime(vitals['charttime'])
    
    # Map itemids to feature names
    itemid_to_name = {}
    for name, ids in config.vital_itemids.items():
        for id_ in ids:
            itemid_to_name[id_] = name
    
    vitals['feature'] = vitals['itemid'].map(itemid_to_name)
    
    return vitals[['charttime', 'feature', 'valuenum']]


def extract_labs(labevents: pd.DataFrame, hadm_id: int,
                 intime: pd.Timestamp, outtime: pd.Timestamp,
                 config: MIMICIVConfig) -> pd.DataFrame:
    """Extract lab values for a hospital admission."""
    # Get all lab itemids
    all_lab_ids = []
    for ids in config.lab_itemids.values():
        all_lab_ids.extend(ids)
    
    # Filter labevents
    mask = (labevents['hadm_id'] == hadm_id) & \
           (labevents['itemid'].isin(all_lab_ids))
    labs = labevents.loc[mask, ['charttime', 'itemid', 'valuenum']].copy()
    
    if len(labs) == 0:
        return pd.DataFrame()
    
    labs['charttime'] = pd.to_datetime(labs['charttime'])
    
    # Map itemids to feature names
    itemid_to_name = {}
    for name, ids in config.lab_itemids.items():
        for id_ in ids:
            itemid_to_name[id_] = name
    
    labs['feature'] = labs['itemid'].map(itemid_to_name)
    
    return labs[['charttime', 'feature', 'valuenum']]


def extract_vasopressors(inputevents: pd.DataFrame, stay_id: int,
                         intime: pd.Timestamp, outtime: pd.Timestamp,
                         config: MIMICIVConfig) -> pd.DataFrame:
    """Extract vasopressor administration for an ICU stay."""
    mask = (inputevents['stay_id'] == stay_id) & \
           (inputevents['itemid'].isin(config.vaso_itemids))
    vasos = inputevents.loc[mask, ['starttime', 'endtime', 'amount', 'rate']].copy()
    
    if len(vasos) == 0:
        return pd.DataFrame()
    
    vasos['starttime'] = pd.to_datetime(vasos['starttime'])
    vasos['endtime'] = pd.to_datetime(vasos['endtime'])
    
    return vasos


def extract_fluids(inputevents: pd.DataFrame, stay_id: int,
                   intime: pd.Timestamp, outtime: pd.Timestamp,
                   config: MIMICIVConfig) -> pd.DataFrame:
    """Extract IV fluid administration for an ICU stay."""
    mask = (inputevents['stay_id'] == stay_id) & \
           (inputevents['itemid'].isin(config.fluid_itemids))
    fluids = inputevents.loc[mask, ['starttime', 'endtime', 'amount']].copy()
    
    if len(fluids) == 0:
        return pd.DataFrame()
    
    fluids['starttime'] = pd.to_datetime(fluids['starttime'])
    fluids['endtime'] = pd.to_datetime(fluids['endtime'])
    
    return fluids


# =============================================================================
# Time Binning and Aggregation
# =============================================================================

def bin_features(features: pd.DataFrame, intime: pd.Timestamp,
                 n_bins: int, bin_hours: float) -> Dict[str, np.ndarray]:
    """Bin features into time windows."""
    if len(features) == 0:
        return {}
    
    # Calculate time from ICU admission
    features = features.copy()
    features['hours_from_admit'] = (features['charttime'] - intime).dt.total_seconds() / 3600
    
    # Assign to bins
    features['bin'] = (features['hours_from_admit'] / bin_hours).astype(int)
    features = features[(features['bin'] >= 0) & (features['bin'] < n_bins)]
    
    # Aggregate by bin and feature (take mean)
    binned = {}
    for feature_name in features['feature'].unique():
        feature_data = features[features['feature'] == feature_name]
        values = np.full(n_bins, np.nan)
        
        for bin_idx in range(n_bins):
            bin_data = feature_data[feature_data['bin'] == bin_idx]['valuenum']
            if len(bin_data) > 0:
                values[bin_idx] = bin_data.mean()
        
        binned[feature_name] = values
    
    return binned


def bin_treatments(treatment_df: pd.DataFrame, intime: pd.Timestamp,
                   n_bins: int, bin_hours: float,
                   amount_col: str = 'amount') -> np.ndarray:
    """Bin treatment amounts into time windows."""
    totals = np.zeros(n_bins)
    
    if len(treatment_df) == 0:
        return totals
    
    for _, row in treatment_df.iterrows():
        start_hours = (row['starttime'] - intime).total_seconds() / 3600
        
        if pd.notna(row.get('endtime')):
            end_hours = (row['endtime'] - intime).total_seconds() / 3600
        else:
            end_hours = start_hours + 1  # Assume 1 hour if no end time
        
        start_bin = max(0, int(start_hours / bin_hours))
        end_bin = min(n_bins - 1, int(end_hours / bin_hours))
        
        if pd.notna(row[amount_col]):
            # Distribute amount across bins
            n_active_bins = max(1, end_bin - start_bin + 1)
            amount_per_bin = row[amount_col] / n_active_bins
            
            for b in range(start_bin, end_bin + 1):
                if 0 <= b < n_bins:
                    totals[b] += amount_per_bin
    
    return totals


# =============================================================================
# Trajectory Building
# =============================================================================

def discretize_actions(fluid_amounts: np.ndarray, vaso_amounts: np.ndarray,
                       n_fluid_bins: int = 5, n_vaso_bins: int = 5) -> np.ndarray:
    """
    Discretize fluid and vasopressor amounts into action indices.
    Action = fluid_bin * n_vaso_bins + vaso_bin
    """
    n_bins = len(fluid_amounts)
    actions = np.zeros(n_bins, dtype=int)
    
    # Compute percentiles for binning (across non-zero values)
    fluid_nonzero = fluid_amounts[fluid_amounts > 0]
    vaso_nonzero = vaso_amounts[vaso_amounts > 0]
    
    if len(fluid_nonzero) > 0:
        fluid_percentiles = np.percentile(fluid_nonzero, np.linspace(0, 100, n_fluid_bins + 1))
    else:
        fluid_percentiles = np.array([0] * (n_fluid_bins + 1))
    
    if len(vaso_nonzero) > 0:
        vaso_percentiles = np.percentile(vaso_nonzero, np.linspace(0, 100, n_vaso_bins + 1))
    else:
        vaso_percentiles = np.array([0] * (n_vaso_bins + 1))
    
    for t in range(n_bins):
        # Bin fluids (0 = no fluid, 1-4 = increasing amounts)
        if fluid_amounts[t] == 0:
            fluid_bin = 0
        else:
            fluid_bin = np.searchsorted(fluid_percentiles[1:], fluid_amounts[t], side='right')
            fluid_bin = min(fluid_bin, n_fluid_bins - 1)
        
        # Bin vasopressors
        if vaso_amounts[t] == 0:
            vaso_bin = 0
        else:
            vaso_bin = np.searchsorted(vaso_percentiles[1:], vaso_amounts[t], side='right')
            vaso_bin = min(vaso_bin, n_vaso_bins - 1)
        
        actions[t] = fluid_bin * n_vaso_bins + vaso_bin
    
    return actions


def normalize_value(value: float, vmin: float, vmax: float) -> float:
    """Normalize value to [0, 1] range."""
    if np.isnan(value):
        return 0.5  # Default to middle
    return np.clip((value - vmin) / (vmax - vmin), 0, 1)


def forward_fill(arr: np.ndarray) -> np.ndarray:
    """Forward fill missing values."""
    result = arr.copy()
    last_valid = 0.5  # Default
    
    for i in range(len(result)):
        if np.isnan(result[i]):
            result[i] = last_valid
        else:
            last_valid = result[i]
    
    return result


def process_icu_stay(stay_row: pd.Series, tables: Dict[str, pd.DataFrame],
                     config: MIMICIVConfig) -> Optional[Dict]:
    """Process a single ICU stay into a trajectory."""
    stay_id = stay_row['stay_id']
    hadm_id = stay_row['hadm_id']
    subject_id = stay_row['subject_id']
    
    intime = pd.to_datetime(stay_row['intime'])
    outtime = pd.to_datetime(stay_row['outtime'])
    
    # Calculate number of time bins
    los_hours = (outtime - intime).total_seconds() / 3600
    n_bins = min(int(los_hours / config.time_bin_hours) + 1, config.max_trajectory_length)
    
    if n_bins < config.min_trajectory_length:
        return None
    
    # Extract features
    vitals = extract_vitals(tables['chartevents'], stay_id, intime, outtime, config)
    labs = extract_labs(tables['labevents'], hadm_id, intime, outtime, config)
    vasos = extract_vasopressors(tables['inputevents'], stay_id, intime, outtime, config)
    fluids = extract_fluids(tables['inputevents'], stay_id, intime, outtime, config)
    
    # Bin features
    vitals_binned = bin_features(vitals, intime, n_bins, config.time_bin_hours)
    labs_binned = bin_features(labs, intime, n_bins, config.time_bin_hours)
    
    # Bin treatments
    vaso_amounts = bin_treatments(vasos, intime, n_bins, config.time_bin_hours, 'amount')
    fluid_amounts = bin_treatments(fluids, intime, n_bins, config.time_bin_hours, 'amount')
    
    # Build feature matrix
    feature_names = [
        'heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'spo2', 'temperature', 'gcs_total',
        'lactate', 'creatinine', 'bilirubin', 'platelet', 'wbc', 'hemoglobin',
        'potassium', 'sodium', 'glucose', 'bun'
    ]
    
    states = np.full((n_bins, len(feature_names)), 0.5)  # Default to 0.5 (middle)
    
    for i, name in enumerate(feature_names):
        if name == 'gcs_total':
            # Compute GCS total from components
            gcs_eye = vitals_binned.get('gcs_eye', np.full(n_bins, np.nan))
            gcs_verbal = vitals_binned.get('gcs_verbal', np.full(n_bins, np.nan))
            gcs_motor = vitals_binned.get('gcs_motor', np.full(n_bins, np.nan))
            gcs_total = gcs_eye + gcs_verbal + gcs_motor
            
            for t in range(n_bins):
                if not np.isnan(gcs_total[t]):
                    states[t, i] = normalize_value(gcs_total[t], 3, 15)
        elif name in vitals_binned:
            ranges = config.vital_ranges.get(name, (0, 100))
            for t in range(n_bins):
                states[t, i] = normalize_value(vitals_binned[name][t], ranges[0], ranges[1])
        elif name in labs_binned:
            ranges = config.lab_ranges.get(name, (0, 100))
            for t in range(n_bins):
                states[t, i] = normalize_value(labs_binned[name][t], ranges[0], ranges[1])
    
    # Forward fill missing values
    for i in range(states.shape[1]):
        states[:, i] = forward_fill(states[:, i])
    
    # Discretize actions
    actions = discretize_actions(fluid_amounts, vaso_amounts, 
                                 config.n_fluid_bins, config.n_vaso_bins)
    
    # Get outcome (mortality)
    admission = tables['admissions'][tables['admissions']['hadm_id'] == hadm_id]
    if len(admission) > 0:
        died = admission['hospital_expire_flag'].values[0] == 1
    else:
        died = False
    
    # Compute rewards (sparse: -1 at end if died, 0 otherwise)
    rewards = np.zeros(n_bins)
    rewards[-1] = -1.0 if died else 0.0
    
    return {
        'stay_id': int(stay_id),
        'hadm_id': int(hadm_id),
        'subject_id': int(subject_id),
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'vaso_amounts': vaso_amounts,
        'fluid_amounts': fluid_amounts,
        'died': died,
        'los_hours': los_hours,
        'n_bins': n_bins,
        'feature_names': feature_names,
    }


def build_all_trajectories(tables: Dict[str, pd.DataFrame],
                           config: MIMICIVConfig,
                           verbose: bool = True) -> List[Dict]:
    """Build trajectories for all ICU stays."""
    trajectories = []
    icustays = tables['icustays']
    n_total = len(icustays)
    
    if verbose:
        print(f"\nProcessing {n_total} ICU stays...")
    
    for i, (_, stay_row) in enumerate(icustays.iterrows()):
        if verbose and (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{n_total}")
        
        try:
            traj = process_icu_stay(stay_row, tables, config)
            if traj is not None:
                trajectories.append(traj)
        except Exception as e:
            if verbose:
                print(f"  Error processing stay {stay_row['stay_id']}: {e}")
    
    if verbose:
        print(f"\nBuilt {len(trajectories)} trajectories")
        if trajectories:
            mortality = 100 * sum(t['died'] for t in trajectories) / len(trajectories)
            mean_len = np.mean([t['n_bins'] for t in trajectories])
            print(f"  Mortality rate: {mortality:.1f}%")
            print(f"  Mean trajectory length: {mean_len:.1f} time bins")
    
    return trajectories


# =============================================================================
# Save/Load
# =============================================================================

def save_trajectories(trajectories: List[Dict], output_path: str,
                      config: MIMICIVConfig, verbose: bool = True):
    """Save trajectories to NPZ file."""
    # Pad to same length
    max_len = max(t['n_bins'] for t in trajectories)
    n_features = trajectories[0]['states'].shape[1]
    
    states_padded = np.zeros((len(trajectories), max_len, n_features))
    actions_padded = np.zeros((len(trajectories), max_len), dtype=int)
    rewards_padded = np.zeros((len(trajectories), max_len))
    lengths = np.array([t['n_bins'] for t in trajectories])
    died = np.array([t['died'] for t in trajectories])
    stay_ids = np.array([t['stay_id'] for t in trajectories])
    
    for i, traj in enumerate(trajectories):
        l = traj['n_bins']
        states_padded[i, :l] = traj['states']
        actions_padded[i, :l] = traj['actions']
        rewards_padded[i, :l] = traj['rewards']
    
    np.savez(
        output_path,
        states=states_padded,
        actions=actions_padded,
        rewards=rewards_padded,
        lengths=lengths,
        died=died,
        stay_ids=stay_ids,
        feature_names=np.array(trajectories[0]['feature_names']),
        n_actions=config.n_fluid_bins * config.n_vaso_bins,
        time_bin_hours=config.time_bin_hours,
    )
    
    if verbose:
        print(f"\nSaved to {output_path}")
        print(f"  States shape: {states_padded.shape}")
        print(f"  Actions shape: {actions_padded.shape}")


def save_summary(trajectories: List[Dict], output_path: str, config: MIMICIVConfig):
    """Save dataset summary."""
    with open(output_path, 'w') as f:
        f.write("MIMIC-IV Demo Dataset Summary (Preprocessed)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of trajectories: {len(trajectories)}\n")
        f.write(f"Mortality rate: {100 * sum(t['died'] for t in trajectories) / len(trajectories):.1f}%\n")
        f.write(f"Mean trajectory length: {np.mean([t['n_bins'] for t in trajectories]):.1f} time bins\n")
        f.write(f"Time bin size: {config.time_bin_hours} hours\n")
        f.write(f"Number of features: {trajectories[0]['states'].shape[1]}\n")
        f.write(f"Number of actions: {config.n_fluid_bins * config.n_vaso_bins}\n")
        f.write(f"\nFeatures: {trajectories[0]['feature_names']}\n")
    
    print(f"Saved summary to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("MIMIC-IV Demo Preprocessing for Concept-Based OPE")
    print("Following MIMIC-Sepsis / Komorowski methodology")
    print("=" * 60)
    
    config = MIMICIVConfig()
    
    # Load data
    print("\n[1/4] Loading tables...")
    tables = load_tables(config, verbose=True)
    
    # Build trajectories
    print("\n[2/4] Building trajectories...")
    trajectories = build_all_trajectories(tables, config, verbose=True)
    
    if not trajectories:
        print("\nERROR: No trajectories built. Check your data files.")
        return
    
    # Save trajectories
    print("\n[3/4] Saving trajectories...")
    save_trajectories(trajectories, 'mimic_iv_trajectories.npz', config, verbose=True)
    
    # Save summary
    print("\n[4/4] Saving summary...")
    save_summary(trajectories, 'mimic_iv_summary.txt', config)
    
    print("\n" + "=" * 60)
    print("DONE! Files created:")
    print("  - mimic_iv_trajectories.npz")
    print("  - mimic_iv_summary.txt")
    print("=" * 60)
    
    return trajectories


if __name__ == "__main__":
    trajectories = main()