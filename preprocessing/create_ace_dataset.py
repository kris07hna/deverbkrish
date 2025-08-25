"""
ACE Corpus dataset preparation script.
Creates a structured dataset from ACE (Acoustic Characterization of Environments) Corpus
for dereverberation training and evaluation.
"""

import os
import argparse
import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path
from glob import glob
from tqdm import tqdm
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ACE Corpus room information
ACE_ROOMS = {
    'Office_1': {'type': 'office', 'size': 'medium', 'rt60_range': [0.3, 0.5]},
    'Office_2': {'type': 'office', 'size': 'medium', 'rt60_range': [0.4, 0.6]},
    'Meeting_Room_1': {'type': 'meeting', 'size': 'medium', 'rt60_range': [0.3, 0.7]},
    'Meeting_Room_2': {'type': 'meeting', 'size': 'large', 'rt60_range': [0.5, 0.9]},
    'Lecture_Room_1': {'type': 'lecture', 'size': 'large', 'rt60_range': [0.8, 1.2]},
    'Lecture_Room_2': {'type': 'lecture', 'size': 'large', 'rt60_range': [0.9, 1.3]},
    'Aula_Carolina_Rediviva': {'type': 'hall', 'size': 'large', 'rt60_range': [2.0, 3.0]},
    'Building_Lobby': {'type': 'lobby', 'size': 'large', 'rt60_range': [1.5, 2.5]}
}


def parse_ace_filename(filename: str) -> dict:
    """
    Parse ACE Corpus filename to extract metadata.
    
    ACE filenames typically follow pattern:
    Mobile_<device>_<room>_<position>_<channel>.wav
    RIR_<room>_<position>_<channel>.wav
    
    Args:
        filename: ACE Corpus filename
        
    Returns:
        Dictionary with extracted metadata
    """
    
    metadata = {}
    name_parts = filename.replace('.wav', '').split('_')
    
    if len(name_parts) >= 3:
        # Extract room information
        room_candidate = '_'.join(name_parts[1:-2]) if 'Mobile' in name_parts[0] else '_'.join(name_parts[1:-2])
        
        # Find matching room
        for room_name in ACE_ROOMS.keys():
            if room_name.replace('_', '').lower() in room_candidate.replace('_', '').lower():
                metadata['room_name'] = room_name
                metadata['room_type'] = ACE_ROOMS[room_name]['type']
                metadata['room_size'] = ACE_ROOMS[room_name]['size']
                metadata['t60_min'] = ACE_ROOMS[room_name]['rt60_range'][0]
                metadata['t60_max'] = ACE_ROOMS[room_name]['rt60_range'][1]
                break
        
        # Extract position and channel
        if len(name_parts) >= 2:
            metadata['position'] = name_parts[-2]
            metadata['channel'] = name_parts[-1]
    
    return metadata


def find_matching_files(clean_files: list, rir_dir: str, reverb_dir: str = None) -> list:
    """
    Find matching RIR and reverberant files for clean speech files.
    
    Args:
        clean_files: List of clean speech files
        rir_dir: Directory containing RIR files
        reverb_dir: Directory containing reverberant files (optional)
        
    Returns:
        List of dictionaries with file matches
    """
    
    matches = []
    rir_files = glob(os.path.join(rir_dir, "**/*.wav"), recursive=True)
    
    # Create RIR lookup by room and position
    rir_lookup = {}
    for rir_file in rir_files:
        rir_basename = os.path.basename(rir_file)
        rir_metadata = parse_ace_filename(rir_basename)
        
        if 'room_name' in rir_metadata and 'position' in rir_metadata:
            key = f"{rir_metadata['room_name']}_{rir_metadata['position']}"
            if key not in rir_lookup:
                rir_lookup[key] = []
            rir_lookup[key].append(rir_file)
    
    # Find reverberant files if directory provided
    reverb_lookup = {}
    if reverb_dir and os.path.exists(reverb_dir):
        reverb_files = glob(os.path.join(reverb_dir, "**/*.wav"), recursive=True)
        for reverb_file in reverb_files:
            reverb_basename = os.path.basename(reverb_file)
            reverb_metadata = parse_ace_filename(reverb_basename)
            
            if 'room_name' in reverb_metadata and 'position' in reverb_metadata:
                key = f"{reverb_metadata['room_name']}_{reverb_metadata['position']}"
                if key not in reverb_lookup:
                    reverb_lookup[key] = []
                reverb_lookup[key].append(reverb_file)
    
    # Match clean files with RIRs and reverberant files
    for clean_file in clean_files:
        clean_basename = os.path.basename(clean_file)
        
        # For each RIR room/position combination
        for key, rir_list in rir_lookup.items():
            room_name, position = key.split('_', 1)
            
            for rir_file in rir_list:
                match = {
                    'clean_path': clean_file,
                    'rir_path': rir_file,
                    'room_name': room_name,
                    'position': position
                }
                
                # Add room metadata
                if room_name in ACE_ROOMS:
                    match.update({
                        'room_type': ACE_ROOMS[room_name]['type'],
                        'room_size': ACE_ROOMS[room_name]['size'],
                        't60_min': ACE_ROOMS[room_name]['rt60_range'][0],
                        't60_max': ACE_ROOMS[room_name]['rt60_range'][1]
                    })
                
                # Add reverberant file if available
                if key in reverb_lookup:
                    # Find best matching reverberant file
                    reverb_candidates = reverb_lookup[key]
                    if reverb_candidates:
                        match['reverb_path'] = reverb_candidates[0]  # Take first match
                
                matches.append(match)
    
    return matches


def convolve_with_rir(clean_audio: np.ndarray, rir_audio: np.ndarray, 
                     normalize: bool = True) -> np.ndarray:
    """
    Convolve clean audio with room impulse response.
    
    Args:
        clean_audio: Clean audio signal
        rir_audio: Room impulse response
        normalize: Whether to normalize output
        
    Returns:
        Reverberant audio signal
    """
    
    # Ensure single channel
    if clean_audio.ndim > 1:
        clean_audio = clean_audio[:, 0]
    if rir_audio.ndim > 1:
        rir_audio = rir_audio[:, 0]
    
    # Convolve
    reverb_audio = np.convolve(clean_audio, rir_audio, mode='full')
    
    # Truncate to original length
    reverb_audio = reverb_audio[:len(clean_audio)]
    
    # Normalize if requested
    if normalize:
        max_val = np.max(np.abs(reverb_audio))
        if max_val > 0:
            reverb_audio = reverb_audio / max_val * 0.9
    
    return reverb_audio


def calculate_drr(clean_audio: np.ndarray, reverb_audio: np.ndarray) -> float:
    """
    Calculate Direct-to-Reverberant Ratio (DRR).
    
    Args:
        clean_audio: Clean audio signal
        reverb_audio: Reverberant audio signal
        
    Returns:
        DRR value in dB
    """
    
    # Calculate energy
    clean_energy = np.mean(clean_audio ** 2)
    reverb_energy = np.mean(reverb_audio ** 2)
    
    # Calculate DRR
    if reverb_energy > 0:
        drr = 10 * np.log10((clean_energy / reverb_energy) + 1e-8)
    else:
        drr = 0.0
    
    return drr


def estimate_t60_from_rir(rir_audio: np.ndarray, sample_rate: int) -> float:
    """
    Estimate T60 from room impulse response using energy decay curve.
    
    Args:
        rir_audio: Room impulse response
        sample_rate: Sample rate in Hz
        
    Returns:
        Estimated T60 in seconds
    """
    
    # Ensure single channel
    if rir_audio.ndim > 1:
        rir_audio = rir_audio[:, 0]
    
    # Calculate energy decay curve
    energy = rir_audio ** 2
    
    # Smooth the energy curve
    window_size = max(1, sample_rate // 100)  # 10ms window
    energy_smooth = np.convolve(energy, np.ones(window_size) / window_size, mode='same')
    
    # Convert to dB
    energy_db = 10 * np.log10(energy_smooth + 1e-10)
    
    # Find peak
    peak_idx = np.argmax(energy_db)
    
    # Find -60dB point
    peak_db = energy_db[peak_idx]
    target_db = peak_db - 60
    
    # Find where energy first drops below target
    decay_region = energy_db[peak_idx:]
    below_target = np.where(decay_region < target_db)[0]
    
    if len(below_target) > 0:
        t60_samples = peak_idx + below_target[0]
        t60 = t60_samples / sample_rate
    else:
        # Estimate from linear fit if -60dB point not reached
        decay_samples = len(decay_region)
        if decay_samples > sample_rate // 10:  # At least 100ms of decay
            # Fit line to decay region
            x = np.arange(decay_samples)
            y = decay_region
            
            # Use robust linear fit
            valid_idx = y > (peak_db - 40)  # Use first 40dB of decay
            if np.sum(valid_idx) > 10:
                x_fit = x[valid_idx]
                y_fit = y[valid_idx]
                
                # Linear regression
                coeffs = np.polyfit(x_fit, y_fit, 1)
                slope = coeffs[0]
                
                if slope < 0:
                    # Extrapolate to -60dB
                    t60_samples = -60 / slope
                    t60 = t60_samples / sample_rate
                else:
                    t60 = 1.0  # Default fallback
            else:
                t60 = 1.0  # Default fallback
        else:
            t60 = 1.0  # Default fallback
    
    # Clamp to reasonable range
    t60 = np.clip(t60, 0.1, 5.0)
    
    return t60


def create_ace_dataset(ace_corpus_dir: str, clean_speech_dir: str, output_dir: str,
                      sample_rate: int = 16000, subset_ratios: dict = None,
                      max_samples_per_room: int = None) -> str:
    """
    Create ACE Corpus dataset for dereverberation.
    
    Args:
        ace_corpus_dir: Path to ACE Corpus directory
        clean_speech_dir: Path to clean speech files
        output_dir: Output directory for processed dataset
        sample_rate: Target sample rate
        subset_ratios: Dictionary with train/valid/test ratios
        max_samples_per_room: Maximum samples per room (for limiting dataset size)
        
    Returns:
        Path to created dataset CSV file
    """
    
    if subset_ratios is None:
        subset_ratios = {'train': 0.7, 'valid': 0.15, 'test': 0.15}
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    audio_dir = os.path.join(output_dir, 'audio')
    for subset in ['train', 'valid', 'test']:
        for subdir in ['clean', 'reverb']:
            os.makedirs(os.path.join(audio_dir, subset, subdir), exist_ok=True)
    
    # Find ACE Corpus RIR files
    rir_dir = os.path.join(ace_corpus_dir, 'RIR')
    if not os.path.exists(rir_dir):
        raise FileNotFoundError(f"RIR directory not found: {rir_dir}")
    
    # Find clean speech files
    clean_files = []
    for ext in ['.wav', '.flac']:
        clean_files.extend(glob(os.path.join(clean_speech_dir, f"**/*{ext}"), recursive=True))
    
    if not clean_files:
        raise FileNotFoundError(f"No audio files found in: {clean_speech_dir}")
    
    logger.info(f"Found {len(clean_files)} clean speech files")
    
    # Find matching RIR files
    logger.info("Finding matching RIR files...")
    matches = find_matching_files(clean_files, rir_dir)
    
    if not matches:
        raise ValueError("No matching RIR files found")
    
    logger.info(f"Found {len(matches)} clean-RIR pairs")
    
    # Limit samples per room if specified
    if max_samples_per_room:
        room_counts = {}
        filtered_matches = []
        
        for match in matches:
            room = match.get('room_name', 'unknown')
            if room not in room_counts:
                room_counts[room] = 0
            
            if room_counts[room] < max_samples_per_room:
                filtered_matches.append(match)
                room_counts[room] += 1
        
        matches = filtered_matches
        logger.info(f"Limited to {len(matches)} samples ({max_samples_per_room} per room)")
    
    # Split into subsets
    np.random.seed(42)
    np.random.shuffle(matches)
    
    n_total = len(matches)
    n_train = int(n_total * subset_ratios['train'])
    n_valid = int(n_total * subset_ratios['valid'])
    
    train_matches = matches[:n_train]
    valid_matches = matches[n_train:n_train + n_valid]
    test_matches = matches[n_train + n_valid:]
    
    logger.info(f"Dataset split: train={len(train_matches)}, valid={len(valid_matches)}, test={len(test_matches)}")
    
    # Process each subset
    all_records = []
    
    for subset_name, subset_matches in [('train', train_matches), 
                                       ('valid', valid_matches), 
                                       ('test', test_matches)]:
        
        logger.info(f"Processing {subset_name} subset...")
        
        for i, match in enumerate(tqdm(subset_matches, desc=f"Processing {subset_name}")):
            try:
                # Load clean audio
                clean_audio, clean_sr = sf.read(match['clean_path'])
                
                # Load RIR
                rir_audio, rir_sr = sf.read(match['rir_path'])
                
                # Resample if necessary
                if clean_sr != sample_rate:
                    # Simple resampling (for production use librosa.resample)
                    clean_audio = clean_audio[::int(clean_sr / sample_rate)]
                
                if rir_sr != sample_rate:
                    rir_audio = rir_audio[::int(rir_sr / sample_rate)]
                
                # Ensure single channel
                if clean_audio.ndim > 1:
                    clean_audio = clean_audio[:, 0]
                if rir_audio.ndim > 1:
                    rir_audio = rir_audio[:, 0]
                
                # Create reverberant audio
                reverb_audio = convolve_with_rir(clean_audio, rir_audio)
                
                # Calculate metrics
                t60 = estimate_t60_from_rir(rir_audio, sample_rate)
                drr = calculate_drr(clean_audio, reverb_audio)
                
                # Generate output filenames
                clean_basename = os.path.basename(match['clean_path'])
                name_base = os.path.splitext(clean_basename)[0]
                room_name = match.get('room_name', 'unknown')
                position = match.get('position', 'pos1')
                
                output_filename = f"{name_base}_{room_name}_{position}_{t60:.2f}_{drr:.1f}.wav"
                
                # Save processed audio
                clean_output = os.path.join(audio_dir, subset_name, 'clean', output_filename)
                reverb_output = os.path.join(audio_dir, subset_name, 'reverb', output_filename)
                
                sf.write(clean_output, clean_audio, sample_rate)
                sf.write(reverb_output, reverb_audio, sample_rate)
                
                # Create record
                record = {
                    'clean_path': clean_output,
                    'reverb_path': reverb_output,
                    'rir_path': match['rir_path'],
                    'subset': subset_name,
                    't60': t60,
                    'drr': drr,
                    'room_name': match.get('room_name', ''),
                    'room_type': match.get('room_type', ''),
                    'room_size': match.get('room_size', ''),
                    'position': match.get('position', ''),
                    'duration': len(clean_audio) / sample_rate,
                    'sample_rate': sample_rate
                }
                
                all_records.append(record)
                
            except Exception as e:
                logger.warning(f"Error processing {match.get('clean_path', 'unknown')}: {e}")
                continue
    
    # Create and save dataset CSV
    df = pd.DataFrame(all_records)
    csv_path = os.path.join(output_dir, 'ace_dataset.csv')
    df.to_csv(csv_path, index=False)
    
    # Print summary
    logger.info(f"\nDataset creation completed!")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"CSV file: {csv_path}")
    logger.info(f"Audio directory: {audio_dir}")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Train/Valid/Test: {len(df[df['subset']=='train'])}/{len(df[df['subset']=='valid'])}/{len(df[df['subset']=='test'])}")
    
    if 't60' in df.columns:
        print(f"  T60 range: {df['t60'].min():.2f} - {df['t60'].max():.2f} seconds")
        print(f"  T60 mean: {df['t60'].mean():.2f} ± {df['t60'].std():.2f} seconds")
    
    if 'drr' in df.columns:
        print(f"  DRR range: {df['drr'].min():.1f} - {df['drr'].max():.1f} dB")
        print(f"  DRR mean: {df['drr'].mean():.1f} ± {df['drr'].std():.1f} dB")
    
    if 'room_type' in df.columns:
        print("  Room type distribution:")
        for room_type, count in df['room_type'].value_counts().items():
            print(f"    {room_type}: {count}")
    
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Create ACE Corpus dataset for dereverberation")
    
    parser.add_argument('--ace_corpus_dir', type=str, required=True,
                       help='Path to ACE Corpus directory')
    parser.add_argument('--clean_speech_dir', type=str, required=True,
                       help='Path to clean speech files directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed dataset')
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='Target sample rate (default: 16000)')
    parser.add_argument('--max_samples_per_room', type=int, default=None,
                       help='Maximum samples per room (for limiting dataset size)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--valid_ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Train, valid, and test ratios must sum to 1.0, got {total_ratio}")
    
    subset_ratios = {
        'train': args.train_ratio,
        'valid': args.valid_ratio,
        'test': args.test_ratio
    }
    
    # Create dataset
    csv_path = create_ace_dataset(
        ace_corpus_dir=args.ace_corpus_dir,
        clean_speech_dir=args.clean_speech_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        subset_ratios=subset_ratios,
        max_samples_per_room=args.max_samples_per_room
    )
    
    print(f"\nDataset created successfully!")
    print(f"CSV file: {csv_path}")


if __name__ == '__main__':
    main()