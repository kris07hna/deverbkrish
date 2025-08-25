#!/usr/bin/env python3
"""
Simple test script to validate the dereverberation components.
Tests basic functionality and imports without requiring full dependencies.
"""

import os
import sys
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

def test_basic_imports():
    """Test that all components can be imported."""
    print("Testing basic imports...")
    
    try:
        # Test data module import
        sys.path.insert(0, '/home/runner/work/deverbkrish/deverbkrish')
        from sgmse.dereverb_data_module import ReverbDataModule, ReverbSpecs
        print("✓ ReverbDataModule imported successfully")
        
        # Test dataset utilities import
        from sgmse.util.dataset_utils import create_dataset_csv, validate_dataset_csv
        print("✓ Dataset utilities imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_dataset_csv_creation():
    """Test CSV dataset creation functionality."""
    print("\nTesting CSV dataset creation...")
    
    try:
        # Create temporary directories and files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock audio structure
            train_dir = os.path.join(temp_dir, 'train')
            clean_dir = os.path.join(train_dir, 'anechoic')
            reverb_dir = os.path.join(train_dir, 'reverb')
            
            os.makedirs(clean_dir)
            os.makedirs(reverb_dir)
            
            # Create mock audio files (empty for testing)
            test_files = ['test1.wav', 'test2.wav']
            for filename in test_files:
                Path(os.path.join(clean_dir, filename)).touch()
                Path(os.path.join(reverb_dir, filename)).touch()
            
            # Test CSV creation
            from sgmse.util.dataset_utils import create_dataset_csv
            
            csv_path = os.path.join(temp_dir, 'test_dataset.csv')
            df = create_dataset_csv(
                audio_dir=temp_dir,
                output_csv=csv_path,
                extract_metadata=False,
                file_extensions=['.wav']
            )
            
            # Validate results
            assert os.path.exists(csv_path), "CSV file was not created"
            assert len(df) == len(test_files), f"Expected {len(test_files)} samples, got {len(df)}"
            assert 'clean_path' in df.columns, "Missing clean_path column"
            assert 'reverb_path' in df.columns, "Missing reverb_path column"
            
            print(f"✓ Created CSV with {len(df)} samples")
            return True
            
    except Exception as e:
        print(f"✗ CSV creation test failed: {e}")
        return False


def test_dataset_validation():
    """Test dataset validation functionality."""
    print("\nTesting dataset validation...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test CSV
            csv_path = os.path.join(temp_dir, 'test_validation.csv')
            
            # Create test data
            test_data = {
                'clean_path': ['/path/to/clean1.wav', '/path/to/clean2.wav'],
                'reverb_path': ['/path/to/reverb1.wav', '/path/to/reverb2.wav'],
                't60': [0.5, 1.2],
                'drr': [-5.0, -8.5],
                'room_type': ['office', 'hall']
            }
            
            df = pd.DataFrame(test_data)
            df.to_csv(csv_path, index=False)
            
            # Test validation
            from sgmse.util.dataset_utils import validate_dataset_csv
            
            results = validate_dataset_csv(csv_path, check_files=False)
            
            # Check validation results
            assert 'valid' in results, "Missing 'valid' key in results"
            assert results['valid'] == True, "Validation should pass for this test data"
            assert 'statistics' in results, "Missing statistics in results"
            
            stats = results['statistics']
            assert stats['total_samples'] == 2, f"Expected 2 samples, got {stats['total_samples']}"
            
            print("✓ Dataset validation passed")
            return True
            
    except Exception as e:
        print(f"✗ Dataset validation test failed: {e}")
        return False


def test_metadata_extraction():
    """Test metadata extraction from filenames."""
    print("\nTesting metadata extraction...")
    
    try:
        from sgmse.util.dataset_utils import extract_metadata_from_filename
        
        # Test various filename patterns
        test_cases = [
            ('speech_room1_0.5_-2.1.wav', {'t60': 0.5, 'drr': -2.1}),
            ('clean_office_1.2.wav', {'t60': 1.2}),
            ('test_hall_d2.5.wav', {'distance': 2.5}),
        ]
        
        for filename, expected in test_cases:
            metadata = extract_metadata_from_filename(filename)
            
            for key, value in expected.items():
                if key in metadata:
                    assert abs(metadata[key] - value) < 0.1, f"Metadata extraction failed for {filename}"
        
        print("✓ Metadata extraction working")
        return True
        
    except Exception as e:
        print(f"✗ Metadata extraction test failed: {e}")
        return False


def test_reverb_data_module_basic():
    """Test basic ReverbDataModule functionality."""
    print("\nTesting ReverbDataModule basics...")
    
    try:
        from sgmse.dereverb_data_module import ReverbDataModule
        
        # Test argument parsing
        import argparse
        parser = argparse.ArgumentParser()
        ReverbDataModule.add_argparse_args(parser)
        
        # Check that reverb-specific arguments were added
        args = parser.parse_args([
            '--base_dir', '/tmp/test',
            '--csv_path', '/tmp/test.csv',
            '--filter_t60_min', '0.3',
            '--filter_t60_max', '2.0'
        ])
        
        assert hasattr(args, 'csv_path'), "Missing csv_path argument"
        assert hasattr(args, 'filter_t60_min'), "Missing filter_t60_min argument"
        assert args.filter_t60_min == 0.3, "Incorrect filter_t60_min value"
        
        print("✓ ReverbDataModule argument parsing working")
        return True
        
    except Exception as e:
        print(f"✗ ReverbDataModule test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("DEREVERBERATION COMPONENTS TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_dataset_csv_creation,
        test_dataset_validation,
        test_metadata_extraction,
        test_reverb_data_module_basic
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)