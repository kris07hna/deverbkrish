#!/usr/bin/env python3
"""
Basic import test for dereverberation components.
Tests that modules can be imported and basic syntax is correct.
"""

import os
import sys

def test_basic_syntax():
    """Test that all Python files have valid syntax."""
    print("Testing Python syntax...")
    
    files_to_check = [
        'sgmse/dereverb_data_module.py',
        'sgmse/util/dataset_utils.py',
        'preprocessing/create_ace_dataset.py',
        'preprocessing/analyze_reverb_dataset.py',
        'train_dereverb.py',
        'evaluate_dereverb.py'
    ]
    
    base_dir = '/home/runner/work/deverbkrish/deverbkrish'
    
    for file_path in files_to_check:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            try:
                with open(full_path, 'r') as f:
                    code = f.read()
                
                # Try to compile the code
                compile(code, full_path, 'exec')
                print(f"✓ {file_path}: Syntax OK")
                
            except SyntaxError as e:
                print(f"✗ {file_path}: Syntax Error - {e}")
                return False
            except Exception as e:
                print(f"? {file_path}: Warning - {e}")
        else:
            print(f"✗ {file_path}: File not found")
            return False
    
    return True


def test_imports_without_dependencies():
    """Test imports that don't require external dependencies."""
    print("\nTesting basic imports...")
    
    # Add project to path
    project_dir = '/home/runner/work/deverbkrish/deverbkrish'
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    
    try:
        # Test if we can import basic Python modules used
        import argparse
        import logging
        from pathlib import Path
        print("✓ Standard library imports OK")
        
        # Test if our modules exist and can be found
        import sgmse
        print("✓ sgmse package found")
        
        # Check if our new files exist
        dereverb_module_path = os.path.join(project_dir, 'sgmse', 'dereverb_data_module.py')
        if os.path.exists(dereverb_module_path):
            print("✓ dereverb_data_module.py exists")
        else:
            print("✗ dereverb_data_module.py not found")
            return False
        
        utils_path = os.path.join(project_dir, 'sgmse', 'util', 'dataset_utils.py')
        if os.path.exists(utils_path):
            print("✓ dataset_utils.py exists")
        else:
            print("✗ dataset_utils.py not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False


def test_file_structure():
    """Test that all expected files were created with reasonable content."""
    print("\nTesting file structure...")
    
    base_dir = '/home/runner/work/deverbkrish/deverbkrish'
    
    expected_files = {
        'sgmse/dereverb_data_module.py': 10000,  # Minimum expected size
        'sgmse/util/dataset_utils.py': 15000,
        'preprocessing/create_ace_dataset.py': 15000,
        'preprocessing/analyze_reverb_dataset.py': 20000,
        'train_dereverb.py': 10000,
        'evaluate_dereverb.py': 15000
    }
    
    for file_path, min_size in expected_files.items():
        full_path = os.path.join(base_dir, file_path)
        
        if not os.path.exists(full_path):
            print(f"✗ {file_path}: File missing")
            return False
        
        file_size = os.path.getsize(full_path)
        if file_size < min_size:
            print(f"✗ {file_path}: File too small ({file_size} < {min_size} bytes)")
            return False
        
        print(f"✓ {file_path}: {file_size} bytes")
    
    return True


def test_class_definitions():
    """Test that key classes are defined in the modules."""
    print("\nTesting class definitions...")
    
    try:
        base_dir = '/home/runner/work/deverbkrish/deverbkrish'
        
        # Check dereverb_data_module
        dereverb_file = os.path.join(base_dir, 'sgmse', 'dereverb_data_module.py')
        with open(dereverb_file, 'r') as f:
            content = f.read()
        
        required_classes = ['ReverbSpecs', 'ReverbDataModule']
        for class_name in required_classes:
            if f'class {class_name}' in content:
                print(f"✓ Found class {class_name}")
            else:
                print(f"✗ Missing class {class_name}")
                return False
        
        # Check for key methods
        required_methods = ['__init__', '__getitem__', '__len__']
        for method in required_methods:
            if f'def {method}' in content:
                print(f"✓ Found method {method}")
            else:
                print(f"✗ Missing method {method}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Class definition test failed: {e}")
        return False


def test_function_definitions():
    """Test that key functions are defined in the utility modules."""
    print("\nTesting function definitions...")
    
    try:
        base_dir = '/home/runner/work/deverbkrish/deverbkrish'
        
        # Check dataset_utils
        utils_file = os.path.join(base_dir, 'sgmse', 'util', 'dataset_utils.py')
        with open(utils_file, 'r') as f:
            content = f.read()
        
        required_functions = [
            'create_dataset_csv',
            'validate_dataset_csv',
            'analyze_dataset_statistics',
            'extract_metadata_from_filename'
        ]
        
        for func_name in required_functions:
            if f'def {func_name}' in content:
                print(f"✓ Found function {func_name}")
            else:
                print(f"✗ Missing function {func_name}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Function definition test failed: {e}")
        return False


def run_basic_tests():
    """Run basic tests that don't require external dependencies."""
    print("=" * 60)
    print("BASIC DEREVERBERATION COMPONENTS TEST")
    print("=" * 60)
    
    tests = [
        test_basic_syntax,
        test_imports_without_dependencies,
        test_file_structure,
        test_class_definitions,
        test_function_definitions
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
        print("🎉 All basic tests passed!")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False


if __name__ == '__main__':
    success = run_basic_tests()
    sys.exit(0 if success else 1)