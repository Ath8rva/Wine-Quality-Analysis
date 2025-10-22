#!/usr/bin/env python3
"""
Test script for performance benchmarking utilities.

This script tests the performance benchmarking functionality with sample data
and operations to ensure it works correctly.
"""

import sys
import pandas as pd
import numpy as np
import time
from pathlib import Path

# Add utils to path
sys.path.append('utils')

# Import benchmarking utilities
from utils.performance_benchmarking import (
    PerformanceBenchmark, 
    benchmark_decorator,
    benchmark_model_training,
    benchmark_data_processing,
    monitor_memory_usage,
    create_performance_visualization
)


def sample_data_processing_function(df):
    """Sample data processing function for testing."""
    # Simulate some processing time
    time.sleep(0.1)
    
    # Do some actual processing
    result = df.copy()
    result['new_column'] = result.iloc[:, 0] * 2
    result = result.dropna()
    
    return result


def sample_slow_function(df):
    """Sample slow function for testing."""
    time.sleep(0.5)  # Simulate slow operation
    return df.describe()


def sample_memory_intensive_function(df):
    """Sample memory intensive function for testing."""
    # Create large temporary arrays
    large_array = np.random.random((1000, 1000))
    result = df.copy()
    
    # Do some operations that use memory
    for i in range(10):
        temp = np.random.random((100, 100))
        result['temp_col_' + str(i)] = np.mean(temp)
    
    return result


def sample_model_training_function(X_train, y_train):
    """Sample model training function for testing."""
    from sklearn.ensemble import RandomForestClassifier
    
    # Simulate model training
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    return model


def test_basic_benchmarking():
    """Test basic benchmarking functionality."""
    print("=" * 60)
    print("TESTING BASIC BENCHMARKING FUNCTIONALITY")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    sample_df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'feature3': np.random.normal(0, 1, 100),
        'target': np.random.choice([0, 1], 100)
    })
    
    # Initialize benchmark tracker
    benchmark = PerformanceBenchmark()
    
    # Test function benchmarking
    print("\n1. Testing function benchmarking...")
    
    result1 = benchmark.benchmark_function(sample_data_processing_function, sample_df)
    print(f"   ✓ Data processing benchmark: {result1['wall_time']:.3f}s")
    
    result2 = benchmark.benchmark_function(sample_slow_function, sample_df)
    print(f"   ✓ Slow function benchmark: {result2['wall_time']:.3f}s")
    
    # Test benchmark summary
    print("\n2. Testing benchmark summary...")
    summary_df = benchmark.get_benchmarks_summary()
    print(f"   ✓ Summary DataFrame shape: {summary_df.shape}")
    print(f"   ✓ Functions benchmarked: {len(summary_df)}")
    
    # Test benchmark report
    print("\n3. Testing benchmark report...")
    benchmark.print_benchmark_report()
    
    return True


def test_decorator_benchmarking():
    """Test decorator-based benchmarking."""
    print("\n" + "=" * 60)
    print("TESTING DECORATOR BENCHMARKING")
    print("=" * 60)
    
    # Create benchmark tracker
    benchmark = PerformanceBenchmark()
    
    # Create decorated function
    @benchmark_decorator(benchmark)
    def decorated_function(df):
        time.sleep(0.2)
        return df.shape
    
    # Create sample data
    sample_df = pd.DataFrame(np.random.random((50, 5)))
    
    # Test decorated function
    print("\n1. Testing decorated function...")
    result = decorated_function(sample_df)
    print(f"   ✓ Function result: {result}")
    print(f"   ✓ Benchmarks recorded: {len(benchmark.benchmarks)}")
    
    # Print report
    benchmark.print_benchmark_report()
    
    return True


def test_model_training_benchmarks():
    """Test model training benchmarking."""
    print("\n" + "=" * 60)
    print("TESTING MODEL TRAINING BENCHMARKS")
    print("=" * 60)
    
    # Create sample training data
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.random((200, 5)))
    y_train = pd.Series(np.random.choice([0, 1, 2], 200))
    
    # Define models to benchmark
    models_dict = {
        'Sample Model': sample_model_training_function
    }
    
    # Run benchmarks
    print("\n1. Running model training benchmarks...")
    results = benchmark_model_training(models_dict, X_train, y_train)
    
    print(f"   ✓ Models benchmarked: {len(results)}")
    print(f"   ✓ Successful trainings: {sum(1 for r in results.values() if r['success'])}")
    
    return True


def test_data_processing_benchmarks():
    """Test data processing benchmarking."""
    print("\n" + "=" * 60)
    print("TESTING DATA PROCESSING BENCHMARKS")
    print("=" * 60)
    
    # Create sample data
    sample_df = pd.DataFrame(np.random.random((500, 10)))
    
    # Define processing functions to benchmark
    processing_functions = {
        'Fast Processing': sample_data_processing_function,
        'Slow Processing': sample_slow_function,
        'Memory Intensive': sample_memory_intensive_function
    }
    
    # Run benchmarks
    print("\n1. Running data processing benchmarks...")
    results = benchmark_data_processing(processing_functions, sample_df)
    
    print(f"   ✓ Functions benchmarked: {len(results)}")
    print(f"   ✓ Successful executions: {sum(1 for r in results.values() if r['success'])}")
    
    return True


def test_memory_monitoring():
    """Test memory usage monitoring."""
    print("\n" + "=" * 60)
    print("TESTING MEMORY MONITORING")
    print("=" * 60)
    
    # Create sample data
    sample_df = pd.DataFrame(np.random.random((100, 10)))
    
    # Test memory monitoring
    print("\n1. Testing memory monitoring...")
    result = monitor_memory_usage(sample_memory_intensive_function, sample_df, interval=0.05)
    
    print(f"   ✓ Function executed: {result['success']}")
    print(f"   ✓ Execution time: {result['execution_time']:.3f}s")
    print(f"   ✓ Memory readings: {result['memory_stats']['readings_count']}")
    print(f"   ✓ Peak memory: {result['memory_stats']['peak_memory']:.2f}MB")
    print(f"   ✓ Memory increase: {result['memory_stats']['memory_increase']:.2f}MB")
    
    return True


def test_performance_visualization():
    """Test performance visualization."""
    print("\n" + "=" * 60)
    print("TESTING PERFORMANCE VISUALIZATION")
    print("=" * 60)
    
    # Create sample benchmark results
    sample_results = {
        'Function A': {
            'wall_time': 0.1,
            'cpu_time': 0.08,
            'memory_used': 5.2,
            'peak_memory': 45.8,
            'success': True
        },
        'Function B': {
            'wall_time': 0.3,
            'cpu_time': 0.25,
            'memory_used': -2.1,
            'peak_memory': 42.3,
            'success': True
        },
        'Function C': {
            'wall_time': 0.05,
            'cpu_time': 0.04,
            'memory_used': 1.8,
            'peak_memory': 44.1,
            'success': True
        }
    }
    
    # Test visualization
    print("\n1. Testing performance visualization...")
    try:
        create_performance_visualization(sample_results, "Test Performance Benchmarks")
        print("   ✓ Visualization created successfully")
        return True
    except Exception as e:
        print(f"   ✗ Visualization failed: {e}")
        return False


def run_all_tests():
    """Run all performance benchmarking tests."""
    print("PERFORMANCE BENCHMARKING UTILITIES - TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Basic Benchmarking", test_basic_benchmarking),
        ("Decorator Benchmarking", test_decorator_benchmarking),
        ("Model Training Benchmarks", test_model_training_benchmarks),
        ("Data Processing Benchmarks", test_data_processing_benchmarks),
        ("Memory Monitoring", test_memory_monitoring),
        ("Performance Visualization", test_performance_visualization)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning {test_name}...")
            success = test_func()
            results.append((test_name, success))
            print(f"✓ {test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            print(f"✗ {test_name}: FAILED with error: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    if passed == total:
        print(f"\n🎉 All tests passed! Performance benchmarking utilities are working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    """Run the test suite."""
    success = run_all_tests()
    
    if success:
        print(f"\n🚀 Performance benchmarking utilities are ready to use!")
        sys.exit(0)
    else:
        print(f"\n❌ Some tests failed. Please fix the issues.")
        sys.exit(1)