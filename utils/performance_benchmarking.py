"""
Performance benchmarking utilities for Wine Quality Analysis.

This module provides functions for timing benchmarks and memory usage monitoring
for model training and data processing operations.
"""

import time
import psutil
import pandas as pd
import numpy as np
from typing import Dict, Any, Callable, Optional, List
import functools
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path to import config
sys.path.append(str(Path(__file__).parent.parent))


class PerformanceBenchmark:
    """Class for tracking performance metrics during ML operations."""
    
    def __init__(self):
        """Initialize performance tracking."""
        self.benchmarks = []
        self.process = psutil.Process()
    
    def benchmark_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Benchmark a function's execution time and memory usage.
        
        Args:
            func: Function to benchmark
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function
            
        Returns:
            Dict containing performance metrics and function result
        """
        # Get initial memory usage
        memory_before = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Record start time
        start_time = time.time()
        start_cpu_time = time.process_time()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        # Record end time
        end_time = time.time()
        end_cpu_time = time.process_time()
        
        # Get final memory usage
        memory_after = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate metrics
        wall_time = end_time - start_time
        cpu_time = end_cpu_time - start_cpu_time
        memory_used = memory_after - memory_before
        peak_memory = memory_after
        
        benchmark_result = {
            'function_name': func.__name__,
            'wall_time': wall_time,
            'cpu_time': cpu_time,
            'memory_before': memory_before,
            'memory_after': memory_after,
            'memory_used': memory_used,
            'peak_memory': peak_memory,
            'success': success,
            'error': error,
            'result': result,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Store benchmark
        self.benchmarks.append(benchmark_result)
        
        return benchmark_result
    
    def get_benchmarks_summary(self) -> pd.DataFrame:
        """
        Get summary of all benchmarks as DataFrame.
        
        Returns:
            DataFrame with benchmark results
        """
        if not self.benchmarks:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.benchmarks)
        
        # Select key columns for summary
        summary_cols = ['function_name', 'wall_time', 'cpu_time', 'memory_used', 
                       'peak_memory', 'success', 'timestamp']
        
        return df[summary_cols]
    
    def print_benchmark_report(self) -> None:
        """Print detailed benchmark report."""
        if not self.benchmarks:
            print("No benchmarks recorded.")
            return
        
        print("=" * 80)
        print("PERFORMANCE BENCHMARK REPORT")
        print("=" * 80)
        
        df = self.get_benchmarks_summary()
        
        print(f"\nTotal functions benchmarked: {len(df)}")
        print(f"Successful executions: {df['success'].sum()}")
        print(f"Failed executions: {(~df['success']).sum()}")
        
        print(f"\nTiming Summary:")
        print(f"  Total wall time: {df['wall_time'].sum():.3f} seconds")
        print(f"  Average wall time: {df['wall_time'].mean():.3f} seconds")
        print(f"  Total CPU time: {df['cpu_time'].sum():.3f} seconds")
        print(f"  Average CPU time: {df['cpu_time'].mean():.3f} seconds")
        
        print(f"\nMemory Summary:")
        print(f"  Total memory used: {df['memory_used'].sum():.2f} MB")
        print(f"  Average memory used: {df['memory_used'].mean():.2f} MB")
        print(f"  Peak memory usage: {df['peak_memory'].max():.2f} MB")
        
        print(f"\nDetailed Results:")
        print("-" * 80)
        
        for _, row in df.iterrows():
            status = "✓" if row['success'] else "✗"
            print(f"{status} {row['function_name']:<25} | "
                  f"Time: {row['wall_time']:6.3f}s | "
                  f"CPU: {row['cpu_time']:6.3f}s | "
                  f"Memory: {row['memory_used']:+6.2f}MB | "
                  f"Peak: {row['peak_memory']:6.2f}MB")
        
        print("=" * 80)


def benchmark_decorator(benchmark_tracker: PerformanceBenchmark):
    """
    Decorator to automatically benchmark function calls.
    
    Args:
        benchmark_tracker: PerformanceBenchmark instance to store results
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = benchmark_tracker.benchmark_function(func, *args, **kwargs)
            return result['result']
        return wrapper
    return decorator


def benchmark_model_training(models_dict: Dict[str, Any], X_train: pd.DataFrame, 
                           y_train: pd.Series) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark model training performance.
    
    Args:
        models_dict: Dictionary of model classes/functions to benchmark
        X_train: Training features
        y_train: Training target
        
    Returns:
        Dict containing benchmark results for each model
    """
    print("=" * 80)
    print("MODEL TRAINING PERFORMANCE BENCHMARKS")
    print("=" * 80)
    
    benchmark_tracker = PerformanceBenchmark()
    results = {}
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Target data shape: {y_train.shape}")
    print(f"Models to benchmark: {list(models_dict.keys())}")
    
    for model_name, model_func in models_dict.items():
        print(f"\nBenchmarking {model_name}...")
        
        # Benchmark model training
        benchmark_result = benchmark_tracker.benchmark_function(
            model_func, X_train, y_train
        )
        
        results[model_name] = benchmark_result
        
        # Print immediate results
        if benchmark_result['success']:
            print(f"  ✓ Training completed in {benchmark_result['wall_time']:.3f}s")
            print(f"  ✓ CPU time: {benchmark_result['cpu_time']:.3f}s")
            print(f"  ✓ Memory used: {benchmark_result['memory_used']:+.2f}MB")
            print(f"  ✓ Peak memory: {benchmark_result['peak_memory']:.2f}MB")
        else:
            print(f"  ✗ Training failed: {benchmark_result['error']}")
    
    # Print summary report
    print(f"\n" + "=" * 80)
    print("MODEL TRAINING BENCHMARK SUMMARY")
    print("=" * 80)
    
    benchmark_tracker.print_benchmark_report()
    
    return results


def benchmark_data_processing(processing_functions: Dict[str, Callable], 
                            data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark data processing functions.
    
    Args:
        processing_functions: Dictionary of processing functions to benchmark
        data: Input data for processing
        
    Returns:
        Dict containing benchmark results for each function
    """
    print("=" * 80)
    print("DATA PROCESSING PERFORMANCE BENCHMARKS")
    print("=" * 80)
    
    benchmark_tracker = PerformanceBenchmark()
    results = {}
    
    print(f"Input data shape: {data.shape}")
    print(f"Functions to benchmark: {list(processing_functions.keys())}")
    
    for func_name, func in processing_functions.items():
        print(f"\nBenchmarking {func_name}...")
        
        # Benchmark function
        benchmark_result = benchmark_tracker.benchmark_function(func, data.copy())
        
        results[func_name] = benchmark_result
        
        # Print immediate results
        if benchmark_result['success']:
            print(f"  ✓ Processing completed in {benchmark_result['wall_time']:.3f}s")
            print(f"  ✓ CPU time: {benchmark_result['cpu_time']:.3f}s")
            print(f"  ✓ Memory used: {benchmark_result['memory_used']:+.2f}MB")
        else:
            print(f"  ✗ Processing failed: {benchmark_result['error']}")
    
    # Print summary report
    print(f"\n" + "=" * 80)
    print("DATA PROCESSING BENCHMARK SUMMARY")
    print("=" * 80)
    
    benchmark_tracker.print_benchmark_report()
    
    return results


def monitor_memory_usage(func: Callable, *args, interval: float = 0.1, **kwargs) -> Dict[str, Any]:
    """
    Monitor memory usage during function execution.
    
    Args:
        func: Function to monitor
        *args: Arguments for function
        interval: Monitoring interval in seconds
        **kwargs: Keyword arguments for function
        
    Returns:
        Dict containing memory usage statistics and function result
    """
    import threading
    import queue
    
    memory_readings = []
    monitoring = True
    
    def memory_monitor():
        """Monitor memory usage in separate thread."""
        process = psutil.Process()
        while monitoring:
            try:
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_readings.append({
                    'timestamp': time.time(),
                    'memory_mb': memory_mb
                })
                time.sleep(interval)
            except:
                break
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=memory_monitor)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Execute function
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        success = True
        error = None
    except Exception as e:
        result = None
        success = False
        error = str(e)
    finally:
        monitoring = False
        monitor_thread.join(timeout=1)
    
    end_time = time.time()
    
    # Analyze memory usage
    if memory_readings:
        memory_values = [r['memory_mb'] for r in memory_readings]
        memory_stats = {
            'initial_memory': memory_values[0],
            'peak_memory': max(memory_values),
            'final_memory': memory_values[-1],
            'average_memory': np.mean(memory_values),
            'memory_increase': max(memory_values) - memory_values[0],
            'readings_count': len(memory_readings)
        }
    else:
        memory_stats = {
            'initial_memory': 0,
            'peak_memory': 0,
            'final_memory': 0,
            'average_memory': 0,
            'memory_increase': 0,
            'readings_count': 0
        }
    
    return {
        'function_name': func.__name__,
        'execution_time': end_time - start_time,
        'success': success,
        'error': error,
        'result': result,
        'memory_stats': memory_stats,
        'memory_readings': memory_readings
    }


def create_performance_visualization(benchmark_results: Dict[str, Dict[str, Any]], 
                                   title: str = "Performance Benchmarks") -> None:
    """
    Create visualizations for performance benchmark results.
    
    Args:
        benchmark_results: Dictionary of benchmark results
        title: Title for the visualization
    """
    if not benchmark_results:
        print("No benchmark results to visualize.")
        return
    
    # Extract data for visualization
    names = []
    wall_times = []
    cpu_times = []
    memory_used = []
    peak_memory = []
    
    for name, result in benchmark_results.items():
        if result['success']:
            names.append(name)
            wall_times.append(result['wall_time'])
            cpu_times.append(result['cpu_time'])
            memory_used.append(result['memory_used'])
            peak_memory.append(result['peak_memory'])
    
    if not names:
        print("No successful benchmarks to visualize.")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Wall time comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(names, wall_times, color='skyblue', alpha=0.7)
    ax1.set_title('Wall Time Comparison')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, time_val in zip(bars1, wall_times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{time_val:.3f}s', ha='center', va='bottom')
    
    # 2. CPU time comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(names, cpu_times, color='lightcoral', alpha=0.7)
    ax2.set_title('CPU Time Comparison')
    ax2.set_ylabel('Time (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, time_val in zip(bars2, cpu_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{time_val:.3f}s', ha='center', va='bottom')
    
    # 3. Memory usage comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(names, memory_used, color='lightgreen', alpha=0.7)
    ax3.set_title('Memory Usage Comparison')
    ax3.set_ylabel('Memory Used (MB)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on bars
    for bar, mem_val in zip(bars3, memory_used):
        ax3.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + (0.5 if mem_val >= 0 else -0.5),
                f'{mem_val:+.1f}MB', ha='center', 
                va='bottom' if mem_val >= 0 else 'top')
    
    # 4. Peak memory comparison
    ax4 = axes[1, 1]
    bars4 = ax4.bar(names, peak_memory, color='gold', alpha=0.7)
    ax4.set_title('Peak Memory Usage')
    ax4.set_ylabel('Peak Memory (MB)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, mem_val in zip(bars4, peak_memory):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{mem_val:.1f}MB', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Generated performance visualization for {len(names)} successful benchmarks")


def benchmark_complete_pipeline(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Benchmark the complete ML pipeline performance.
    
    Args:
        df: Input dataset
        
    Returns:
        Dict containing comprehensive benchmark results
    """
    print("=" * 100)
    print("COMPLETE PIPELINE PERFORMANCE BENCHMARK")
    print("=" * 100)
    
    # Import required modules
    from utils.data_loader import load_wine_dataset
    from utils.preprocessing import preprocess_data
    from utils.models import build_random_forest, build_svm, build_gradient_boosting
    from utils.models import evaluate_models
    
    benchmark_tracker = PerformanceBenchmark()
    pipeline_results = {}
    
    print(f"Input dataset shape: {df.shape}")
    
    # 1. Benchmark data preprocessing
    print(f"\n1. BENCHMARKING DATA PREPROCESSING")
    print("-" * 50)
    
    preprocess_result = benchmark_tracker.benchmark_function(
        preprocess_data, df, 'quality'
    )
    pipeline_results['preprocessing'] = preprocess_result
    
    if preprocess_result['success']:
        X_train, X_test, y_train, y_test = preprocess_result['result']
        print(f"  ✓ Preprocessing completed in {preprocess_result['wall_time']:.3f}s")
        print(f"  ✓ Training data: {X_train.shape}")
        print(f"  ✓ Test data: {X_test.shape}")
    else:
        print(f"  ✗ Preprocessing failed: {preprocess_result['error']}")
        return pipeline_results
    
    # 2. Benchmark model training
    print(f"\n2. BENCHMARKING MODEL TRAINING")
    print("-" * 50)
    
    models_to_benchmark = {
        'Random Forest': build_random_forest,
        'SVM': build_svm,
        'Gradient Boosting': build_gradient_boosting
    }
    
    model_results = {}
    trained_models = {}
    
    for model_name, model_func in models_to_benchmark.items():
        print(f"\n  Benchmarking {model_name}...")
        
        model_result = benchmark_tracker.benchmark_function(
            model_func, X_train, y_train
        )
        model_results[model_name] = model_result
        
        if model_result['success']:
            trained_models[model_name] = model_result['result']
            print(f"    ✓ Training completed in {model_result['wall_time']:.3f}s")
        else:
            print(f"    ✗ Training failed: {model_result['error']}")
    
    pipeline_results['model_training'] = model_results
    
    # 3. Benchmark model evaluation
    if trained_models:
        print(f"\n3. BENCHMARKING MODEL EVALUATION")
        print("-" * 50)
        
        evaluation_result = benchmark_tracker.benchmark_function(
            evaluate_models, trained_models, X_test, y_test, X_train, y_train
        )
        pipeline_results['evaluation'] = evaluation_result
        
        if evaluation_result['success']:
            print(f"  ✓ Evaluation completed in {evaluation_result['wall_time']:.3f}s")
        else:
            print(f"  ✗ Evaluation failed: {evaluation_result['error']}")
    
    # 4. Generate comprehensive report
    print(f"\n4. COMPREHENSIVE BENCHMARK REPORT")
    print("-" * 50)
    
    benchmark_tracker.print_benchmark_report()
    
    # 5. Create performance visualization
    print(f"\n5. PERFORMANCE VISUALIZATION")
    print("-" * 50)
    
    # Prepare data for visualization
    viz_data = {}
    
    # Add preprocessing
    if pipeline_results['preprocessing']['success']:
        viz_data['Preprocessing'] = pipeline_results['preprocessing']
    
    # Add model training results
    for model_name, result in model_results.items():
        if result['success']:
            viz_data[f'{model_name} Training'] = result
    
    # Add evaluation
    if 'evaluation' in pipeline_results and pipeline_results['evaluation']['success']:
        viz_data['Model Evaluation'] = pipeline_results['evaluation']
    
    # Create visualization
    create_performance_visualization(viz_data, "Complete ML Pipeline Performance")
    
    # 6. Performance insights
    print(f"\n6. PERFORMANCE INSIGHTS")
    print("-" * 50)
    
    total_time = sum(r['wall_time'] for r in benchmark_tracker.benchmarks if r['success'])
    total_memory = sum(r['memory_used'] for r in benchmark_tracker.benchmarks if r['success'])
    
    print(f"  Total pipeline execution time: {total_time:.3f} seconds")
    print(f"  Total memory usage: {total_memory:+.2f} MB")
    
    # Find bottlenecks
    successful_benchmarks = [b for b in benchmark_tracker.benchmarks if b['success']]
    if successful_benchmarks:
        slowest = max(successful_benchmarks, key=lambda x: x['wall_time'])
        memory_intensive = max(successful_benchmarks, key=lambda x: x['memory_used'])
        
        print(f"  Slowest operation: {slowest['function_name']} ({slowest['wall_time']:.3f}s)")
        print(f"  Most memory intensive: {memory_intensive['function_name']} ({memory_intensive['memory_used']:+.2f}MB)")
    
    print("=" * 100)
    print("PIPELINE BENCHMARK COMPLETED")
    print("=" * 100)
    
    return pipeline_results


def save_benchmark_results(benchmark_results: Dict[str, Any], 
                         filename: str = "benchmark_results.json") -> None:
    """
    Save benchmark results to file.
    
    Args:
        benchmark_results: Dictionary of benchmark results
        filename: Output filename
    """
    import json
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Clean results for serialization
    clean_results = {}
    for key, value in benchmark_results.items():
        if isinstance(value, dict):
            clean_value = {}
            for k, v in value.items():
                if k != 'result':  # Skip actual model objects
                    clean_value[k] = convert_numpy_types(v)
            clean_results[key] = clean_value
        else:
            clean_results[key] = convert_numpy_types(value)
    
    # Save to file
    try:
        with open(filename, 'w') as f:
            json.dump(clean_results, f, indent=2, default=str)
        print(f"Benchmark results saved to {filename}")
    except Exception as e:
        print(f"Error saving benchmark results: {e}")


if __name__ == "__main__":
    """
    Example usage of performance benchmarking utilities.
    """
    print("Performance Benchmarking Utilities")
    print("This module provides tools for benchmarking ML pipeline performance.")
    print("Import and use the functions in your analysis scripts.")