"""
Performance benchmarks for GRIFFIN components.
"""

import time
import numpy as np
import psutil
import os
from typing import Dict, Any, Callable, List
from contextlib import contextmanager

# Test utilities
from test_utils import TestDataGenerator, TestConfigBuilder


class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    def __init__(self):
        self.results = {}
    
    @contextmanager
    def timer(self, operation_name: str):
        """Context manager for timing operations."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            self.results[operation_name] = {
                'execution_time': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'peak_memory': end_memory
            }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def benchmark_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Benchmark a single function call."""
        # Warm up
        try:
            func(*args, **kwargs)
        except:
            pass  # Ignore errors in warmup
        
        # Actual benchmark
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        return {
            'execution_time': end_time - start_time,
            'memory_delta': end_memory - start_memory,
            'peak_memory': end_memory,
            'success': success,
            'error': error,
            'result': result
        }
    
    def run_scaling_test(self, func: Callable, 
                        sizes: List[int], 
                        data_generator: Callable) -> Dict[int, Dict[str, Any]]:
        """Run scaling tests with different data sizes."""
        results = {}
        
        for size in sizes:
            print(f"Testing with size: {size}")
            
            # Generate data
            X, y = data_generator(size)
            
            # Benchmark
            benchmark_result = self.benchmark_function(func, X, y)
            results[size] = benchmark_result
            
            # Print intermediate results
            if benchmark_result['success']:
                print(f"  Time: {benchmark_result['execution_time']:.3f}s")
                print(f"  Memory: {benchmark_result['memory_delta']:.1f}MB")
            else:
                print(f"  Failed: {benchmark_result['error']}")
        
        return results
    
    def generate_report(self) -> str:
        """Generate a performance report."""
        report = ["Performance Benchmark Report", "=" * 30, ""]
        
        for operation, metrics in self.results.items():
            report.append(f"Operation: {operation}")
            report.append(f"  Execution Time: {metrics['execution_time']:.3f} seconds")
            report.append(f"  Memory Delta: {metrics['memory_delta']:.1f} MB")
            report.append(f"  Peak Memory: {metrics['peak_memory']:.1f} MB")
            report.append("")
        
        return "\n".join(report)


def benchmark_data_preprocessing():
    """Benchmark data preprocessing performance."""
    print("Benchmarking Data Preprocessing...")
    
    benchmark = PerformanceBenchmark()
    
    # Test different data sizes
    sizes = [1000, 5000, 10000, 50000]
    
    def create_and_preprocess(size):
        """Create data and run preprocessing."""
        # Import here to avoid issues if modules aren't available
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        try:
            from src.data.preprocessing import DataLoader
        except ImportError:
            print("Warning: Could not import DataLoader, using mock")
            return None
        
        config = TestConfigBuilder.comprehensive_config()
        
        # Generate test data
        X, y = TestDataGenerator.generate_ciciot_like_data(
            n_samples=size, 
            random_state=42
        )
        
        # Run preprocessing
        loader = DataLoader(config)
        loader.set_data(X, y)
        data_splits = loader.prepare_data()
        
        return data_splits
    
    # Run scaling tests
    scaling_results = benchmark.run_scaling_test(
        create_and_preprocess,
        sizes,
        lambda size: (None, None)  # Data generation is handled inside
    )
    
    # Analyze results
    print("\nScaling Analysis:")
    print("Size\tTime(s)\tMemory(MB)\tSamples/sec")
    print("-" * 40)
    
    for size, result in scaling_results.items():
        if result['success']:
            time_per_sample = result['execution_time'] / size
            samples_per_sec = 1.0 / time_per_sample if time_per_sample > 0 else float('inf')
            print(f"{size}\t{result['execution_time']:.3f}\t{result['memory_delta']:.1f}\t\t{samples_per_sec:.1f}")
        else:
            print(f"{size}\tFAILED\t{result['error']}")


def benchmark_data_corruption_handling():
    """Benchmark handling of corrupted data."""
    print("\nBenchmarking Corrupted Data Handling...")
    
    benchmark = PerformanceBenchmark()
    
    def process_corrupted_data(corruption_rate):
        """Process data with different corruption rates."""
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        try:
            from src.data.preprocessing import DataLoader
        except ImportError:
            print("Warning: Could not import DataLoader")
            return None
        
        config = TestConfigBuilder.comprehensive_config()
        
        # Generate corrupted data
        X, y = TestDataGenerator.generate_corrupted_data(
            n_samples=5000,
            corruption_rate=corruption_rate,
            random_state=42
        )
        
        # Process
        loader = DataLoader(config)
        loader.set_data(X, y)
        data_splits = loader.prepare_data()
        
        return data_splits
    
    # Test different corruption rates
    corruption_rates = [0.0, 0.05, 0.1, 0.2, 0.5]
    
    print("Corruption Rate\tTime(s)\tMemory(MB)")
    print("-" * 35)
    
    for rate in corruption_rates:
        result = benchmark.benchmark_function(process_corrupted_data, rate)
        
        if result['success']:
            print(f"{rate:.2f}\t\t{result['execution_time']:.3f}\t{result['memory_delta']:.1f}")
        else:
            print(f"{rate:.2f}\t\tFAILED\t{result['error'][:30]}...")


def benchmark_memory_efficiency():
    """Benchmark memory efficiency with large datasets."""
    print("\nBenchmarking Memory Efficiency...")
    
    def check_memory_usage(n_samples, n_features):
        """Check memory usage for given data size."""
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Generate data
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features).astype(np.float32)  # Use float32 to save memory
        y = np.random.choice(3, n_samples)
        
        data_size_mb = X.nbytes / (1024 * 1024)
        
        try:
            from src.data.preprocessing import DataLoader
        except ImportError:
            print(f"Data size: {data_size_mb:.1f}MB - Cannot test processing")
            return data_size_mb
        
        config = TestConfigBuilder.minimal_config()
        config['model']['feature_groups'] = {'group1': n_features}
        
        # Process
        loader = DataLoader(config)
        loader.set_data(X, y)
        data_splits = loader.prepare_data()
        
        return data_size_mb
    
    # Test different data sizes
    test_cases = [
        (10000, 50),    # 1.9 MB
        (50000, 50),    # 9.5 MB  
        (100000, 50),   # 19 MB
        (100000, 100),  # 38 MB
    ]
    
    benchmark = PerformanceBenchmark()
    
    print("Samples\tFeatures\tData Size(MB)\tTime(s)\tMemory Delta(MB)")
    print("-" * 60)
    
    for n_samples, n_features in test_cases:
        result = benchmark.benchmark_function(check_memory_usage, n_samples, n_features)
        
        if result['success']:
            data_size = result['result']
            print(f"{n_samples}\t{n_features}\t\t{data_size:.1f}\t\t{result['execution_time']:.3f}\t{result['memory_delta']:.1f}")
        else:
            print(f"{n_samples}\t{n_features}\t\tFAILED\t\t{result['error'][:30]}...")


def benchmark_configuration_validation():
    """Benchmark configuration validation performance."""
    print("\nBenchmarking Configuration Validation...")
    
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from src.utils.common import ConfigManager
    except ImportError:
        print("Warning: Could not import ConfigManager")
        return
    
    benchmark = PerformanceBenchmark()
    
    # Test different config complexities
    configs = {
        'minimal': TestConfigBuilder.minimal_config(),
        'comprehensive': TestConfigBuilder.comprehensive_config(),
        'imbalanced': TestConfigBuilder.imbalanced_config()
    }
    
    print("Config Type\tValidation Time(ms)")
    print("-" * 30)
    
    for config_name, config in configs.items():
        # Run validation multiple times for more accurate timing
        times = []
        for _ in range(100):
            start = time.time()
            try:
                ConfigManager.validate_config(config)
                success = True
            except:
                success = False
            end = time.time()
            times.append(end - start)
        
        avg_time_ms = np.mean(times) * 1000
        print(f"{config_name}\t\t{avg_time_ms:.2f}")


def run_comprehensive_benchmarks():
    """Run all performance benchmarks."""
    print("GRIFFIN Performance Benchmarks")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        benchmark_data_preprocessing()
    except Exception as e:
        print(f"Data preprocessing benchmark failed: {e}")
    
    try:
        benchmark_data_corruption_handling()
    except Exception as e:
        print(f"Data corruption benchmark failed: {e}")
    
    try:
        benchmark_memory_efficiency()
    except Exception as e:
        print(f"Memory efficiency benchmark failed: {e}")
    
    try:
        benchmark_configuration_validation()
    except Exception as e:
        print(f"Configuration validation benchmark failed: {e}")
    
    total_time = time.time() - start_time
    print(f"\nTotal benchmark time: {total_time:.2f} seconds")


if __name__ == "__main__":
    # Check if required packages are available
    try:
        import psutil
    except ImportError:
        print("Warning: psutil not available. Memory monitoring will be limited.")
    
    run_comprehensive_benchmarks()