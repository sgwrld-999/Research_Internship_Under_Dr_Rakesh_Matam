# Test Suite for GRIFFIN

## Overview

This directory contains comprehensive test suites for the GRIFFIN (Group-Regularized Intrusion Flow Feature Integration Network) implementation. The tests are designed to validate functionality, performance, and robustness of all components.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # pytest fixtures and configurations  
├── test_griffin.py             # Unit tests for core components
├── test_integration.py         # Integration tests for workflows
├── test_performance.py         # Performance benchmarks
├── test_utils.py              # Test utilities and helpers
└── README.md                  # This file
```

## Test Categories

### 1. Unit Tests (`test_griffin.py`)
- **ConfigManager**: Configuration loading, validation, and error handling
- **Logger**: Logging functionality with console and file outputs
- **ReproducibilityManager**: Random seed management and reproducibility
- **DataCleaner**: Data cleaning with NaN/infinity handling
- **FeatureScaler**: Feature scaling and normalization
- **DataLoader**: Data loading and preparation pipeline

### 2. Integration Tests (`test_integration.py`)
- **Complete Workflow**: End-to-end data preprocessing pipeline
- **Configuration Validation**: Comprehensive config validation scenarios
- **Corrupted Data Handling**: Robustness with various data corruptions
- **Device Management**: CPU/GPU device selection
- **Stratified Splitting**: Class-balanced data splitting
- **Error Handling**: Invalid data format and edge case handling

### 3. Performance Tests (`test_performance.py`)
- **Scaling Analysis**: Performance with different data sizes
- **Memory Efficiency**: Memory usage optimization validation
- **Corruption Handling**: Performance impact of data cleaning
- **Configuration Validation**: Validation speed benchmarks

### 4. Test Utilities (`test_utils.py`)
- **TestDataGenerator**: Synthetic data generation (CICIoT-like, corrupted)
- **TestConfigBuilder**: Configuration builders for different scenarios
- **TempDirectoryManager**: Temporary file and directory management
- **AssertionHelpers**: Common test assertion utilities

## Running Tests

### Prerequisites
```bash
pip install pytest pytest-cov numpy pandas scikit-learn pyyaml
```

### Basic Test Execution
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_griffin.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Performance Benchmarks
```bash
# Run performance tests
python tests/test_performance.py

# Run specific benchmark
python -c "from tests.test_performance import benchmark_data_preprocessing; benchmark_data_preprocessing()"
```

## Test Fixtures

### Configuration Fixtures
- `sample_config`: Comprehensive configuration for testing
- `temp_config_file`: Temporary YAML configuration file
- `minimal_config`: Minimal valid configuration
- `comprehensive_config`: Full-featured configuration

### Data Fixtures
- `sample_data`: Synthetic CICIoT-like dataset (1000 samples, 50 features)
- `small_sample_data`: Quick test dataset (100 samples)
- `corrupted_data`: Data with NaN, infinity, and constant features
- `class_weights`: Balanced weights for imbalanced datasets

### Environment Fixtures
- `temp_directory`: Temporary directory for file operations
- `setup_reproducibility`: Ensures reproducible test results
- `mock_torch_model`: Mock PyTorch model for testing

## Test Data Generation

### CICIoT-like Data
The `TestDataGenerator.generate_ciciot_like_data()` creates synthetic data that mimics the characteristics of the CICIoT dataset:
- **Feature Groups**: Protocol (5), Packet (8), Flow (10), Statistical (15), Behavioral (12)
- **Class Imbalance**: Realistic distribution with dominant normal traffic
- **Correlation Structure**: Within-group feature correlations
- **Class Separability**: Class-dependent feature shifts

### Corrupted Data
The `TestDataGenerator.generate_corrupted_data()` creates data with various corruption issues:
- **NaN Values**: Missing data simulation
- **Infinite Values**: Overflow/underflow simulation
- **Constant Features**: Zero-variance features
- **Low Variance Features**: Near-constant features

## Mock Objects

### MockDataLoader
Lightweight data loader for testing without file I/O:
```python
loader = MockDataLoader(config)
loader.set_data(X, y)
data_splits = loader.prepare_data()
```

### Mock PyTorch Model
Simulated PyTorch model for testing without GPU dependencies:
```python
model = mock_torch_model()  # From fixture
output = model(input_tensor)
```

## Test Configuration Examples

### Minimal Configuration
```python
config = TestConfigBuilder.minimal_config()
# Contains only essential parameters for quick testing
```

### Comprehensive Configuration
```python
config = TestConfigBuilder.comprehensive_config()
# Contains all available parameters with realistic values
```

### Imbalanced Dataset Configuration
```python
config = TestConfigBuilder.imbalanced_config()
# Optimized for imbalanced datasets with focal loss and balanced metrics
```

## Performance Benchmarks

### Data Preprocessing Scaling
Tests preprocessing performance with datasets of increasing size:
- 1,000 samples → 50,000 samples
- Measures execution time and memory usage
- Provides samples/second processing rate

### Memory Efficiency
Validates memory usage with large datasets:
- Tracks memory delta during processing
- Tests with up to 100,000 samples
- Uses float32 for memory optimization

### Corruption Handling Performance
Tests performance impact of data cleaning:
- Different corruption rates (0% → 50%)
- Measures cleaning overhead
- Validates robustness

## Assertion Helpers

### Data Split Validation
```python
AssertionHelpers.assert_data_splits_valid(data_splits, original_size)
```

### Feature Quality Validation
```python
AssertionHelpers.assert_features_clean(X)  # No NaN/inf values
```

### Configuration Validation
```python
AssertionHelpers.assert_config_valid(config)  # Required sections present
```

### Reproducibility Validation
```python
AssertionHelpers.assert_reproducible_results(results1, results2)
```

## Error Testing Scenarios

### Invalid Data Formats
- String inputs instead of arrays
- Empty datasets
- Mismatched X and y lengths
- Single-class datasets

### Configuration Errors
- Missing required sections
- Invalid parameter values
- Inconsistent feature group specifications

### Resource Limitations
- Large dataset memory handling
- Device availability (CPU/GPU)
- File system permissions

## Continuous Integration

The test suite is designed for CI/CD environments:

### GitHub Actions Example
```yaml
- name: Run Tests
  run: |
    python -m pytest tests/ --cov=src --cov-report=xml
    
- name: Run Performance Tests
  run: |
    python tests/test_performance.py
```

### Test Coverage Goals
- **Unit Tests**: >90% code coverage
- **Integration Tests**: Critical workflow validation
- **Performance Tests**: Scalability validation

## Troubleshooting

### Common Issues

1. **Import Errors**: Install required dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. **Memory Errors**: Reduce test data sizes in `conftest.py`

3. **Slow Tests**: Run specific test files or use markers
   ```bash
   python -m pytest tests/test_griffin.py::TestConfigManager -v
   ```

4. **Device Errors**: Tests default to CPU-only mode

### Test Debugging

Enable verbose output:
```bash
python -m pytest tests/ -v -s --tb=long
```

Run individual test methods:
```bash
python -m pytest tests/test_griffin.py::TestConfigManager::test_load_valid_config -v
```

## Contributing to Tests

### Adding New Tests
1. Follow existing naming conventions (`test_*.py`)
2. Use appropriate fixtures from `conftest.py`
3. Include both positive and negative test cases
4. Add performance tests for new features

### Test Guidelines
- **Isolation**: Each test should be independent
- **Reproducibility**: Use fixed random seeds
- **Cleanup**: Use fixtures for temporary resources
- **Documentation**: Clear test names and docstrings

### Performance Test Guidelines
- **Baseline**: Establish performance baselines
- **Scaling**: Test with multiple data sizes
- **Resource Monitoring**: Track memory and time
- **Regression Detection**: Compare against previous results

This test suite ensures the GRIFFIN implementation is robust, performant, and production-ready.