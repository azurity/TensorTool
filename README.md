# TensorTool

A C++ header-only tensor manipulation library built on top of [Eigen](https://eigen.tuxfamily.org/).

## Overview

TensorTool provides convenient tensor operations for Eigen's Tensor module. It offers utilities for dimension manipulation, broadcasting, and tensor operations.

## Features

- **Dimension Operations**: `squeeze`, `unsqueeze` for adding/removing dimensions
- **Tensor Stacking**: `stack` to combine tensors along a new dimension
- **Broadcasting**: Element-wise operations with automatic broadcasting support
- **Sequence Utilities**: `splice` for manipulating dimension arrays

## Requirements

- C++17 or later
- CMake 3.10+
- Eigen3

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Testing

```bash
ctest
```

Or run the tests directly:

```bash
./TensorToolTests
```

## Usage

```cpp
#include <TensorTool/dims.hpp>
#include <TensorTool/ops.hpp>
#include <TensorTool/seq.hpp>

using namespace TensorTool;

// Squeeze/unsqueeze dimensions
auto t = Eigen::Tensor<double, 3>(2, 1, 3);
auto squeezed = squeeze<1>(t);  // Remove dimension 1
auto unsqueezed = unsqueeze<0>(t);  // Add dimension at position 0

// Stack tensors
auto t1 = Eigen::Tensor<float, 2>(2, 3);
auto t2 = Eigen::Tensor<float, 2>(2, 3);
auto stacked = stack<0>(t1, t2);  // Stack along dimension 0

// Broadcast operations
auto a = Eigen::Tensor<double, 2>(3, 1);
auto b = Eigen::Tensor<double, 2>(1, 4);
auto c = cwise_add(a, b);  // Broadcasting add
```

## Project Structure

```
├── include/TensorTool/
│   ├── dims.hpp    # Dimension operations
│   ├── ops.hpp     # Tensor operations
│   └── seq.hpp     # Sequence utilities
├── tests/          # Unit tests
└── CMakeLists.txt  # Build configuration
```

## License

MIT
