![License](https://img.shields.io/badge/license-MIT-green)
[![Build and Package Faiss](https://github.com/Binozo/Faiss.NET/actions/workflows/build-faiss.yaml/badge.svg)](https://github.com/Binozo/Faiss.NET/actions/workflows/build-faiss.yaml)

# Faiss.NET

C# bindings for [Faiss](https://github.com/facebookresearch/faiss).

- Faiss [v1.14.1](https://github.com/facebookresearch/faiss/releases/tag/v1.14.1)
- .NET 9.0

> [!WARNING]
> This library is under active construction and currently not usable yet.

#### Implemented Indexes
- IndexFlatIP
- IndexFlatL2
- IndexHNSW

Also, a generic Index factory is implemented which can be used to instantiate all other Faiss supported indexes.

This library aims to be as "bare metal" as possible while being straightforward to work with.

Additionally, Index serialization/deserialization and GPU Indexes + sharding is supported.

#### Supported Platforms:

| Platform | x64 | arm64 |
|----------|-----|-------|
| Windows  | ✅   | ✅     |
| Linux    | ✅   | ✅     |
| MacOS    | ✅   | ✅     |

#### GPU Acceleration:

| Platform       | x64 | arm64 |
|----------------|-----|-------|
| CUDA (Windows) | ✅   | ❌     |
| CUDA (Linux)   | ✅   | ✅     |
| ROCm (Linux)   | ✅   | ❌     |

## Installation

### Prerequisites
#### Windows

You need a C++ redistributable installed. E.g.

```shell
$ winget install --id Microsoft.VCRedist.2015+.x64 --silent
```

#### Linux

OpenBLAS, OpenMP and Fortran runtimes must be installed. E.g.

```shell
$ sudo apt-get install -y libopenblas0 libgomp1 libgfortran5
```

#### MacOS

An OpenMP runtime is required. E.g.
```shell
$ brew install libomp
```

### NuGet
Get the base NuGet:

```shell
$ dotnet add package Faiss.NET
```