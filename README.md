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

Then pick a native package:

Either [cpu only](#cpu-only) **or** with additional [gpu](#cpugpu) support.

#### CPU only
```shell
$ dotnet add package Faiss.NET.Native
```

#### CPU+GPU
```shell
$ dotnet add package Faiss.NET.Native.Gpu.Cuda # Linux only
$ dotnet add package Faiss.NET.Windows
$ dotnet add package Faiss.NET.MacOS
```

or

```shell
$ dotnet add package Faiss.NET.Native.Gpu.Rocm # Linux x64 only
$ dotnet add package Faiss.NET.Windows
$ dotnet add package Faiss.NET.MacOS
```

## Usage
### Basic
```csharp
using Faiss.NET;

int dimensions = 4;
using var index = new FaissIndex<FaissIndexFlatL2>(new FaissIndexFlatL2(dimensions));

float[] vectors = { 1.0f, 2.0f, 3.0f, 4.0f };

index.Add(vectors);

using var searchResult = index.Search(vectors, 1);

float distance = searchResult.Distances[0]; // 0.0f
long label = searchResult.Labels[0]; // 0
```

### Examples
#### Low-Level index usage
```csharp
using Faiss.NET;

int dimensions = 2;
using var index = new FaissIndexFlatIP(dimensions);

float[] vectors = { 2.0f, 3.0f };
index.Add(1, vectors);

float[] distances = new float[1];
long[] labels = new long[1];

index.Search(1, vectors, 1, distances, labels);

long foundLabel = labels[0]; // 0
float foundDistance = distances[0]; // 13
```
