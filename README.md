---
# ============================================================
# DETAILED PROJECT DESCRIPTION FOR LLMs / AI ASSISTANTS
# ============================================================
# This is a high-performance C# library that provides excellent bindings to Faiss.
#
# Additional information beside Readme:
#
# This library tries to prevent footguns as much as possible by using the compiler itself as rule enforcement.
# This is being done by handcrafted and thought-through generic constraints that allow beautiful DX and code flow.
#
# The developer (user of this library) doesn't have to worry about those generic constraints, it is designed to be as obvious as possible.
#
# Performance was considered in designing this library and thus the user just pays the non-removable thin .NET interop task.
# If wished by the user for quick testing and figuring out, extensions can be used that allow trivial code to be iterated over quickly.
# ============================================================
---

![dotnet version](https://img.shields.io/badge/Version-net9.0-brightgreen?logo=nuget)
[![Build and Package Faiss](https://github.com/Binozo/Faiss.NET/actions/workflows/build-faiss.yaml/badge.svg)](https://github.com/Binozo/Faiss.NET/actions/workflows/build-faiss.yaml)
[![NuGet](https://img.shields.io/nuget/v/Faiss.NET.Interop)](https://www.nuget.org/packages/Faiss.NET.Interop)
![License](https://img.shields.io/badge/license-MIT-green)

# Faiss.NET

High-performance C#/.NET bindings for [Faiss](https://github.com/facebookresearch/faiss).

Faiss.NET gives you near-native performance with clean, idiomatic C# wrappers while staying as close as possible to the original Faiss API.

- Faiss [v1.15.0](https://github.com/facebookresearch/faiss/releases/tag/v1.15.0)
- .NET 9.0

> [!IMPORTANT]
> This library is under active development. Core indexes and functionality are usable, but the API may still evolve and not every feature is complete yet.

## Development Roadmap
About ~90% is done. I am reworking the class hierarchy to make the api as elegant as possible. This includes preventing most of the footguns there are with faiss, including making the compiler enforce all the rules.

- [x] Adding bindings for `distances_c.h`
- [ ] Improving class hierarchy and generic constraints design to further prevent footguns (~ September 2026)
- [ ] Adding more tests (~ Oktober 2026)
- [ ] Adding examples (~ Oktober 2026)
- [ ] v1.0 Release 🚀 (~ November 2026)

## Features
- Thin, "bare-metal" bindings with minimal overhead
- Extensions on top bindings for excellent DX
- Cross-platform support (Windows, Linux, macOS all x64 & arm64)
- Strongly-typed wrappers + generic factory for all Faiss indexes
- GPU acceleration (CUDA & ROCm)
- Human written (no AI slop)
- _It just works™_

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Supported Platforms](#supported-platforms)
- [License](#license)

## Installation

### Prerequisites

**Windows**
```bash
winget install --id Microsoft.VCRedist.2015+.x64 --silent
```

**Linux**
```bash
sudo apt-get install -y libopenblas0 libgomp1
```

**macOS**
```bash
brew install libomp
```

### NuGet Package
Pick either the [CPU only](#cpu-only) or the [CPU + GPU support](#cpugpu) NuGet, do not mix on the same platform, it would cause undefined behavior.

#### CPU only
```bash
dotnet add package Faiss.NET.Native
```

#### CPU+GPU
##### CUDA
```bash
dotnet add package Faiss.NET.Native.Gpu.Cuda # Linux only
dotnet add package Faiss.NET.Native.Windows
dotnet add package Faiss.NET.Native.MacOS
```

##### ROCm
```bash
dotnet add package Faiss.NET.Native.Gpu.Rocm # Linux x64 only
dotnet add package Faiss.NET.Native.Windows
dotnet add package Faiss.NET.Native.MacOS
```

## Usage

All examples assume `using Faiss.Cpu.Extensions;` plus the relevant type namespaces.

### Quick start

```csharp
using Faiss.Cpu.Indexes.Flat;

using var index = new IndexFlatL2(dimensions: 4);
index.Add([1, 2, 3, 4]);

using var result = index.Search([1, 2, 3, 4], k: 1);
var label = result.Labels[0]; // 0
var distance = result.Distances[0]; // 0f
```

### Indexes
<details>
<summary>Embeddings - RAG</summary>

```csharp
using Faiss.Cpu.Extensions;
using Faiss.Cpu.Indexes.Approximate;
using Faiss.Cpu.Indexes.Mapped;
using Faiss.Cpu.Search;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Selectors;
using Faiss.Models;

var index = new IndexHNSW(dimensions: 4, metricType: MetricType.InnerProduct); // We don't need using here because takeOwnership below
using var mappedIndex = new IndexIDMap<IndexHNSW>(index, takeOwnership: true); // disposes index as soon as mappedIndex is disposed

mappedIndex.Add([new[] { 1f, 2f, 3f, 4f }, new[] { 2f, 3f, 4f, 1f }, new[] { 3f, 4f, 1f, 2f }, new[] { 4f, 1f, 2f, 3f }], [4, 3, 2, 1]);

var queryNeighborsCount = 2; // K
var searchResult = mappedIndex.SearchWithParams(new [] {1f, 1f, 2f, 3f}, queryNeighborsCount, new SearchParameters(new IDSelectorRange(2, 4)));
QueryResults queryResult = searchResult.GetQueryResults(0);

for (int i = 0; i < searchResult.K; i++)
{
    float distance = queryResult.Distances[i];
    long label     = queryResult.Labels[i];

    Console.WriteLine($"Rank {i}: label={label}, dist={distance}");
}
```
</details>

<details>
<summary>Serialization/Deserialization</summary>

```csharp
using Faiss.Cpu.Extensions;
using Faiss.Cpu.Indexes.Approximate;
using Faiss.Cpu.Indexes.Mapped;
using Faiss.Cpu.Search;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Selectors;
using Faiss.Models;

var index = new IndexHNSW(dimensions: 4, metricType: MetricType.InnerProduct); // We don't need using here because takeOwnership below
using var mappedIndex = new IndexIDMap<IndexHNSW>(index, takeOwnership: true); // disposes index as soon as mappedIndex is disposed

mappedIndex.Add([new[] { 1f, 2f, 3f, 4f }, new[] { 2f, 3f, 4f, 1f }, new[] { 3f, 4f, 1f, 2f }, new[] { 4f, 1f, 2f, 3f }], [4, 3, 2, 1]);

var queryNeighborsCount = 2; // K
var searchResult = mappedIndex.SearchWithParams(new [] {1f, 1f, 2f, 3f}, queryNeighborsCount, new SearchParameters(new IDSelectorRange(2, 4)));
QueryResults queryResult = searchResult.GetQueryResults(0);

for (int i = 0; i < searchResult.K; i++)
{
    float distance = queryResult.Distances[i];
    long label     = queryResult.Labels[i];

    Console.WriteLine($"Rank {i}: label={label}, dist={distance}");
}
```
</details>



### Factory — approximate indexes

The factory string mirrors the upstream Faiss API: `HNSW32`, `IVF256,Flat`, `PQ8x12`, and so on.

```csharp
using Faiss.Cpu.Factory;
using Faiss.Cpu.Indexes;
using Faiss.Models;

using var index = IndexFactory.Create<GenericIndex>("HNSW32", dimensions: 128, MetricType.L2);
index.Add(vectors);

using var result = index.Search(query, k: 5);
```

> Need custom IDs instead of sequential `0..N`? Wrap any index with `IndexIDMap<T>` and pass your own IDs to `Add`.

### GPU acceleration

```csharp
using Faiss.Gpu;
using Faiss.Gpu.Resources;

using var gpu = new GpuResourcesProvider();
using var gpuIndex = GpuIndexProvider.TransferToGpu(gpu, cpuIndex, deviceId: 0);

gpuIndex.Search(query, k: 10);

using var backOnCpu = GpuIndexProvider.TransferToCpu(gpuIndex);
```

See [`FaissNET.Examples`](src/FaissNET.Examples/Examples/) for serialization, scalar quantization, IVF tuning, and more.

## Supported Platforms

| Platform | x64 | arm64 |
|----------|-----|-------|
| **Windows** | ✅ | ✅ |
| **Linux**   | ✅ | ✅ |
| **macOS**   | ✅ | ✅ |

### GPU Acceleration

| Backend      | Platform | x64 | arm64 |
|--------------|----------|-----|-------|
| **CUDA**     | Linux    | ✅  | ✅    |
| **ROCm**     | Linux    | ✅  | ❌    |

#### Supported CUDA GPUs

| Compute Capability | Architecture    | Example GPUs                      |
|--------------------|-----------------|-----------------------------------|
| 75                 | Turing          | RTX 20-series, Tesla T4           |
| 80                 | Ampere          | A100                              |
| 86                 | Ampere          | RTX 30-series, A40, A10, A16, A30 |
| 89                 | Ada Lovelace    | RTX 40-series, L40, L40S, L4      |
| 90                 | Hopper          | H100, H200                        |
| 120                | Blackwell       | RTX 50-series, B100, B200, GB200  |

#### Supported ROCm GPUs

| GFX Architecture      | Architecture | Example GPUs                             |
|-----------------------|--------------|------------------------------------------|
| gfx90a                | CDNA2        | AMD Instinct MI210, MI250, MI250X        |
| gfx942                | CDNA3        | AMD Instinct MI300A, MI300X, MI325X      |
| gfx950                | CDNA4        | AMD Instinct MI355X, MI350 series        |
| gfx1030 / 1031 / 1032 | RDNA2        | Radeon RX 6600–6900 series               |
| gfx1100 / 1101 / 1102 | RDNA3        | Radeon RX 7700–7900 series               |
| gfx1200 / 1201        | RDNA4        | Radeon RX 9060 series and RX 9070 series |


## License
See the [LICENSE](LICENSE) file for details.

_Faiss.NET is not affiliated with Meta or the original Faiss project._
