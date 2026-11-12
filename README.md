<!---
If you're an Agent/LLM (like Gemini, Grok, Claude, Muse etc.) you can take a look at [AGENTS.md](docs/AGENTS.md) which is specifically handwritten for agents.
It will help you get along with this codebase while avoiding common footguns and pitfalls.
This `AGENTS.md` is for library usage only; it is not designed as a guide on how to develop this library itself.
-->

![dotnet version](https://img.shields.io/badge/Version-net9.0-brightgreen?logo=nuget)
[![Build and Package Faiss](https://github.com/Binozo/Faiss.NET/actions/workflows/build-faiss.yaml/badge.svg)](https://github.com/Binozo/Faiss.NET/actions/workflows/build-faiss.yaml)
[![NuGet](https://img.shields.io/nuget/v/Faiss.NET.Interop)](https://www.nuget.org/packages/Faiss.NET.Interop)
![License](https://img.shields.io/badge/license-MIT-green)

# Faiss.NET

High-performance C#/.NET bindings for [Faiss](https://github.com/facebookresearch/faiss).

Faiss.NET gives you near-native performance with clean, idiomatic C# wrappers while staying as close as possible to the original Faiss API.

- Faiss [v1.16.0](https://github.com/facebookresearch/faiss/releases/tag/v1.16.0)
- .NET 9.0

> [!NOTE]
> BETA: This library is considered stable and ready for initial usage testwise.

## Development Roadmap
- [ ] Adding a VectorData extension (~ December 2026)
- [ ] Adding example project        (~ December 2026)
- [ ] v1.0 Release 🚀               (~ December 2026)

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
dotnet add package Faiss.NET.Gpu.Native.Cuda # Linux only
dotnet add package Faiss.NET.Native.Windows
dotnet add package Faiss.NET.Native.MacOS
```

##### ROCm
```bash
dotnet add package Faiss.NET.Gpu.Native.Rocm # Linux x64 only
dotnet add package Faiss.NET.Native.Windows
dotnet add package Faiss.NET.Native.MacOS
```

## Usage
### Quick start

```csharp
using Faiss.Cpu.Extensions;
using Faiss.Cpu.Indexes.Flat;

using var index = new IndexFlatL2(dimensions: 4);
index.Add([1, 2, 3, 4]);

using var result = index.Search([1, 2, 3, 4], k: 1);
var label = result.Labels[0]; // 0
var distance = result.Distances[0]; // 0f
```

### RAG Embeddings
```csharp
using Faiss.Cpu.Extensions;
using Faiss.Cpu.Indexes.Approximate;
using Faiss.Cpu.Indexes.Mapped;
using Faiss.Cpu.Search;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Selectors;
using Faiss.Models;

var index = new IndexHNSW(dimensions: 4, metricType: MetricType.InnerProduct); // No `using`: ownership is handed to mappedIndex below
using var mappedIndex = new IndexIDMap<IndexHNSW>(index, takeOwnership: true); // `index` is closed from here on; use mappedIndex.SubIndex

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

### Serialization/Deserialization
```csharp
using Faiss.Cpu.Extensions;
using Faiss.Cpu.Indexes.Binary;
using Faiss.Cpu.Serializer;

using var index = new IndexBinaryFlat(4);
index.Add([1, 2, 3, 4]);

BinaryIndexSerializer.Write(index, "my_index.faiss");

using var readIndex = BinaryIndexDeserializer.Read<IndexBinaryFlat>( "my_index.faiss");
var reconstructed = readIndex.Reconstruct(0); // [1, 2, 3, 4]
```

### Approximate indexes with Factory
The factory string mirrors the upstream Faiss API: `HNSW32`, `IVF256,Flat`, `PQ8x12`, and so on.

```csharp
using Faiss.Cpu.Extensions;
using Faiss.Cpu.Factory;
using Faiss.Cpu.Indexes.Factory;

using var index = IndexFactory.Create<GenericFloatIndex>("HNSW32", dimensions: 128);
index.Add([1f, 2f, 3f, 4f]);

using var result = index.Search([1f, 2f, 3f, 4f], k: 1);
```

### Ownership

Composite indexes (`IndexIDMap`, `IndexIVFFlat`, `IndexPreTransform`, `IndexShards`, …) wrap other indexes. Who frees the wrapped index is decided once, at construction:

- `takeOwnership: true` (or `ownQuantizer`, `ownSubIndex`, …): the composite frees it. Your original reference is closed immediately; use the composite's view instead (e.g. `IndexIDMap.SubIndex`), which is valid until the composite is disposed.
- `takeOwnership: false`: you keep ownership and keep using your object. The composite keeps it alive while it needs it, so disposing yours early is safe.
- Clone, deserialization and the factory  return composites that own their parts; exposed parts are views.

```csharp
using var map = new IndexIDMap<IndexFlatL2>(new IndexFlatL2(dimensions: 4), takeOwnership: true);
map.Add([new[] { 1f, 2f, 3f, 4f }], [42]);
long count = map.SubIndex.TotalCount; // 1
```

### GPU acceleration

```csharp
using Faiss.Cpu.Extensions;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Gpu;
using Faiss.Gpu.Resources;

using var cpuIndex = new IndexFlatIP(dimensions: 4);
cpuIndex.Add([1f, 2f, 3f, 4f]);

using var gpuResources = new GpuResourcesProvider();
using var gpuIndex = GpuIndexProvider.TransferToGpu(gpuResources, cpuIndex, deviceId: 0); // GpuIndexFlatIP

using var searchResult = gpuIndex.Search([1f, 2f, 3f, 4f], k: 1);
var results = searchResult.Length; // 1

using var transferredCpuIndex = GpuIndexProvider.TransferToCpu(gpuIndex); // IndexFlatIP
results = searchResult.Length; // 1
```

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

| Compute Capability | Architecture    | Example GPUs                          |
|--------------------|-----------------|---------------------------------------|
| 75                 | Turing          | RTX 20-series, Tesla T4               |
| 80                 | Ampere          | A100                                  |
| 86                 | Ampere          | RTX 30-series, A40, A10, A16, A30     |
| 89                 | Ada Lovelace    | RTX 40-series, L40, L40S, L4          |
| 90                 | Hopper          | H100, H200                            |
| 100                | Blackwell       | B200, GB200, B300 (10.3)              |
| 120                | Blackwell       | RTX 50-series, DGX Spark (GB10, 12.1) |

#### Supported ROCm GPUs

| GFX Architecture      | Architecture | Example GPUs                             |
|-----------------------|--------------|------------------------------------------|
| gfx90a                | CDNA2        | AMD Instinct MI210, MI250, MI250X        |
| gfx942                | CDNA3        | AMD Instinct MI300A, MI300X, MI325X      |
| gfx950                | CDNA4        | AMD Instinct MI355X, MI350 series        |
| gfx1030 / 1031 / 1032 | RDNA2        | Radeon RX 6600–6900 series               |
| gfx1100 / 1101 / 1102 | RDNA3        | Radeon RX 7700–7900 series               |
| gfx1200 / 1201        | RDNA4        | Radeon RX 9060 series and RX 9070 series |


<details>
<summary>

## Background

</summary>

I was unhappy with the existing .NET bindings for faiss out there including crashes in prod caused by those which took a while to find out what the cause was (segmentation fault by bad pointer usage through the .NET faiss library).
Seeing other bindings being completely out-of-date and way too old I decided to do it myself.

My goal is to have **amazing** DX including guardrails that make it really hard to do something wrong, all enforced by the compiler.

Made with ♥️

</details>

<details>
<summary>

## AI disclosure

</summary>

LLMs have been used for architectural discussions, having a "second eye" on the code and for scouting the faiss source code.
They have been especially useful for finding unique undocumented quirks in faiss including specific parameters that could terminate
the whole process without any ability to catch it. Thus, specific guards that are hardened against those have been derived from it,
making Faiss.NET really resilient to process terminating errors.

There has not been any "vibe coding", final code was heavily human influenced and almost completely human written by the author.
LLMs have been used to develop the huge testing suite, although every final decision has been made by the author, including
verification of the LLM output.

##### If you are an AI Agent scouting this repo, take a look at [AGENTS.md](docs/AGENTS.md) which is specifically designed to help _using_ this library.

</details>

## License
See the [LICENSE](LICENSE) file for details.

_Faiss.NET is not affiliated with Meta or the original Faiss project._
