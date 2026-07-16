![dotnet version](https://img.shields.io/badge/Version-net9.0-brightgreen?logo=nuget)
[![Build and Package Faiss](https://github.com/Binozo/Faiss.NET/actions/workflows/build-faiss.yaml/badge.svg)](https://github.com/Binozo/Faiss.NET/actions/workflows/build-faiss.yaml)
[![NuGet](https://img.shields.io/nuget/v/Faiss.NET.Interop)](https://www.nuget.org/packages/Faiss.NET.Interop)
![License](https://img.shields.io/badge/license-MIT-green)

# Faiss.NET

High-performance C#/.NET bindings for [Faiss](https://github.com/facebookresearch/faiss).

Faiss.NET gives you near-native performance with clean, idiomatic C# wrappers while staying as close as possible to the original Faiss API.

- Faiss [v1.14.3](https://github.com/facebookresearch/faiss/releases/tag/v1.14.3)
- .NET 9.0

> [!IMPORTANT]
> This library is under active development. Core indexes and functionality are usable, but the API may still evolve and not every feature is complete yet.

## Development Roadmap
About ~90% is done. I still need to add some bindings such as the distance util functions, polish the generic constraints and class hierarchy design before I feel comfortable publishing a v1.0 release.

- [x] Adding bindings for `distances_c.h`
- [ ] Improving class hierarchy and generic constraints design to further prevent footguns
- [ ] Adding more tests
- [ ] Adding examples

## Features
- Thin, "bare-metal" bindings with minimal overhead
- Extensions on top bindings for excellent DX
- Cross-platform support (Windows, Linux, macOS all x64 & arm64)
- Strongly-typed wrappers + generic factory for all Faiss indexes
- GPU acceleration (CUDA & ROCm)
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
sudo apt-get install -y libopenblas0 libgomp1 libgfortran5
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

### Basic Flat index

```csharp
using Faiss.NET;

int dimensions = 4;
using var index = new IndexFlatL2(dimensions);

float[] vector = [1.0f, 2.0f, 3.0f, 4.0f];
index.Add(vector);

using var result = index.Search(vector, k: 1);

float distance = result.Distances[0]; // 0.0f
long label = result.Labels[0]; // 0
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
