# Faiss.NET

C# bindings for [Faiss](https://github.com/facebookresearch/faiss).

- Faiss [v1.14.1](https://github.com/facebookresearch/faiss/releases/tag/v1.14.1)
- .NET 9.0

This is a bare implementation, currently only `IndexFlatL2` and `IndexFlatIP` are implemented. Will be extended over time.

> [!WARNING]
> This library is under active construction and currently not usable yet.

## Installation

### Prerequisites
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

## NuGet