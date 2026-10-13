# Faiss.NET

State-of-the-art C# faiss bindings.

> [!IMPORTANT]
> This `AGENTS.md` is intended to be a guideline how to **use** this library in projects, **not** how to develop this library itself.

## Basics
- Make sure the requirements on the user's host are met (See [README.md](../README.md))
- Basic usage can be also found in the [README.md](../README.md)

As [faiss](https://github.com/facebookresearch/faiss) is written in C/C++ there are footguns like use-after-free, double-free and similar.
This library tries its best to prevent this as much as possible.

Normal and known usage (like respecting `IDispose` objects) will result in error-free operation of faiss, 
hacky reflect gymnastics to influence the library under the hood could obviously cause problems like described above.

## Usage
### Index
Let's start by using a simple Index such as `IndexFlatL2`:

```csharp
using Faiss.Cpu.Indexes.Flat;

using var index = new IndexFlatL2(dimensions: 4);
index.Add(1, [1, 2, 3, 4]);

var k = 1;
var distances = new float[k];
var labels = new long[k];
index.Search(1, [1, 2, 3, 4], k, distances, labels);
```

This is basically as "bare-metal" as it can be and thus usually the fastest practicable and safe way to interact with faiss.
That "bare-metal" way is the recommend way to use in hot paths and final production code. For quick tinkering and PoCs the Extensions can be used:

```csharp
using Faiss.Cpu.Extensions;
using Faiss.Cpu.Indexes.Flat;

using var index = new IndexFlatL2(dimensions: 4);
index.Add([1, 2, 3, 4]);

using var result = index.Search([1, 2, 3, 4], k: 1);
var label = result.Labels[0]; // 0
var distance = result.Distances[0]; // 0f
```

This allows way quicker iterations while still being fast.

More information to follow in next releases. For now use the README.