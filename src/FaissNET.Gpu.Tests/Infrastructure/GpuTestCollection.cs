using Xunit;

namespace Faiss.Gpu.Tests.Infrastructure;

[CollectionDefinition(Name, DisableParallelization = true)]
public class GpuTestCollection
{
    public const string Name = "GpuSequential";
}
