using Faiss.Cpu.Distances;
using Xunit;

namespace Faiss.Tests.Distances;

/// <summary>
/// Tests for Blas operations.
/// </summary>
public class BlasTests
{
    [Fact]
    public void DistanceComputeBlasThreshold_Initial_And_Modification()
    {
        Assert.Equal(128000, Blas.DistanceComputeBlasThreshold);

        var newThreshold = 128_000 / 2;
        Blas.DistanceComputeBlasThreshold = newThreshold;
        
        Assert.Equal(newThreshold, Blas.DistanceComputeBlasThreshold);
    }
}