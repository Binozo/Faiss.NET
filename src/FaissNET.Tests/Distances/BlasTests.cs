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
    
    [Fact]
    public void DistanceComputeBlasBlockSizes_Initial_And_Modification()
    {
        Assert.Equal(4096, Blas.DistanceComputeBlasBlockSizes);

        var newThreshold = 4096 / 2;
        Blas.DistanceComputeBlasBlockSizes = newThreshold;
        
        Assert.Equal(newThreshold, Blas.DistanceComputeBlasBlockSizes);
    }
    
    [Fact]
    public void DistanceComputeBlasDatabaseBlockSizes_Initial_And_Modification()
    {
        Assert.Equal(1024, Blas.DistanceComputeBlasDatabaseBlockSizes);

        var newThreshold = 1024 / 2;
        Blas.DistanceComputeBlasDatabaseBlockSizes = newThreshold;
        
        Assert.Equal(newThreshold, Blas.DistanceComputeBlasDatabaseBlockSizes);
    }
    
    [Fact]
    public void DistanceComputeComputeMinKReservoir_Initial_And_Modification()
    {
        Assert.Equal(1024, Blas.DistanceComputeMinKReservoir);

        var newThreshold = 1024 / 2;
        Blas.DistanceComputeMinKReservoir = newThreshold;
        
        Assert.Equal(newThreshold, Blas.DistanceComputeMinKReservoir);
    }
}