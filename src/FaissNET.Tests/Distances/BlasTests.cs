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
        Blas.DistanceComputeBlasThreshold = 128_000;
        
        Assert.Equal(128_000, Blas.DistanceComputeBlasThreshold);
    }
    
    [Fact]
    public void DistanceComputeBlasBlockSizes_Initial_And_Modification()
    {
        Blas.DistanceComputeBlasBlockSizes = 2048;
        
        Assert.Equal(2048, Blas.DistanceComputeBlasBlockSizes);
    }
    
    [Fact]
    public void DistanceComputeBlasDatabaseBlockSizes_Initial_And_Modification()
    {
        Blas.DistanceComputeBlasDatabaseBlockSizes = 512;
        
        Assert.Equal(512, Blas.DistanceComputeBlasDatabaseBlockSizes);
    }
    
    [Fact]
    public void DistanceComputeComputeMinKReservoir_Initial_And_Modification()
    {
        Blas.DistanceComputeMinKReservoir = 256;
        
        Assert.Equal(256, Blas.DistanceComputeMinKReservoir);
    }
}