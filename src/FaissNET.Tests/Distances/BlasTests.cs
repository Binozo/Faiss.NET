using Faiss.Cpu.Distances;
using Xunit;

namespace Faiss.Tests.Distances;

/// <summary>
/// These knobs are process-global faiss state, so every test restores the original value.
/// </summary>
public class BlasTests
{
    [Fact]
    public void Threshold_RoundTrips()
    {
        var original = Blas.DistanceComputeBlasThreshold;
        try
        {
            Blas.DistanceComputeBlasThreshold = original + 7;
            Assert.Equal(original + 7, Blas.DistanceComputeBlasThreshold);
        }
        finally
        {
            Blas.DistanceComputeBlasThreshold = original;
        }

        Assert.Equal(original, Blas.DistanceComputeBlasThreshold);
    }

    [Fact]
    public void EachKnobIsWiredToItsOwnNativeSetting()
    {
        var threshold = Blas.DistanceComputeBlasThreshold;
        var queryBlock = Blas.DistanceComputeBlasBlockSizes;
        var databaseBlock = Blas.DistanceComputeBlasDatabaseBlockSizes;
        var reservoir = Blas.DistanceComputeMinKReservoir;

        try
        {
            // Distinct values: a property wired to the wrong native getter/setter reads back a neighbour's value.
            Blas.DistanceComputeBlasThreshold = 1001;
            Blas.DistanceComputeBlasBlockSizes = 1002;
            Blas.DistanceComputeBlasDatabaseBlockSizes = 1003;
            Blas.DistanceComputeMinKReservoir = 1004;

            Assert.Equal(1001, Blas.DistanceComputeBlasThreshold);
            Assert.Equal(1002, Blas.DistanceComputeBlasBlockSizes);
            Assert.Equal(1003, Blas.DistanceComputeBlasDatabaseBlockSizes);
            Assert.Equal(1004, Blas.DistanceComputeMinKReservoir);
        }
        finally
        {
            Blas.DistanceComputeBlasThreshold = threshold;
            Blas.DistanceComputeBlasBlockSizes = queryBlock;
            Blas.DistanceComputeBlasDatabaseBlockSizes = databaseBlock;
            Blas.DistanceComputeMinKReservoir = reservoir;
        }
    }
}
