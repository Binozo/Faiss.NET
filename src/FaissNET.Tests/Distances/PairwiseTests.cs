using Faiss.Cpu.Distances;
using Faiss.Tests.Data;
using Xunit;

namespace Faiss.Tests.Distances;

/// <summary>
/// Tests for pairwise operations.
/// </summary>
public class PairwiseTests
{
    [Fact]
    public void PairwiseL2Sqr_CalculatesCorrect()
    {
        float[] distanceMatrix = new float[1];
        float[] expectedDistanceMatrix = [0.353011966f];
        
        Pairwise.L2Sqr(Embeddings.Dimension, 1, Embeddings.Query, 1, Embeddings.Documents[0], distanceMatrix);

        Assert.Equal(expectedDistanceMatrix, distanceMatrix);
    }
}