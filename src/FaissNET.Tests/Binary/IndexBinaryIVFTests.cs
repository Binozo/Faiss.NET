using System.Threading.Tasks;
using Faiss.Cpu.Indexes;
using Faiss.Cpu.Indexes.Binary;
using Xunit;

namespace Faiss.Tests.Binary;

/// <summary>
/// Tests for binary IVF indexes.
/// </summary>
public class IndexBinaryIVFTests
{
    [Fact]
    public async Task TrainAddAndSearch_BinaryIVF_ReturnsExactHammingDistance()
    {
        int dimensions = 16;
        int nlist = 2;
        using var index = new IndexBinaryIVF(dimensions, nlist);

        Assert.False(index.IsTrained);
        Assert.Equal(dimensions, index.Dimensions);
        Assert.Equal(0, index.TotalCount);

        int numTrainVectors = 80;
        byte[] trainingData = new byte[numTrainVectors * 2];
        for (int i = 0; i < trainingData.Length; i++)
        {
            trainingData[i] = (byte)(i % 255);
        }

        await index.TrainAsync(numTrainVectors, trainingData);
        Assert.True(index.IsTrained);
        Assert.Equal(0, index.TotalCount);

        index.NProbe = 2;
        Assert.Equal(2, index.NProbe);

        byte[] targetVector = { 255, 0 };
        index.Add(1, targetVector);
        Assert.Equal(1, index.TotalCount);

        int[] distances = new int[1];
        long[] labels = new long[1];
        index.Search(1, targetVector, 1, distances, labels);

        Assert.Equal(0, labels[0]);
        Assert.Equal(0, distances[0]);
    }
}
