using Faiss.Cpu.Exceptions;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Xunit;

namespace Faiss.Tests;

/// <summary>
/// Tests for IVF flat indexes.
/// </summary>
public class IndexIVFFlatTests
{
    [Fact]
    public void Add_WithoutTraining_ThrowsFaissUntrainedException()
    {
        int dimensions = 4;
        using var quantizer = new IndexFlatL2(dimensions);
        using var index = new IndexIVFFlat(quantizer, dimensions, nlist: 2);

        float[] vectors = { 1.0f, 2.0f, 3.0f, 4.0f };

        Assert.Throws<FaissUntrainedException>(() => index.Add(1, vectors));
    }

    [Fact]
    public async Task Train_Add_AndSearch_ReturnsExactMatch()
    {
        int dimensions = 4;
        int nlist = 2;

        using var quantizer = new IndexFlatL2(dimensions);
        using var index = new IndexIVFFlat(quantizer, dimensions, nlist);

        float[] trainingData = new float[78 * dimensions];
        for (int i = 0; i < trainingData.Length; i++)
        {
            trainingData[i] = i * 0.1f;
        }

        await index.TrainAsync(78, trainingData);
        Assert.True(index.IsTrained);
        Assert.Equal(0, index.TotalCount);

        float[] targetVector = { 10.0f, 10.0f, 10.0f, 10.0f };
        index.Add(1, targetVector);
        Assert.Equal(1, index.TotalCount);

        float[] distances = new float[1];
        long[] labels = new long[1];
        index.Search(1, targetVector, 1, distances, labels);

        Assert.Equal(0, labels[0]);
        Assert.Equal(0.0f, distances[0]);
    }
}
