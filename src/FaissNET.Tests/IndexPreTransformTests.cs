using Faiss.Cpu.Extensions;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.Transform;
using Faiss.Cpu.Transforms;
using Xunit;

namespace Faiss.Tests;

/// <summary>
/// Tests for index pre-transformation pipelines.
/// </summary>
public class IndexPreTransformTests
{
    [Fact]
    public async Task PcaTransform_ReducesDimensions_AndSearchesCorrectly()
    {
        int dIn = 16;
        int dOut = 4;
        int numVectors = 100;

        using var pca = new PCAMatrix(dIn, dOut);

        float[] trainingData = new float[numVectors * dIn];
        for (int i = 0; i < trainingData.Length; i++)
        {
            trainingData[i] = i * 0.1f;
        }

        await pca.TrainAsync(numVectors, trainingData);

        using var baseIndex = new IndexFlatL2(dOut);
        using var preTransformIndex = new IndexPreTransform(pca, baseIndex);

        Assert.Equal(dIn, preTransformIndex.Dimensions);
        Assert.Equal(0, preTransformIndex.TotalCount);

        float[] targetVector = new float[16];
        for (int i = 0; i < 16; i++)
        {
            targetVector[i] = 1.0f;
        }

        preTransformIndex.Add(1, targetVector);
        Assert.Equal(1, preTransformIndex.TotalCount);

        float[] distances = new float[1];
        long[] labels = new long[1];
        preTransformIndex.Search(1, targetVector, 1, distances, labels);

        Assert.Equal(0, labels[0]);
        Assert.Equal(0.0f, distances[0]);
    }
}
