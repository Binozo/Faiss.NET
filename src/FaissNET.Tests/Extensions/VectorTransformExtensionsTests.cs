using Faiss.Cpu.Extensions;
using Faiss.Cpu.Transforms;
using Faiss.Exceptions;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Extensions;

public class VectorTransformExtensionsTests
{
    private const int Dimensions = 8;
    private const int Count = 64;

    [Fact]
    public async Task TrainAsync_Memory_TrainsTheTransform()
    {
        using var transform = new CenteringTransform(Dimensions);
        Assert.False(transform.IsTrained);

        var train = Vectors.Random(seed: 230, count: Count, dimensions: Dimensions);
        await transform.TrainAsync(Count, train);

        Assert.True(transform.IsTrained);
    }

    [Fact]
    public async Task TrainAsync_Memory_MatchesSynchronousTraining()
    {
        var train = Vectors.Random(seed: 231, count: Count, dimensions: Dimensions);
        var sample = train.AsSpan(0, Dimensions).ToArray();

        using var asyncTransform = new CenteringTransform(Dimensions);
        await asyncTransform.TrainAsync(Count, train);

        using var syncTransform = new CenteringTransform(Dimensions);
        syncTransform.Train(Count, train);

        FloatAssert.Equal(syncTransform.Apply(1, sample), asyncTransform.Apply(1, sample));
    }

    [Fact]
    public async Task TrainAsync_MemoryList_FlattensChunksInOrder()
    {
        var train = Vectors.Random(seed: 232, count: Count, dimensions: Dimensions);

        var chunks = new List<ReadOnlyMemory<float>>();
        for (var i = 0; i < Count; i++)
        {
            chunks.Add(new ReadOnlyMemory<float>(train, i * Dimensions, Dimensions));
        }

        using var chunked = new CenteringTransform(Dimensions);
        await chunked.TrainAsync(chunks);

        using var flat = new CenteringTransform(Dimensions);
        flat.Train(Count, train);

        var sample = train.AsSpan(0, Dimensions).ToArray();
        FloatAssert.Equal(flat.Apply(1, sample), chunked.Apply(1, sample));
    }

    [Fact]
    public async Task TrainAsync_MemoryList_WrongChunkLength_Throws()
    {
        using var transform = new CenteringTransform(Dimensions);

        var chunks = new List<ReadOnlyMemory<float>>
        {
            new float[Dimensions],
            new float[Dimensions - 1],
        };

        await Assert.ThrowsAsync<ArgumentException>(() => transform.TrainAsync(chunks));
        Assert.False(transform.IsTrained);
    }

    [Fact]
    public async Task TrainAsync_MemoryList_Empty_IsNoOp()
    {
        using var transform = new CenteringTransform(Dimensions);

        await transform.TrainAsync(new List<ReadOnlyMemory<float>>());

        Assert.False(transform.IsTrained);
    }

    [Fact]
    public async Task TrainAsync_Memory_PropagatesBaseValidation()
    {
        using var transform = new CenteringTransform(Dimensions);

        await Assert.ThrowsAsync<ArgumentException>(() => transform.TrainAsync(4, new float[Dimensions]));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => transform.TrainAsync(-1, new float[Dimensions]));
    }

    [Fact]
    public async Task TrainAsync_MemoryList_PooledBufferIsNotObservedAfterReturn()
    {
        var first = Vectors.Random(seed: 233, count: Count, dimensions: Dimensions);
        var second = Vectors.Random(seed: 234, count: Count, dimensions: Dimensions);

        using var a = new CenteringTransform(Dimensions);
        await a.TrainAsync(Chunk(first));

        using var b = new CenteringTransform(Dimensions);
        await b.TrainAsync(Chunk(second));

        using var expected = new CenteringTransform(Dimensions);
        expected.Train(Count, second);

        var sample = second.AsSpan(0, Dimensions).ToArray();
        FloatAssert.Equal(expected.Apply(1, sample), b.Apply(1, sample));
    }

    [Fact]
    public async Task TrainAsync_AlreadyTrainedTransform_IsANoOp()
    {
        using var transform = new NormalizationTransform(Dimensions);
        Assert.True(transform.IsTrained);

        await transform.TrainAsync(Count, Vectors.Random(seed: 235, count: Count, dimensions: Dimensions));

        Assert.True(transform.IsTrained);
    }

    [Fact]
    public async Task TrainAsync_ThenApply_NoLongerThrowsUntrained()
    {
        using var transform = new CenteringTransform(Dimensions);
        var sample = new float[Dimensions];

        Assert.Throws<FaissUntrainedException>(() => transform.Apply(1, sample));

        await transform.TrainAsync(Count, Vectors.Random(seed: 236, count: Count, dimensions: Dimensions));

        Assert.Equal(Dimensions, transform.Apply(1, sample).Length);
    }

    private static List<ReadOnlyMemory<float>> Chunk(float[] vectors)
    {
        var chunks = new List<ReadOnlyMemory<float>>();
        for (var i = 0; i < Count; i++)
        {
            chunks.Add(new ReadOnlyMemory<float>(vectors, i * Dimensions, Dimensions));
        }

        return chunks;
    }
}
