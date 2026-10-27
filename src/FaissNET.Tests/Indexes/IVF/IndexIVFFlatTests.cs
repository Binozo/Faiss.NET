using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Indexes.Mapped;
using Faiss.Exceptions;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Indexes.IVF;

public class IndexIVFFlatTests
{
    [Fact]
    public void Constructor_SetsNlistAndDefaultNprobe()
    {
        using var quantizer = new IndexFlatL2(4);
        using var index = new IndexIVFFlat<IndexFlatL2>(quantizer, 4, nlist: 2);

        Assert.Equal(2, index.Nlist);
        Assert.Equal(1, index.Nprobe);

        index.Nprobe = 2;
        Assert.Equal(2, index.Nprobe);
    }

    [Fact]
    public void Constructor_QuantizerDimensionMismatch_Throws()
    {
        using var quantizer = new IndexFlatL2(4);
        Assert.Throws<ArgumentOutOfRangeException>(() => new IndexIVFFlat<IndexFlatL2>(quantizer, dimensions: 8, nlist: 2));
    }

    [Fact]
    public void Dispose_WithoutOwningQuantizer_LeavesQuantizerUsable()
    {
        using var quantizer = new IndexFlatL2(4);
        var index = new IndexIVFFlat<IndexFlatL2>(quantizer, 4, nlist: 2, ownQuantizer: false);
        index.Dispose();

        quantizer.Add(1, new float[4]);
        Assert.Equal(1, quantizer.TotalCount);
    }

    [Fact]
    public async Task NotOwned_CallerDisposesQuantizerEarly_IndexKeepsWorking()
    {
        var quantizer = new IndexFlatL2(4);
        using var index = new IndexIVFFlat<IndexFlatL2>(quantizer, 4, nlist: 2, ownQuantizer: false);
        await index.TrainAsync(80, Vectors.Random(seed: 801, count: 80, dimensions: 4));
        index.Nprobe = index.Nlist;

        var vectors = Vectors.Random(seed: 802, count: 8, dimensions: 4);
        index.Add(8, vectors);

        quantizer.Dispose();
        GC.Collect();
        GC.WaitForPendingFinalizers();

        var distances = new float[1];
        var labels = new long[1];
        index.Search(1, vectors.AsSpan(0, 4), k: 1, distances, labels);

        Assert.Equal(0, labels[0]);
        FloatAssert.Equal(0f, distances[0]);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ViewOfAnotherIndex_AsQuantizer_IsRefused(bool ownQuantizer)
    {
        var vectors = Vectors.Random(seed: 803, count: 4, dimensions: 4);
        using var map = new IndexIDMap<IndexFlatL2>(new IndexFlatL2(4), takeOwnership: true);
        map.Add(4, vectors, [10, 11, 12, 13]);

        Assert.Throws<ArgumentException>(() => new IndexIVFFlat<IndexFlatL2>(map.SubIndex, 4, nlist: 2, ownQuantizer: ownQuantizer));

        var distances = new float[1];
        var labels = new long[1];
        map.Search(1, vectors.AsSpan(0, 4), k: 1, distances, labels);

        Assert.Equal(10, labels[0]);
        Assert.Equal(4, map.SubIndex.TotalCount);
    }

    [Fact]
    public async Task AddCore_StoresEveryVectorUnderTheSuppliedId()
    {
        const int dimensions = 8;
        const int count = 4;

        using var quantizer = new IndexFlatL2(dimensions);
        using var index = new IndexIVFFlat<IndexFlatL2>(quantizer, dimensions, nlist: 2);
        await index.TrainAsync(64, Vectors.Random(seed: 301, count: 64, dimensions: dimensions));

        var vectors = Vectors.Random(seed: 302, count: count, dimensions: dimensions);

        long[] ids = [100, 200, 300, 400];

        var listNumbers = new long[count];
        quantizer.Assign(count, vectors, k: 1, listNumbers);

        index.AddCore(vectors, ids, listNumbers);

        Assert.Equal(count, index.TotalCount);

        index.Nprobe = index.Nlist;

        var distances = new float[1];
        var labels = new long[1];

        for (var i = 0; i < count; i++)
        {
            index.Search(1, vectors.AsSpan(i * dimensions, dimensions), k: 1, distances, labels);

            Assert.Equal(ids[i], labels[0]);
            FloatAssert.Equal(0f, distances[0]);
        }
    }

    [Fact]
    public async Task AddCore_InvalidArguments_Throw()
    {
        const int dimensions = 8;
        const int count = 4;

        using var quantizer = new IndexFlatL2(dimensions);
        using var index = new IndexIVFFlat<IndexFlatL2>(quantizer, dimensions, nlist: 2);
        await index.TrainAsync(64, Vectors.Random(seed: 303, count: 64, dimensions: dimensions));

        var vectors = Vectors.Random(seed: 304, count: count, dimensions: dimensions);
        long[] ids = [100, 200, 300, 400];
        var listNumbers = new long[count];
        quantizer.Assign(count, vectors, k: 1, listNumbers);

        // Vector span length is not divisible by dimensions
        Assert.ThrowsAny<ArgumentException>(() => index.AddCore(vectors.AsSpan(0, dimensions + 1), ids, listNumbers));

        // Missing one id entry
        Assert.ThrowsAny<ArgumentException>(() => index.AddCore(vectors, ids.AsSpan(0, count - 1), listNumbers));

        // Missing one list assignment
        Assert.ThrowsAny<ArgumentException>(() => index.AddCore(vectors, ids, listNumbers.AsSpan(0, count - 1)));

        Assert.Equal(0, index.TotalCount);
    }

    [Fact]
    public async Task UpdateVectors_ReplacesTheStoredVector()
    {
        using var index = await TrainedOnTwoClusters();
        index.Add(4, Clustered(seed: 1, perCluster: 2));
        index.DirectMap = true;

        var replacement = Vectors.Random(seed: 862, count: 1, dimensions: 4);
        index.UpdateVectors([1], replacement);

        FloatAssert.Equal(replacement, index.Reconstruct(1));
        Assert.Equal(4, index.TotalCount);
    }

    [Fact]
    public async Task UpdateVectors_WithoutDirectMap_Throws()
    {
        using var index = await TrainedOnTwoClusters();
        index.Add(4, Clustered(seed: 1, perCluster: 2));

        Assert.Throws<FaissNativeException>(() => index.UpdateVectors([1], new float[4]));
    }

    [Fact]
    public async Task UpdateVectors_VectorLengthMustMatchIdCount()
    {
        using var index = await TrainedOnTwoClusters();
        index.Add(4, Clustered(seed: 1, perCluster: 2));
        index.DirectMap = true;

        Assert.Equal("vectors", Assert.Throws<ArgumentException>(() => index.UpdateVectors([1, 2], new float[4])).ParamName);

        // Reject trailing partial vector
        Assert.Equal("vectors", Assert.Throws<ArgumentException>(() => index.UpdateVectors([1], new float[5])).ParamName);
    }

    [Fact]
    public async Task ImbalanceFactor_ReflectsHowListsAreFilled()
    {
        using var empty = await TrainedOnTwoClusters();
        Assert.True(double.IsNaN(empty.ImbalanceFactor), "An empty index has no distribution (0/0).");

        using var balanced = await TrainedOnTwoClusters();
        balanced.Add(8, Clustered(seed: 2, perCluster: 4));
        Assert.Equal(1.0, balanced.ImbalanceFactor, precision: 9);

        using var skewed = await TrainedOnTwoClusters();
        skewed.Add(4, Clustered(seed: 3, perCluster: 4).AsSpan(0, 4 * 4));
        Assert.Equal(2.0, skewed.ImbalanceFactor, precision: 9);
    }

    private static async Task<IndexIVFFlat<IndexFlatL2>> TrainedOnTwoClusters()
    {
        var index = new IndexIVFFlat<IndexFlatL2>(new IndexFlatL2(4), 4, nlist: 2, ownQuantizer: true);
        await index.TrainAsync(64, Clustered(seed: 861, perCluster: 32));
        return index;
    }

    private static float[] Clustered(int seed, int perCluster)
    {
        var noise = Vectors.Random(seed, 2 * perCluster, 4);
        for (var i = 0; i < noise.Length; i++)
        {
            noise[i] = (i < perCluster * 4 ? 10f : -10f) + (0.1f * noise[i]);
        }

        return noise;
    }
}
