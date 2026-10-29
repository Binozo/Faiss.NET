using Faiss.Cpu.Indexes.Approximate;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Indexes.Refine;
using Faiss.Cpu.Indexes.Replicas;
using Faiss.Cpu.Indexes.Sharding;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Serializer;
using Faiss.Models;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Contract;

public class SerializableTypeCoverageTests
{
    private const int Dimensions = 8;

    // NNDescent refuses to build a graph from fewer than 100 vectors (NNDescent.cpp:429),
    // so NSG needs a larger dataset than the other families.
    private const int Count = 128;

    [Fact]
    public void NsgFlat_StreamRoundtrip_PreservesGraphResults()
    {
        using var original = new IndexNSGFlat(Dimensions, r: 16);
        var vectors = Vectors.Random(seed: 90, count: Count, dimensions: Dimensions);
        original.Add(Count, vectors);

        using var stream = new MemoryStream();
        IndexSerializer.Write(original, stream);
        stream.Position = 0;

        using var restored = IndexDeserializer.Read<IndexNSGFlat>(stream);

        Assert.Equal(original.Dimensions, restored.Dimensions);
        Assert.Equal(Count, restored.TotalCount);

        // The NSG graph itself must survive, not just the nested flat storage.
        AssertSameNeighbours(original, restored, vectors.AsSpan(0, Dimensions), k: 5);
    }

    [Fact]
    public async Task IvfScalarQuantizer_StreamRoundtrip_PreservesTrainingAndResults()
    {
        using var quantizer = new IndexFlatL2(Dimensions);
        using var original = new IndexIVFScalarQuantizer<IndexFlatL2>(quantizer, Dimensions, nlist: 4, QuantizerType.QT_8bit);
        await original.TrainAsync(256, Vectors.Random(seed: 91, count: 256, dimensions: Dimensions));

        var vectors = Vectors.Random(seed: 92, count: Count, dimensions: Dimensions);
        original.Add(Count, vectors);
        original.Nprobe = original.Nlist;

        using var stream = new MemoryStream();
        IndexSerializer.Write(original, stream);
        stream.Position = 0;

        using var restored = IndexDeserializer.Read<IndexIVFScalarQuantizer<IndexFlatL2>>(stream);

        // Quantizer trained state and the SQ ranges must both come back.
        Assert.True(restored.IsTrained);
        Assert.Equal(original.Nlist, restored.Nlist);
        Assert.Equal(Count, restored.TotalCount);

        restored.Nprobe = restored.Nlist;
        AssertSameNeighbours(original, restored, vectors.AsSpan(0, Dimensions), k: 5);
    }

    [Fact(Skip = NativeSupport.RefineFlatBaseIndex)]
    public void RefineFlat_StreamRoundtrip_RematerializesAsRefineFlat()
    {
        using var baseIndex = new IndexFlatL2(Dimensions);
        using var original = new IndexRefineFlat<IndexFlatL2>(baseIndex);

        var vectors = Vectors.Random(seed: 93, count: Count, dimensions: Dimensions);
        original.Add(Count, vectors);

        using var stream = new MemoryStream();
        IndexSerializer.Write(original, stream);
        stream.Position = 0;

        // faiss writes fourcc IxRF and, when the refine index reads back as an IndexFlat,
        // re-materializes the object as IndexRefineFlat rather than a plain IndexRefine.
        using var restored = IndexDeserializer.Read<IndexRefineFlat<IndexFlatL2>>(stream);

        Assert.Equal(original.Dimensions, restored.Dimensions);
        Assert.Equal(Count, restored.TotalCount);

        AssertSameNeighbours(original, restored, vectors.AsSpan(0, Dimensions), k: 5);
    }

    [Fact]
    public void Shards_AreNotSerializable()
    {
        using var shards = new IndexShards(Dimensions);
        shards.AddIndex(new IndexFlatL2(Dimensions));
        shards.Add(4, Vectors.Random(seed: 94, count: 4, dimensions: Dimensions));

        // faiss's writer has no IndexShards branch
        Assert.False(shards is ISerializableFloatIndex);
    }

    [Fact]
    public void Replicas_AreNotMarkedSerializable()
    {
        // faiss's writer has no IndexReplicas branch
        Assert.False(typeof(ISerializableFloatIndex).IsAssignableFrom(typeof(IndexReplicas<IndexFlatL2>)));
    }

    private static void AssertSameNeighbours(Faiss.Cpu.Indexes.FloatIndex expected, Faiss.Cpu.Indexes.FloatIndex actual, ReadOnlySpan<float> query, int k)
    {
        var expectedDistances = new float[k];
        var expectedLabels = new long[k];
        expected.Search(1, query, k, expectedDistances, expectedLabels);

        var actualDistances = new float[k];
        var actualLabels = new long[k];
        actual.Search(1, query, k, actualDistances, actualLabels);

        Assert.Equal(expectedLabels, actualLabels);
        FloatAssert.Equal(expectedDistances, actualDistances);
    }
}
