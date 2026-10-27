using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Indexes.Mapped;
using Faiss.Models;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Indexes.IVF;

public class IndexIVFScalarQuantizerTests
{
    private const int Dimensions = 8;

    [Theory]
    [InlineData(QuantizerType.QT_8bit)]
    [InlineData(QuantizerType.QT_4bit)]
    [InlineData(QuantizerType.QT_fp16)]
    public void UsesTheRequestedQuantizerType(QuantizerType type)
    {
        using var quantizer = new IndexFlatL2(Dimensions);
        using var index = new IndexIVFScalarQuantizer<IndexFlatL2>(quantizer, Dimensions, nlist: 4, type);

        Assert.Equal(Dimensions, index.Dimensions);
        Assert.Equal(4, index.Nlist);
        Assert.Equal(type, index.ScalarQuantizer.QuantizerType);
    }

    [Fact]
    public async Task ExhaustiveSearch_FindsEveryStoredVector()
    {
        const int count = 32;

        using var quantizer = new IndexFlatL2(Dimensions);
        using var index = new IndexIVFScalarQuantizer<IndexFlatL2>(quantizer, Dimensions, nlist: 4, QuantizerType.QT_8bit);
        await index.TrainAsync(256, Vectors.Random(seed: 402, count: 256, dimensions: Dimensions));

        var vectors = Vectors.Random(seed: 403, count: count, dimensions: Dimensions);
        index.Add(count, vectors);
        index.Nprobe = index.Nlist;

        var distances = new float[1];
        var labels = new long[1];

        for (var i = 0; i < count; i++)
        {
            index.Search(1, vectors.AsSpan(i * Dimensions, Dimensions), k: 1, distances, labels);
            Assert.Equal(i, labels[0]);
        }
    }

    [Fact]
    public async Task NotOwned_CallerDisposesQuantizerEarly_IndexKeepsWorking()
    {
        var quantizer = new IndexFlatL2(Dimensions);
        using var index = new IndexIVFScalarQuantizer<IndexFlatL2>(quantizer, Dimensions, nlist: 4, QuantizerType.QT_8bit, ownQuantizer: false);
        await index.TrainAsync(256, Vectors.Random(seed: 404, count: 256, dimensions: Dimensions));
        index.Nprobe = index.Nlist;

        var vectors = Vectors.Random(seed: 405, count: 8, dimensions: Dimensions);
        index.Add(8, vectors);

        quantizer.Dispose();
        GC.Collect();
        GC.WaitForPendingFinalizers();

        var distances = new float[1];
        var labels = new long[1];
        index.Search(1, vectors.AsSpan(0, Dimensions), k: 1, distances, labels);

        Assert.Equal(0, labels[0]);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ViewOfAnotherIndex_AsQuantizer_IsRefused(bool ownQuantizer)
    {
        using var map = new IndexIDMap<IndexFlatL2>(new IndexFlatL2(Dimensions), takeOwnership: true);

        Assert.Throws<ArgumentException>(() =>
            new IndexIVFScalarQuantizer<IndexFlatL2>(map.SubIndex, Dimensions, nlist: 4, QuantizerType.QT_8bit, ownQuantizer: ownQuantizer));
        Assert.Equal(Dimensions, map.SubIndex.Dimensions);
    }

    [Fact]
    public void ScalarQuantizer_OutlivesAnUndisposedIndex()
    {
        var sq = Abandon();
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        GC.WaitForPendingFinalizers();

        Assert.Equal(QuantizerType.QT_8bit, sq.QuantizerType);

        static ScalarQuantizer Abandon() => new IndexIVFScalarQuantizer<IndexFlatL2>(new IndexFlatL2(Dimensions), Dimensions, nlist: 2, QuantizerType.QT_8bit, ownQuantizer: true).ScalarQuantizer;
    }

    [Fact]
    public void ScalarQuantizer_AfterIndexDispose_Throws()
    {
        var index = new IndexIVFScalarQuantizer<IndexFlatL2>(new IndexFlatL2(Dimensions), Dimensions, nlist: 2, QuantizerType.QT_8bit, ownQuantizer: true);
        var sq = index.ScalarQuantizer;
        index.Dispose();

        Assert.Throws<ObjectDisposedException>(() => sq.QuantizerType);
    }
}
