using Faiss.Cpu.Indexes.Flat;
using Faiss.Models;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Indexes.Flat;

public class IndexScalarQuantizerTests
{
    private const int Dimensions = 8;
    private const int Count = 8;

    [Fact]
    public async Task TrainAddSearch_ApproximatesInput()
    {
        using var index = new IndexScalarQuantizer(Dimensions, QuantizerType.QT_8bit);
        Assert.False(index.IsTrained);

        var train = Vectors.Random(seed: 51, count: 128, dimensions: Dimensions);
        await index.TrainAsync(128, train);
        Assert.True(index.IsTrained);

        var vectors = Vectors.Random(seed: 52, count: Count, dimensions: Dimensions);
        index.Add(Count, vectors);
        Assert.Equal(Count, index.TotalCount);

        var distances = new float[1];
        var labels = new long[1];
        index.Search(1, vectors.AsSpan(0, Dimensions), 1, distances, labels);

        Assert.Equal(0, labels[0]);

        Assert.InRange(distances[0], 0f, 0.05f);
    }

    [Theory]
    [InlineData(QuantizerType.QT_8bit, 8)]
    [InlineData(QuantizerType.QT_4bit, 4)]
    [InlineData(QuantizerType.QT_6bit, 6)]
    [InlineData(QuantizerType.QT_fp16, 16)]
    public void ScalarQuantizer_Accessors_MatchConstruction(QuantizerType type, int bits)
    {
        using var index = new IndexScalarQuantizer(Dimensions, type);
        var sq = index.ScalarQuantizer;

        Assert.Equal(type, sq.QuantizerType);
        Assert.Equal(bits, sq.Bits);
        Assert.Equal(Dimensions, sq.Dimension);

        Assert.Equal((bits * Dimensions + 7) / 8, sq.CodeSize);

        Assert.Equal(RangeStat.RS_minmax, sq.RangeStat);
        Assert.Equal(0f, sq.RangeStatArg);
    }

    [Fact]
    public async Task ScalarQuantizer_TrainedValues_AppearOnlyAfterTraining()
    {
        using var index = new IndexScalarQuantizer(Dimensions, QuantizerType.QT_8bit);
        var sq = index.ScalarQuantizer;

        Assert.Equal(0u, sq.TrainedSize);

        await index.TrainAsync(256, Vectors.Random(seed: 401, count: 256, dimensions: Dimensions));

        var size = (int)sq.TrainedSize;
        Assert.True(size > 0, "Training must populate the per-dimension ranges.");

        var trained = new float[size];
        sq.CopyTrainedValues(trained);
        Assert.All(trained, v => Assert.True(float.IsFinite(v)));
        Assert.Contains(trained, v => v != 0f);

        Assert.Throws<ArgumentOutOfRangeException>(() => sq.CopyTrainedValues(new float[size - 1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => sq.CopyTrainedValues(new float[size + 1]));
    }

    [Fact]
    public void ScalarQuantizer_AfterIndexDispose_Throws()
    {
        var index = new IndexScalarQuantizer(Dimensions, QuantizerType.QT_8bit);
        var sq = index.ScalarQuantizer;
        index.Dispose();

        Assert.Throws<ObjectDisposedException>(() => sq.Bits);
    }

    [Fact]
    public void ScalarQuantizer_OutlivesAnUndisposedIndex()
    {
        var sq = Abandon();
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        GC.WaitForPendingFinalizers();

        Assert.Equal(Dimensions, sq.Dimension);
        Assert.Equal(QuantizerType.QT_8bit, sq.QuantizerType);

        static ScalarQuantizer Abandon() => new IndexScalarQuantizer(Dimensions, QuantizerType.QT_8bit).ScalarQuantizer;
    }
}
