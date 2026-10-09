using Faiss.Cpu.AutoTune;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Exceptions;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.AutoTune;

public class ParameterSpaceTests
{
    private const int Dimensions = 4;
    private const int Nlist = 8;

    [Fact]
    public void FreshSpace_HasASingleEmptyCombination()
    {
        using var space = new ParameterSpace();

        Assert.Equal(1, space.CombinationCount);
        Assert.Equal(string.Empty, space.GetCombinationName(0));
    }

    [Fact]
    public void AddRange_ProducesANamedButEmptyRange()
    {
        using var space = new ParameterSpace();

        var range = space.AddRange("nprobe");

        Assert.Equal("nprobe", range.Name);

        Assert.Empty(range.Values.ToArray());
        Assert.Equal(0, space.CombinationCount);
    }

    [Fact]
    public void GetCombinationName_WithAnEmptyRange_ThrowsInsteadOfDividingByZero()
    {
        using var space = new ParameterSpace();
        space.AddRange("nprobe");

        Assert.Throws<InvalidOperationException>(() => space.GetCombinationName(0));
    }

    [Fact]
    public async Task SetParameters_ByCombinationNo_WithAnEmptyRange_Throws()
    {
        using var space = new ParameterSpace();
        space.AddRange("nprobe");

        using var index = await CreateIndexAsync();

        Assert.Throws<InvalidOperationException>(() => space.SetParameters(index, 0));
    }

    [Fact]
    public void GetCombinationName_OutOfRange_Throws()
    {
        using var space = new ParameterSpace();

        Assert.Throws<ArgumentOutOfRangeException>(() => space.GetCombinationName(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => space.GetCombinationName(space.CombinationCount));
    }

    [Fact]
    public async Task SetParameter_AppliesTheValueToTheIndex()
    {
        using var space = new ParameterSpace();
        using var index = await CreateIndexAsync();

        Assert.Equal(1, index.Nprobe);

        space.SetParameter(index, "nprobe", 5);

        Assert.Equal(5, index.Nprobe);
    }

    [Fact]
    public async Task SetParameter_UnknownName_Throws()
    {
        using var space = new ParameterSpace();
        using var index = await CreateIndexAsync();

        Assert.Throws<FaissNativeException>(() => space.SetParameter(index, "not_a_parameter", 1));
        Assert.Equal(1, index.Nprobe);
    }

    [Fact]
    public async Task SetParameters_ByDescription_AppliesEveryAssignment()
    {
        using var space = new ParameterSpace();
        using var index = await CreateIndexAsync();

        space.SetParameters(index, "nprobe=3");

        Assert.Equal(3, index.Nprobe);
    }

    [Fact]
    public async Task SetParameters_UnknownKeyInDescription_Throws()
    {
        using var space = new ParameterSpace();
        using var index = await CreateIndexAsync();

        Assert.Throws<FaissNativeException>(() => space.SetParameters(index, "bogus=3"));
    }

    [Fact]
    public async Task SetParameter_ChangesSearchBehaviour()
    {
        using var space = new ParameterSpace();
        using var index = await CreateIndexAsync();

        var vectors = Vectors.Random(seed: 161, count: 64, dimensions: Dimensions);
        index.Add(64, vectors);

        space.SetParameter(index, "nprobe", Nlist);

        var distances = new float[1];
        var labels = new long[1];
        index.Search(1, vectors.AsSpan(0, Dimensions), 1, distances, labels);

        Assert.Equal(0, labels[0]);
        FloatAssert.Equal(0f, distances[0]);
    }

    [Fact]
    public void Display_DoesNotThrow()
    {
        using var space = new ParameterSpace();

        space.Display();
    }

    [Fact]
    public void UseAfterDispose_Throws()
    {
        var space = new ParameterSpace();
        space.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = space.CombinationCount);
    }

    private static async Task<IndexIVFFlat<IndexFlatL2>> CreateIndexAsync()
    {
        var quantizer = new IndexFlatL2(Dimensions);
        var index = new IndexIVFFlat<IndexFlatL2>(quantizer, Dimensions, Nlist);

        await index.TrainAsync(256, Vectors.Random(seed: 160, count: 256, dimensions: Dimensions));

        return index;
    }
}
