using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.Transform;
using Faiss.Cpu.Transforms;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Indexes.Transform;

public class IndexPreTransformTests
{
    private const int Dimensions = 8;
    private const int Count = 32;

    [Fact]
    public void Constructor_WithoutTransform_DelegatesToTheSubIndex()
    {
        using var inner = new IndexFlatL2(Dimensions);
        using var index = new IndexPreTransform<IndexFlatL2>(inner);

        Assert.Equal(Dimensions, index.Dimensions);
        Assert.Empty(index.TransformChain);

        var vectors = Vectors.Random(seed: 250, count: Count, dimensions: Dimensions);
        index.Add(Count, vectors);

        Assert.Equal(Count, index.TotalCount);

        using var reference = new IndexFlatL2(Dimensions);
        reference.Add(Count, vectors);

        AssertSameNeighbours(reference, index, vectors.AsSpan(0, Dimensions), k: 4);
    }

    [Fact]
    public void Constructor_WithTransform_RegistersItInTheChain()
    {
        using var inner = new IndexFlatL2(Dimensions);
        using var index = new IndexPreTransform<IndexFlatL2>(inner, new NormalizationTransform(Dimensions));

        Assert.Single(index.TransformChain);
        Assert.IsType<NormalizationTransform>(index.TransformChain[0]);
    }

    [Fact]
    public void Search_AppliesTheTransformToStoredVectorsAndQueries()
    {
        using var inner = new IndexFlatL2(Dimensions);
        using var index = new IndexPreTransform<IndexFlatL2>(inner, new NormalizationTransform(Dimensions));

        var vectors = Vectors.Random(seed: 251, count: Count, dimensions: Dimensions);
        index.Add(Count, vectors);

        using var normalizer = new NormalizationTransform(Dimensions);
        var normalized = normalizer.Apply(Count, vectors);

        using var reference = new IndexFlatL2(Dimensions);
        reference.Add(Count, normalized);

        var query = vectors.AsSpan(0, Dimensions);
        var normalizedQuery = normalizer.Apply(1, query);

        var expectedDistances = new float[4];
        var expectedLabels = new long[4];
        reference.Search(1, normalizedQuery, 4, expectedDistances, expectedLabels);

        var actualDistances = new float[4];
        var actualLabels = new long[4];
        index.Search(1, query, 4, actualDistances, actualLabels);

        Assert.Equal(expectedLabels, actualLabels);
        FloatAssert.Equal(expectedDistances, actualDistances, relTol: 1e-4f, absTol: 1e-5f);
    }

    [Fact]
    public void PrependTransform_PutsTheNewTransformFirst()
    {
        using var inner = new IndexFlatL2(Dimensions);
        using var index = new IndexPreTransform<IndexFlatL2>(inner, new NormalizationTransform(Dimensions));

        index.PrependTransform(new CenteringTransform(Dimensions));

        Assert.Equal(2, index.TransformChain.Count);
        Assert.IsType<CenteringTransform>(index.TransformChain[0]);
        Assert.IsType<NormalizationTransform>(index.TransformChain[1]);
    }

    [Fact]
    public async Task Train_TrainsTheChainBeforeTheSubIndex()
    {
        using var inner = new IndexFlatL2(Dimensions);
        using var index = new IndexPreTransform<IndexFlatL2>(inner, new CenteringTransform(Dimensions));

        Assert.False(index.IsTrained);

        await index.TrainAsync(Count, Vectors.Random(seed: 252, count: Count, dimensions: Dimensions));

        Assert.True(index.IsTrained);
        Assert.True(index.TransformChain[0].IsTrained);
    }

    [Fact]
    public async Task Reconstruct_AppliesTheReverseTransform()
    {
        using var inner = new IndexFlatL2(Dimensions);
        using var index = new IndexPreTransform<IndexFlatL2>(inner, new CenteringTransform(Dimensions));

        var vectors = Vectors.Random(seed: 253, count: Count, dimensions: Dimensions);
        await index.TrainAsync(Count, vectors);
        index.Add(Count, vectors);

        FloatAssert.Equal(vectors.AsSpan(0, Dimensions), index.Reconstruct(0), relTol: 1e-4f, absTol: 1e-5f);
    }

    [Fact]
    public void SubIndexRemainsAccessibleThroughTheWrapper()
    {
        using var inner = new IndexFlatL2(Dimensions);
        using var index = new IndexPreTransform<IndexFlatL2>(inner, new NormalizationTransform(Dimensions));

        index.Add(Count, Vectors.Random(seed: 254, count: Count, dimensions: Dimensions));

        Assert.Equal(Count, index.Index.TotalCount);
    }

    private static void AssertSameNeighbours(IndexFlatL2 expected, IndexPreTransform<IndexFlatL2> actual, ReadOnlySpan<float> query, int k)
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
