using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.Mapped;
using Faiss.Cpu.Indexes.Transform;
using Faiss.Cpu.Transforms;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Indexes.Transform;
public class IndexPreTransformOwnershipTests
{
    private const int Dimensions = 4;
    private const int Count = 4;

    private static readonly float[] Data = Vectors.Random(seed: 701, count: Count, dimensions: Dimensions);

    private static void AssertFindsFirstVector(IndexPreTransform<IndexFlatL2> index)
    {
        var distances = new float[1];
        var labels = new long[1];
        index.Search(1, Data.AsSpan(0, Dimensions), k: 1, distances, labels);

        Assert.Equal(0, labels[0]);
        FloatAssert.Equal(0f, distances[0], relTol: 1e-4f, absTol: 1e-5f);
    }

    [Fact]
    public void Index_Owned_IsAViewOfTheWrappedIndex()
    {
        var original = new IndexFlatL2(Dimensions);
        using var index = new IndexPreTransform<IndexFlatL2>(original, ownSubIndex: true);
        index.Add(Count, Data);

        Assert.NotSame(original, index.Index);
        Assert.Equal(Dimensions, index.Index.Dimensions);
        Assert.Equal(Count, index.Index.TotalCount);
    }

    [Fact]
    public void Index_Owned_ThrowsOnceTheParentIsDisposed()
    {
        var index = new IndexPreTransform<IndexFlatL2>(new IndexFlatL2(Dimensions), ownSubIndex: true);
        index.Add(Count, Data);
        var sub = index.Index;

        index.Dispose();

        Assert.Throws<ObjectDisposedException>(() => sub.Dimensions);
    }

    [Fact]
    public void Index_OfAClone_IsAUsableView()
    {
        using var index = new IndexPreTransform<IndexFlatL2>(new IndexFlatL2(Dimensions), ownSubIndex: true);
        index.Add(Count, Data);

        var clone = index.Clone();
        var sub = clone.Index;

        Assert.Equal(Count, sub.TotalCount);
        AssertFindsFirstVector(clone);

        clone.Dispose();
        Assert.Throws<ObjectDisposedException>(() => sub.TotalCount);
    }

    [Fact]
    public void Index_NotOwned_IsTheCallersObject()
    {
        using var original = new IndexFlatL2(Dimensions);
        using var index = new IndexPreTransform<IndexFlatL2>(original, ownSubIndex: false);

        Assert.Same(original, index.Index);
    }

    [Fact]
    public void NotOwned_CallerDisposesPartsEarly_IndexKeepsWorking()
    {
        var sub = new IndexFlatL2(Dimensions);
        var transform = new NormalizationTransform(Dimensions);
        var renormalized = transform.Apply(Count, Data);

        using var index = new IndexPreTransform<IndexFlatL2>(sub, transform, ownSubIndex: false);
        index.Add(Count, Data);

        sub.Dispose();
        transform.Dispose();
        GC.Collect();
        GC.WaitForPendingFinalizers();

        Assert.Equal(Count, index.TotalCount);
        Assert.Equal(Dimensions, index.Dimensions);
        AssertFindsFirstVector(index);
        FloatAssert.Equal(renormalized.AsSpan(0, Dimensions), index.Reconstruct(0).AsSpan(), relTol: 1e-4f, absTol: 1e-5f);
    }

    [Fact]
    public void NotOwned_PrependedTransformDisposedEarly_IndexKeepsWorking()
    {
        using var sub = new IndexFlatL2(Dimensions);
        using var index = new IndexPreTransform<IndexFlatL2>(sub, ownSubIndex: false);

        var transform = new NormalizationTransform(Dimensions);
        index.PrependTransform(transform);
        index.Add(Count, Data);

        transform.Dispose();
        GC.Collect();
        GC.WaitForPendingFinalizers();

        AssertFindsFirstVector(index);
    }

    [Fact]
    public void PrependTransform_ThroughAViewOfANonOwningIndex_IsRefused()
    {
        using var sub = new IndexFlatL2(Dimensions);
        var preTransform = new IndexPreTransform<IndexFlatL2>(sub, ownSubIndex: false);
        using var map = new IndexIDMap<IndexPreTransform<IndexFlatL2>>(preTransform, takeOwnership: true);

        var view = map.SubIndex;
        Assert.False(view.OwnSubIndex);

        using var transform = new NormalizationTransform(Dimensions);
        Assert.Throws<InvalidOperationException>(() => view.PrependTransform(transform));
        Assert.Empty(view.TransformChain);
    }

    [Fact]
    public void TransformChain_Owned_HoldsAWorkingViewOfTheSameType()
    {
        var original = new NormalizationTransform(Dimensions);
        using var index = new IndexPreTransform<IndexFlatL2>(new IndexFlatL2(Dimensions), original, ownSubIndex: true);

        Assert.Throws<ObjectDisposedException>(() => original.DIn);

        var view = Assert.IsType<NormalizationTransform>(Assert.Single(index.TransformChain));
        Assert.NotSame(original, view);
        Assert.Equal(Dimensions, view.DIn);
        Assert.Equal(2f, view.Norm);

        var applied = view.Apply(1, Data.AsSpan(0, Dimensions));
        FloatAssert.Equal(1f, MathF.Sqrt(Oracles.InnerProduct(applied, applied)), relTol: 1e-4f, absTol: 1e-5f);
    }

    [Fact]
    public void TransformChain_Owned_ViewThrowsOnceTheIndexIsDisposed()
    {
        var index = new IndexPreTransform<IndexFlatL2>(new IndexFlatL2(Dimensions), new NormalizationTransform(Dimensions), ownSubIndex: true);
        var view = index.TransformChain[0];

        index.Dispose();

        Assert.Throws<ObjectDisposedException>(() => view.DIn);
    }

    [Fact]
    public void TransformChain_NotOwned_HoldsTheCallersObject()
    {
        using var sub = new IndexFlatL2(Dimensions);
        using var transform = new NormalizationTransform(Dimensions);
        using var index = new IndexPreTransform<IndexFlatL2>(sub, transform, ownSubIndex: false);

        Assert.Same(transform, Assert.Single(index.TransformChain));
    }

    [Theory]
    [MemberData(nameof(EveryTransformType))]
    public void PrependTransform_Owned_ChainViewKeepsTheConcreteType(string name, Func<VectorTransform> create)
    {
        _ = name;
        using var index = new IndexPreTransform<IndexFlatL2>(new IndexFlatL2(Dimensions), ownSubIndex: true);
        var original = create();
        var type = original.GetType();

        index.PrependTransform(original);

        var view = Assert.Single(index.TransformChain);
        Assert.IsType(type, view);
        Assert.NotSame(original, view);
        Assert.Equal(Dimensions, view.DOut);
    }

    public static TheoryData<string, Func<VectorTransform>> EveryTransformType { get; } = new()
    {
        { "Centering", () => new CenteringTransform(Dimensions) },
        { "ITQMatrix", () => new ITQMatrix(Dimensions) },
        { "ITQTransform", () => new ITQTransform(Dimensions, Dimensions) },
        { "Normalization", () => new NormalizationTransform(Dimensions) },
        { "OPQMatrix", () => new OPQMatrix(Dimensions, m: 2, d2: Dimensions) },
        { "PCAMatrix", () => new PCAMatrix(Dimensions, Dimensions) },
        { "RandomRotation", () => new RandomRotationMatrix(Dimensions, Dimensions) },
        { "RemapDimensions", () => new RemapDimensionsTransform(Dimensions, Dimensions) },
    };

    [Fact]
    public void ChainViewFromAnotherIndex_IsRejected()
    {
        using var first = new IndexPreTransform<IndexFlatL2>(new IndexFlatL2(Dimensions), new NormalizationTransform(Dimensions), ownSubIndex: true);
        var borrowed = first.TransformChain[0];

        Assert.Throws<ArgumentException>(() => new IndexPreTransform<IndexFlatL2>(new IndexFlatL2(Dimensions), borrowed, ownSubIndex: true));

        using var second = new IndexPreTransform<IndexFlatL2>(new IndexFlatL2(Dimensions), ownSubIndex: true);
        Assert.Throws<ArgumentException>(() => second.PrependTransform(borrowed));
        Assert.Empty(second.TransformChain);

        Assert.Equal(Dimensions, borrowed.DIn);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ViewOfAnotherIndex_AsSubIndex_IsRefused(bool ownSubIndex)
    {
        using var map = new IndexIDMap<IndexFlatL2>(new IndexFlatL2(Dimensions), takeOwnership: true);

        Assert.Throws<ArgumentException>(() => new IndexPreTransform<IndexFlatL2>(map.SubIndex, ownSubIndex));
        Assert.Throws<ArgumentException>(() => new IndexPreTransform<IndexFlatL2>(map.SubIndex, new NormalizationTransform(Dimensions), ownSubIndex));
        Assert.Equal(Dimensions, map.SubIndex.Dimensions);
    }

    [Fact]
    public void Owned_WithChainViews_NeverDisposed_IsSafeUnderGcPressure()
    {
        for (var i = 0; i < 50; i++)
        {
            Abandon();
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        GC.WaitForPendingFinalizers();

        static void Abandon()
        {
            var index = new IndexPreTransform<IndexFlatL2>(new IndexFlatL2(Dimensions), new NormalizationTransform(Dimensions), ownSubIndex: true);
            index.PrependTransform(new CenteringTransform(Dimensions));
            _ = index.TransformChain[0].DIn + index.TransformChain[1].DIn;
        }
    }
}
