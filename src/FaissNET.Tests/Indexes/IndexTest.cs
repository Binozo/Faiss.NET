using Faiss.Cpu.Extensions;
using Faiss.Cpu.Indexes;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Search.Range;
using Faiss.Cpu.Selectors;
using Faiss.Interfaces;
using Xunit;

namespace Faiss.Tests.Indexes;

[Collection("FaissCleanup")]
public abstract class IndexTest<T> : IFloatAccuracyTest where T : Index<T>
{
    protected void AssertFloatsEqual(float expected, float actual, float relTol = 1e-5f, float absTol = 1e-6f) => ((IFloatAccuracyTest)this).AssertFloatsEqual(expected, actual, relTol, absTol);

    protected abstract int Dimensions { get; }
    protected Func<T> IndexCreateCallback;

    protected IndexTest(Func<T> indexCreateCallback)
    {
        IndexCreateCallback = indexCreateCallback;
    }

    [Fact]
    public void AddAndSearch_WithParams()
    {
        using var index = IndexCreateCallback();
        index.Add([2.0f, 3.0f]);
        index.Add([2.0f, 3.1f]);
        
        long[] labels = new long[1];
        float[] distances = new float[1];

        index.SearchWithParams(1, [2.0f, 3.0f], 1, new SearchParameters(new IDSelectorNot(new IDSelectorBatch([0]))), distances, labels);
        Assert.Equal(1, labels[0]);
        AssertFloatsEqual(13.2999992f, distances[0]);
    }
    
    [Fact]
    public void AddAndSearch_Range()
    {
        using var index = IndexCreateCallback();

        index.Add([2.0f, 3.0f]);

        using var rangeSearchResult = new RangeSearchResult(1);
        index.RangeSearch(1, [2.0f, 3.1f], 1f, rangeSearchResult);
        
        Assert.Equal(1, rangeSearchResult.Nq);
        var queryResult = rangeSearchResult.GetQueryResult(0);
        
        Assert.Equal(1, queryResult.Labels.Length);
        Assert.Equal(1, queryResult.Distances.Length);
        
        Assert.Equal(0, queryResult.Labels[0]);
        AssertFloatsEqual(13.2999992f, queryResult.Distances[0]);
    }
    
    [Fact]
    public void AddAndAssign()
    {
        using var index = IndexCreateCallback();

        index.Add([2.0f, 3.0f]);

        using var rangeSearchResult = new RangeSearchResult(1);
        index.RangeSearch(1, [2.0f, 3.1f], 1f, rangeSearchResult);
        
        Assert.Equal(1, rangeSearchResult.Nq);
        var queryResult = rangeSearchResult.GetQueryResult(0);
        
        Assert.Equal(1, queryResult.Labels.Length);
        Assert.Equal(1, queryResult.Distances.Length);
        
        Assert.Equal(0, queryResult.Labels[0]);
        AssertFloatsEqual(13.2999992f, queryResult.Distances[0]);
    }
    
    [Fact]
    public void Reset()
    {
        using var index = IndexCreateCallback();

        index.Add([2.0f, 3.0f]);
        Assert.Equal(1, index.TotalCount);
        
        index.Reset();
        Assert.Equal(0, index.TotalCount);
    }
    
    [Fact]
    public void AddAndRemove()
    {
        using var index = IndexCreateCallback();

        index.Add([2.0f, 3.0f]);
        index.Add([2.0f, 3.1f]);
        index.Add([2.0f, 3.2f]);
        index.Add([2.0f, 3.3f]);
        
        Assert.Equal(4, index.TotalCount);

        index.RemoveIds(new IDSelectorBatch([1, 2]));
        Assert.Equal(2, index.TotalCount);
    }
}