using Faiss.Cpu.Extensions;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Indexes.Flat;

public class IndexFlatL2Tests
{
    private const int Dimensions = 2;

    [Fact]
    public void SearchOverloads_AgreeOnNeighbors()
    {
        using var index = new IndexFlatL2(Dimensions);
        float[] xb =
        [
            0f, 0f,
            1f, 0f,
            0f, 2f,
        ];
        index.Add(3, xb);

        float[] query = [0f, 0f];
        const int k = 2;

        var lowDistances = new float[k];
        var lowLabels = new long[k];
        index.Search(1, query, k, lowDistances, lowLabels);

        using var owned = index.Search(query, k);

        Span<float> viewDistances = stackalloc float[k];
        Span<long> viewLabels = stackalloc long[k];
        var view = index.Search(query, k, viewDistances, viewLabels);

        for (var rank = 0; rank < k; rank++)
        {
            Assert.Equal(lowLabels[rank], owned[rank].Label);
            Assert.Equal(lowLabels[rank], view[rank].Label);
            FloatAssert.Equal(lowDistances[rank], owned[rank].Distance);
            FloatAssert.Equal(lowDistances[rank], view[rank].Distance);
        }
    }
}
