using Faiss.Cpu.Indexes.Approximate;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Indexes.Approximate;

public class IndexIVFPQTests
{
    private const int Dimensions = 8;
    private const int Count = 64;

    [Fact]
    public async Task Reconstruct_BatchOverload_WorksWithoutADirectMap()
    {
        using var index = await CreateAsync();

        var batch = index.Reconstruct(0, 4);

        index.DirectMap = true;
        for (var i = 0; i < 4; i++)
        {
            FloatAssert.Equal(index.Reconstruct(i), batch.AsSpan(i * Dimensions, Dimensions));
        }
    }

    private static async Task<IndexIVFPQ> CreateAsync()
    {
        var index = new IndexIVFPQ(Dimensions, n: 4, m: 4);
        await index.TrainAsync(1024, Vectors.Random(seed: 130, count: 1024, dimensions: Dimensions));

        index.Add(Count, Vectors.Random(seed: 131, count: Count, dimensions: Dimensions));
        index.Nprobe = index.Nlist;

        return index;
    }
}
