using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Contract;

public class LifetimeContractTests
{
    [Theory]
    [MemberData(nameof(IndexCatalog.Float), MemberType = typeof(IndexCatalog))]
    public void Dispose_Twice_DoesNotThrow(IndexCase c)
    {
        var index = c.CreateIndex();
        index.Dispose();
        index.Dispose();
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.Binary), MemberType = typeof(IndexCatalog))]
    public void Dispose_Twice_DoesNotThrow_Binary(BinaryIndexCase c)
    {
        var index = c.CreateIndex();
        index.Dispose();
        index.Dispose();
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.Float), MemberType = typeof(IndexCatalog))]
    public void UseAfterDispose_Throws(IndexCase c)
    {
        var index = c.CreateIndex();
        index.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = index.Dimensions);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.Binary), MemberType = typeof(IndexCatalog))]
    public void UseAfterDispose_Throws_Binary(BinaryIndexCase c)
    {
        var index = c.CreateIndex();
        index.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = index.Dimensions);
    }
}
