using Faiss.Models;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Models;

/// <summary>
/// Managed values checked against the vendored faiss C++ headers.
/// </summary>
[Trait("Category", FaissHeaders.Trait)]
public class FaissHeaderMappingTests
{
    [Theory]
    [InlineData(RangeStat.RS_minmax, "RS_minmax")]
    [InlineData(RangeStat.RS_meanstd, "RS_meanstd")]
    [InlineData(RangeStat.RS_quantiles, "RS_quantiles")]
    [InlineData(RangeStat.RS_optim, "RS_optim")]
    public void RangeStat_ValuesMatchTheNativeEnumeration(RangeStat managed, string member)
    {
        // No exposing in the c_api thus we need to verify the source ourselves
        Assert.Equal(FaissHeaders.EnumOrdinal("faiss/impl/ScalarQuantizer.h", "RangeStat", member), (int)managed);
    }

    [Theory]
    [InlineData(IoFlags.SkipStorage, "IO_FLAG_SKIP_STORAGE")]
    [InlineData(IoFlags.ReadOnly, "IO_FLAG_READ_ONLY")]
    [InlineData(IoFlags.OnDiskSameDir, "IO_FLAG_ONDISK_SAME_DIR")]
    [InlineData(IoFlags.SkipIvfData, "IO_FLAG_SKIP_IVF_DATA")]
    [InlineData(IoFlags.SkipPrecomputeTable, "IO_FLAG_SKIP_PRECOMPUTE_TABLE")]
    [InlineData(IoFlags.PqSkipSdcTable, "IO_FLAG_PQ_SKIP_SDC_TABLE")]
    [InlineData(IoFlags.MmapIfc, "IO_FLAG_MMAP_IFC")]
    [InlineData(IoFlags.Mmap, "IO_FLAG_MMAP")]
    public void IoFlags_ValuesMatchTheFaissHeader(IoFlags managed, string constant)
    {
        Assert.Equal(FaissHeaders.IntConstant("faiss/index_io.h", constant), (int)managed);
    }

    /// <summary>
    /// The C API's own <c>FAISS_IO_FLAG_MMAP</c> is 1 (c_api/index_io_c.h), which is actually
    /// <c>IO_FLAG_SKIP_STORAGE</c>. <see cref="IoFlags"/> deliberately mirrors the C++ header instead,
    /// so a caller asking for mmap must not silently get skip-storage.
    /// </summary>
    [Fact]
    public void IoFlags_Mmap_IsNotTheMislabelledCApiMacro()
    {
        // Locals rather than constants, so the assertions read native-header value first, managed value second.
        var skipStorage = IoFlags.SkipStorage;
        var mmap = IoFlags.Mmap;
        var expected = FaissHeaders.IntConstant("faiss/index_io.h", "IO_FLAG_SKIP_IVF_DATA") | 0x646f0000;

        Assert.NotEqual(skipStorage, mmap);
        Assert.Equal(expected, (int)mmap);
    }
}
