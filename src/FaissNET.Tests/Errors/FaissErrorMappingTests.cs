using Faiss.Cpu.Factory;
using Faiss.Cpu.Indexes.Factory;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Exceptions;
using Faiss.Models;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Errors;
public class FaissErrorMappingTests
{
    [Fact]
    public void FaissException_MapsToFaissNativeException()
    {
        // An unparseable factory string raises faiss::FaissException (code -2).
        var ex = Assert.Throws<FaissNativeException>(
            () => IndexFactory.Create<GenericFloatIndex>("NotAnIndex", 8, MetricType.L2)
        );

        Assert.Contains("index_factory", ex.Message);
    }

    [Fact]
    public void StdException_MapsToFaissRuntimeException()
    {
        // An nlist that overflows the parser raises std::out_of_range.
        var ex = Assert.Throws<FaissRuntimeException>(
            () => IndexFactory.Create<GenericFloatIndex>("IVF99999999999999999999,Flat", 8, MetricType.L2)
        );

        Assert.Contains("std::exception", ex.Message);
    }

    [Fact]
    public void EveryFaissErrorSharesTheFaissExceptionBaseType()
    {
        Assert.True(typeof(FaissException).IsAssignableFrom(typeof(FaissNativeException)));
        Assert.True(typeof(FaissException).IsAssignableFrom(typeof(FaissRuntimeException)));
        Assert.True(typeof(FaissException).IsAssignableFrom(typeof(FaissUnknownException)));
        Assert.True(typeof(FaissException).IsAssignableFrom(typeof(FaissArgumentException)));
        Assert.True(typeof(FaissException).IsAssignableFrom(typeof(FaissUntrainedException)));

        var caught = Record.Exception(() => IndexFactory.Create<GenericFloatIndex>("NotAnIndex", 8, MetricType.L2));
        Assert.IsAssignableFrom<FaissException>(caught);
    }

    [Fact]
    public void Message_IsPrefixedWithTheFailingOperation()
    {
        using var index = new IndexFlatL2(4);
        index.Add(4, Vectors.Random(seed: 190, count: 4, dimensions: 4));

        var ex = Assert.Throws<FaissNativeException>(() => index.Reconstruct(99));

        Assert.StartsWith("Reconstruct failed: ", ex.Message);
    }

    [Fact]
    public void Message_CarriesTheNativeErrorText()
    {
        using var index = new IndexFlatL2(4);

        var ex = Assert.Throws<FaissNativeException>(() => index.Reconstruct(0));
        Assert.DoesNotContain("Unknown Faiss error occurred", ex.Message);
        Assert.Contains("Error in", ex.Message);
    }
}
