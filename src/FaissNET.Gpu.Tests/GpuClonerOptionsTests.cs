namespace Faiss.Gpu.Tests;

using Xunit;

using Gpu.Cloning;

/// <summary>
/// Tests for <see cref="GpuClonerOptions"/> property getters and setters.
/// </summary>
public class GpuClonerOptionsTests
{
    [Fact]
    public void UseFloat16_DefaultIsFalse()
    {
        using var options = new GpuClonerOptions();
        Assert.False(options.UseFloat16);
    }

    [Fact]
    public void UseFloat16_CanBeSetAndRetrieved()
    {
        using var options = new GpuClonerOptions();
        options.UseFloat16 = true;
        Assert.True(options.UseFloat16);
    }

    [Fact]
    public void UseFloat16CoarseQuantizer_DefaultIsFalse()
    {
        using var options = new GpuClonerOptions();
        Assert.False(options.UseFloat16CoarseQuantizer);
    }

    [Fact]
    public void UseFloat16CoarseQuantizer_CanBeSetAndRetrieved()
    {
        using var options = new GpuClonerOptions();
        options.UseFloat16CoarseQuantizer = true;
        Assert.True(options.UseFloat16CoarseQuantizer);
    }

    [Fact]
    public void UsePrecomputed_DefaultIsFalse()
    {
        using var options = new GpuClonerOptions();
        Assert.False(options.UsePrecomputed);
    }

    [Fact]
    public void UsePrecomputed_CanBeSetAndRetrieved()
    {
        using var options = new GpuClonerOptions();
        options.UsePrecomputed = true;
        Assert.True(options.UsePrecomputed);
    }

    [Fact]
    public void StoreTransposed_DefaultIsFalse()
    {
        using var options = new GpuClonerOptions();
        Assert.False(options.StoreTransposed);
    }

    [Fact]
    public void StoreTransposed_CanBeSetAndRetrieved()
    {
        using var options = new GpuClonerOptions();
        options.StoreTransposed = true;
        Assert.True(options.StoreTransposed);
    }

    [Fact]
    public void Verbose_DefaultIsFalse()
    {
        using var options = new GpuClonerOptions();
        Assert.False(options.Verbose);
    }

    [Fact]
    public void Verbose_CanBeSetAndRetrieved()
    {
        using var options = new GpuClonerOptions();
        options.Verbose = true;
        Assert.True(options.Verbose);
    }

    [Fact]
    public void StorageMode_CanBeSetAndRetrieved()
    {
        using var options = new GpuClonerOptions();

        var original = options.StorageMode;

        options.StorageMode = IndicesOptions.Bit64;
        Assert.Equal(IndicesOptions.Bit64, options.StorageMode);

        options.StorageMode = IndicesOptions.Bit32;
        Assert.Equal(IndicesOptions.Bit32, options.StorageMode);

        options.StorageMode = IndicesOptions.Cpu;
        Assert.Equal(IndicesOptions.Cpu, options.StorageMode);

        options.StorageMode = original;
        Assert.Equal(original, options.StorageMode);
    }

    [Fact]
    public void ReserveVecs_DefaultIsZero()
    {
        using var options = new GpuClonerOptions();
        Assert.Equal(0, options.ReserveVecs);
    }

    [Fact]
    public void ReserveVecs_CanBeSetAndRetrieved()
    {
        using var options = new GpuClonerOptions();
        options.ReserveVecs = 1024;
        Assert.Equal(1024, options.ReserveVecs);
    }

    [Fact]
    public void MultipleProperties_CanBeSetTogether()
    {
        using var options = new GpuClonerOptions
        {
            UseFloat16 = true,
            UsePrecomputed = true,
            StoreTransposed = false,
            Verbose = false,
            StorageMode = IndicesOptions.Bit32,
            ReserveVecs = 2048
        };

        Assert.True(options.UseFloat16);
        Assert.True(options.UsePrecomputed);
        Assert.False(options.StoreTransposed);
        Assert.False(options.Verbose);
        Assert.Equal(IndicesOptions.Bit32, options.StorageMode);
        Assert.Equal(2048, options.ReserveVecs);
    }
}
