using Faiss.Cpu.Factory;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.Approximate;

/// <summary>
/// Hierarchical Navigable Small World index.
/// Industry-standard graph-based approximate nearest neighbor search.
/// </summary>
public sealed class IndexHNSW : CpuIndex<IndexHNSW>, IFromNativeHandle<IndexHNSW>
{
    /// <summary>
    /// Creates an HNSW index with flat (exact) storage.
    /// </summary>
    /// <param name="dimensions">Vector dimensionality</param>
    /// <param name="m">Number of neighbors per graph node (default 32)</param>
    /// <param name="metricType">Distance metric</param>
    public IndexHNSW(int dimensions, int m = 32, MetricType metricType = MetricType.L2)
    {
        // Factory string: "HNSW32" or "HNSW32,Flat" for flat storage
        string description = $"HNSW{m}";

        FaissErrorHandler.ThrowIfError(
            Native.faiss_index_factory(out IntPtr ptr, dimensions, description, metricType)
        );
        SafeHandle = new FaissIndexHandle(ptr);
    }

    /// <summary>
    /// Creates an HNSW index with Product Quantization storage.
    /// </summary>
    public static IndexHNSW WithProductQuantization(int dimensions, int m = 32, int pqBytes = 16, MetricType metricType = MetricType.L2)
    {
        string description = $"HNSW{m}_PQ{pqBytes}";
        return IndexFactory.Create<IndexHNSW>(description, dimensions, metricType);
    }

    /// <summary>
    /// Creates an HNSW index with Scalar Quantization storage.
    /// </summary>
    public static IndexHNSW WithScalarQuantization(int dimensions, int m = 32, QuantizerType qt = QuantizerType.QT_8bit, MetricType metricType = MetricType.L2)
    {
        string qtStr = qt switch
        {
            QuantizerType.QT_4bit => "SQ4",
            QuantizerType.QT_8bit => "SQ8",
            QuantizerType.QT_fp16 => "SQfp16",
            _ => throw new ArgumentException($"Unsupported quantizer for HNSW: {qt}")
        };

        string description = $"HNSW{m}_{qtStr}";
        return IndexFactory.Create<IndexHNSW>(description, dimensions, metricType);
    }

    private IndexHNSW(IntPtr handle) : base(handle)
    {
    }

    static IndexHNSW IFromNativeHandle<IndexHNSW>.FromHandle(IntPtr handle) => new(handle);

    /// <summary>
    /// Search with HNSW-specific parameters (efSearch, etc.).
    /// </summary>
    public void Search(long count, ReadOnlySpan<float> queryVectors, int k,
        SearchParametersHNSW parameters, Span<float> distances, Span<long> labels)
    {
        SearchWithParams(count, queryVectors, k, parameters, distances, labels);
    }
}