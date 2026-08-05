using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Models;

namespace Faiss.Cpu.Factory;

public static class IndexFactory
{
    /// <summary>
    /// Creates a float index from a string descriptor.
    /// </summary>
    /// <param name="description">Factory string (e.g., "Flat", "IVF256,Flat", "HNSW32", "IVF256,PQ16").</param>
    /// <param name="dimensions">Vector dimensionality.</param>
    /// <param name="metric">Distance metric.</param>
    public static T Create<T>(string description, int dimensions, MetricType metric = MetricType.L2)
        where T : INativeIndex, IFromNativeIndexHandle<T>
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_index_factory(out IntPtr ptr, dimensions, description, metric)
        );

        return T.FromPointer(ptr);
    }
}