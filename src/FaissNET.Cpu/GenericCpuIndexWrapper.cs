namespace Faiss.Cpu;

using Interfaces;
using Interop.ErrorHandling;
using Interop.NativeMethods;
using Interop.SafeHandles;

public sealed class GenericCpuIndexWrapper : IFaissIndex, INativeIndex
{
    private readonly FaissIndexHandle _handle;

    public IntPtr Handle => _handle.DangerousGetHandle();

    internal GenericCpuIndexWrapper(IntPtr ptr)
    {
        _handle = new FaissIndexHandle(ptr);
    }

    public int Dimensions => Native.faiss_Index_d(_handle);
    
    public long TotalCount => Native.faiss_Index_ntotal(_handle);
    
    public bool IsTrained => Native.faiss_Index_is_trained(_handle) != 0;

    public unsafe void Add(long count, ReadOnlySpan<float> vectors)
    {
        fixed (float* pVectors = vectors)
        {
            FaissErrorHandler.ThrowIfError(Native.faiss_Index_add(_handle, count, pVectors));
        }
    }

    public unsafe void Search(long count, ReadOnlySpan<float> queryVectors, int k, Span<float> distances, Span<long> labels)
    {
        fixed (float* pQuery = queryVectors)
        fixed (float* pDistances = distances)
        fixed (long* pLabels = labels)
        {
            FaissErrorHandler.ThrowIfError(Native.faiss_Index_search(_handle, count, pQuery, k, pDistances, pLabels));
        }
    }

    public void Reset()
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_Index_reset(_handle));
    }

    public void Dispose()
    {
        _handle.Dispose();
    }
}