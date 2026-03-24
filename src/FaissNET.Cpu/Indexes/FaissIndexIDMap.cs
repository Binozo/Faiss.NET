namespace Faiss.Cpu.Indexes;

using Faiss.Interfaces;

using Interfaces;
using Interop.ErrorHandling;
using Interop.NativeMethods;
using Interop.SafeHandles;

/// <summary>
/// Index wrapper to handle custom database IDs.
/// </summary>
public sealed class FaissIndexIDMap : FaissCpuIndex, IFaissIndexWithIds, INativeFaissCpuIndex
{
    private readonly FaissIndexHandle _handle;
    
    private protected override FaissIndexHandle NativeHandle => _handle;
    
    // Keep reference to prevent GC fuckup
    private readonly IFaissIndex _nativeIndex;

    public FaissIndexIDMap(INativeFaissIndex nativeIndex)
    {
        _nativeIndex = nativeIndex;

        FaissErrorHandler.ThrowIfError(
            IndexIDMapNativeMethods.faiss_IndexIDMap_new(out IntPtr ptr, nativeIndex.Handle)
        );

        _handle = new FaissIndexHandle(ptr);
    }

    internal FaissIndexIDMap(IntPtr handle)
    {
        _handle = new FaissIndexHandle(handle);
    }

    static INativeFaissIndex INativeFaissCpuIndex.FromHandle(IntPtr handle) => new FaissIndexFlatL2(handle);

    public unsafe void AddWithIds(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids)
    {
        if (xids.Length < count)
        {
             throw new ArgumentException("Not enough custom IDs for the vectors.", nameof(xids));
        }

        fixed (float* pVectors = vectors)
        fixed (long* pXids = xids)
        {
            FaissErrorHandler.ThrowIfError(
                IndexIDMapNativeMethods.faiss_Index_add_with_ids(_handle.DangerousGetHandle(), count, pVectors, pXids)
            );
        }
    }
}