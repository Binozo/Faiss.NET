using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Binary;

/// <summary>
/// Locality-Sensitive Hashing index.
/// Hashes vectors into compact binary signatures for fast approximate search.
/// </summary>
public sealed class IndexLSH : CpuIndex<IndexLSH>, IFromNativeHandle<IndexLSH>
{
    public IndexLSH(int dimensions, int nbits, bool rotateData = true, bool trainThresholds = false)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_IndexLSH_new_with_options(out IntPtr ptr, dimensions, nbits, rotateData, trainThresholds)
        );
        
        SafeHandle = new FaissIndexHandle(ptr);
    }

    private IndexLSH(IntPtr handle) : base(handle)
    {
        
    }

    static IndexLSH IFromNativeHandle<IndexLSH>.FromHandle(IntPtr handle) => new(handle);

    /// <summary>Number of bits per binary signature.</summary>
    public int Nbits => Native.faiss_IndexLSH_nbits(SafeHandle);
    
    /// <summary>Size of the binary code in bytes.</summary>
    public int CodeSize => Native.faiss_IndexLSH_code_size(SafeHandle);
    
    /// <summary>Whether random rotation is applied before hashing.</summary>
    public bool RotateData => Native.faiss_IndexLSH_rotate_data(SafeHandle) != 0;
    
    /// <summary>Whether thresholds are trained from data.</summary>
    public bool TrainThresholds => Native.faiss_IndexLSH_train_thresholds(SafeHandle) != 0;
}