using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Search;

namespace Faiss.Cpu.Indexes;

/// <inheritdoc cref="ICpuIndex" />
public abstract class CpuIndex<T> : Index<T>, ICpuIndex<T> where T : Index<T>, INativeIndex<T>, IFromNativeHandle<T>
{
    protected CpuIndex()
    {
        
    }

    protected CpuIndex(IntPtr handle) : base(handle)
    {
        
    }

    public long RemoveIds(IIDSelector selector)
    {
        ArgumentNullException.ThrowIfNull(selector);

        FaissErrorHandler.ThrowIfError(
            Native.faiss_Index_remove_ids(SafeHandle, selector.ToNative(), out nuint removedCount)
        );

        return (long)removedCount;
    }
    
    public float[] Reconstruct(long key)
    {
        float[] vector = new float[Dimensions];
        unsafe
        {
            fixed (float* pVector = vector)
            {
                FaissErrorHandler.ThrowIfError(
                    Native.faiss_Index_reconstruct(SafeHandle, key, pVector)
                );
            }
        }
        return vector;
    }

    public float[] ReconstructBatch(long startKey, long count)
    {
        float[] vectors = new float[count * Dimensions];
        unsafe
        {
            fixed (float* pVectors = vectors)
            {
                FaissErrorHandler.ThrowIfError(
                    Native.faiss_Index_reconstruct_n(SafeHandle, startKey, count, pVectors)
                );
            }
        }
        return vectors;
    }

    public T Clone()
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_clone_index(SafeHandle, out IntPtr clonedPtr)
        );

        return T.FromHandle(clonedPtr);
    }
}