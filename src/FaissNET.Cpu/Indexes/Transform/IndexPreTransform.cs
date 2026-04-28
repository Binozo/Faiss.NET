using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Transforms;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Transform;

/// <summary>
/// Wraps an index and applies a chain of vector transforms before adding or searching vectors.
/// </summary>
public sealed class IndexPreTransform : CpuIndex<IndexPreTransform>, IFromNativeHandle<IndexPreTransform>
{
    private readonly INativeIndex _index;
    private readonly List<VectorTransform> _chain = new();

    public IndexPreTransform(VectorTransform transform, INativeIndex index, bool takeOwnership = false)
    {
        _index = index;
        
        FaissErrorHandler.ThrowIfError(
            Native.faiss_IndexPreTransform_new_with_transform(
                out IntPtr ptr,
                transform.SafeHandle.DangerousGetHandle(),
                index.Handle)
        );
        
        SafeHandle = new FaissIndexHandle(ptr);
        
        Native.faiss_IndexPreTransform_set_own_fields(SafeHandle, takeOwnership);
        
        // IndexPreTransform always owns the transform chain
        transform.ReleaseOwnership();
        
        _chain.Add(transform);

        if (takeOwnership)
        {
            index.Handle.SetHandleAsInvalid();
        }
    }

    private IndexPreTransform(IntPtr handle) : base(handle)
    {
        Native.faiss_IndexPreTransform_set_own_fields(SafeHandle, true);
    }

    static IndexPreTransform IFromNativeHandle<IndexPreTransform>.FromHandle(IntPtr handle) => new(handle);

    /// <summary>
    /// Prepends a transform to the chain. The prepended transform is applied first during add and search operations.
    /// </summary>
    /// <param name="transform">The transform to prepend.</param>
    public void PrependTransform(VectorTransform transform)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_IndexPreTransform_prepend_transform(
                SafeHandle,
                transform.SafeHandle.DangerousGetHandle())
        );

        transform.ReleaseOwnership();
        _chain.Add(transform);
    }

    /// <summary>
    /// Gets the transform chain in application order.
    /// </summary>
    public IReadOnlyList<VectorTransform> TransformChain => _chain.AsReadOnly();
}