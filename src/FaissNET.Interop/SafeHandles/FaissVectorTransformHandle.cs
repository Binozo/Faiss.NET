using System.Runtime.InteropServices;
using Faiss.Interop.NativeMethods;
using Microsoft.Win32.SafeHandles;

namespace Faiss.Interop.SafeHandles;

internal class FaissVectorTransformHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public FaissVectorTransformHandle() : base(true)
    {
    }

    internal FaissVectorTransformHandle(IntPtr preexistingHandle) : base(true)
    {
        SetHandle(preexistingHandle);
    }

    protected override bool ReleaseHandle()
    {
        if (handle != IntPtr.Zero)
        {
            Native.faiss_VectorTransform_free(handle);
            handle = IntPtr.Zero;
        }

        return true;
    }
}

/// <summary>
/// A non-owning handle to a native VectorTransform that lives inside another native object,
/// such as a transform in an IndexPreTransform chain.
/// </summary>
/// <remarks>
/// This handle never frees the transform, instead it holds a reference on its owner's handle while open
/// to make sure the parent transform can never be freed while the borrowed handle is still active.
/// Created with <c>ownsHandle: true</c> only.
/// </remarks>
internal sealed class FaissBorrowedVectorTransformHandle : FaissVectorTransformHandle
{
    private readonly SafeHandle _owner;

    /// <param name="borrowed">Pointer to the transform; must not be null.</param>
    /// <param name="owner">Handle of the native object that owns <paramref name="borrowed"/>.</param>
    internal FaissBorrowedVectorTransformHandle(IntPtr borrowed, SafeHandle owner)
    {
        _owner = BorrowedHandle.Acquire(borrowed, owner);
        SetHandle(borrowed);
    }

    protected override bool ReleaseHandle()
    {
        _owner.DangerousRelease();
        return true;
    }
}
