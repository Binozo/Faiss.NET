using System.Runtime.InteropServices;

namespace Faiss.Interop.SafeHandles;

internal abstract class FaissScalarQuantizerHandle : FaissHandle
{
    protected FaissScalarQuantizerHandle() : base(ownsHandle: true)
    {
    }
}

internal sealed class FaissBorrowedScalarQuantizerHandle : FaissScalarQuantizerHandle
{
    private readonly SafeHandle _owner;

    /// <param name="borrowed">Pointer to the ScalarQuantizer. Must not be null.</param>
    /// <param name="owner">Handle of the index that contains <paramref name="borrowed"/>.</param>
    internal FaissBorrowedScalarQuantizerHandle(IntPtr borrowed, SafeHandle owner)
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
