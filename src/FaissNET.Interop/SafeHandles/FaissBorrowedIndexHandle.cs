using System.Runtime.InteropServices;

namespace Faiss.Interop.SafeHandles;

internal static class BorrowedHandle
{
    /// <summary>
    /// Takes a reference on <paramref name="owner"/> so its native memory
    /// cannot be released while the borrowing handle is open.
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="borrowed"/> is null.</exception>
    /// <exception cref="ObjectDisposedException"><paramref name="owner"/> is already closed.</exception>
    internal static SafeHandle Acquire(IntPtr borrowed, SafeHandle owner)
    {
        ArgumentNullException.ThrowIfNull(owner);

        if (borrowed == IntPtr.Zero)
            throw new ArgumentException("The borrowed native pointer is null.", nameof(borrowed));

        var added = false;
        owner.DangerousAddRef(ref added);
        return owner;
    }
}

/// <summary>
/// A non-owning handle to a native index that lives inside another native object,
/// such as the sub-index of an IndexIDMap or the base index of an IndexRefineFlat for instance.
/// </summary>
/// <remarks>
/// See <see cref="BorrowedHandle"/>.
/// </remarks>
internal sealed class FaissBorrowedIndexHandle : FaissIndexHandle
{
    private readonly SafeHandle _owner;

    /// <param name="borrowed">Pointer to the sub-object. Must not be null.</param>
    /// <param name="owner">Handle of the native object that owns, or outlives, <paramref name="borrowed"/>.</param>
    internal FaissBorrowedIndexHandle(IntPtr borrowed, SafeHandle owner)
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

/// <inheritdoc cref="FaissBorrowedIndexHandle"/>
internal sealed class FaissBorrowedBinaryIndexHandle : FaissBinaryIndexHandle
{
    private readonly SafeHandle _owner;

    /// <param name="borrowed">Pointer to the sub-object. Must not be null.</param>
    /// <param name="owner">Handle of the native object that owns, or outlives, <paramref name="borrowed"/>.</param>
    internal FaissBorrowedBinaryIndexHandle(IntPtr borrowed, SafeHandle owner)
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
