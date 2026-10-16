using System.Runtime.InteropServices;
using Faiss.Interfaces;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Interfaces;

// TODO: Can things be made internal?

public interface INativeObject
{
    internal FaissHandle Handle { get; }
}

public interface INativeObject<out T> : INativeObject where T : FaissHandle
{
    protected internal new T Handle { get; }

    FaissHandle INativeObject.Handle => Handle;
}

/// <inheritdoc cref="IIndex" />
public interface INativeIndex : IFloatIndex, INativeObject<FaissIndexHandle>
{
    internal new FaissIndexHandle Handle { get; }

    FaissIndexHandle INativeObject<FaissIndexHandle>.Handle => Handle;
}

/// <inheritdoc cref="IIndex" />
public interface INativeBinaryIndex : IBinaryIndex, INativeObject<FaissBinaryIndexHandle>
{
    internal new FaissBinaryIndexHandle Handle { get; }

    FaissBinaryIndexHandle INativeObject<FaissBinaryIndexHandle>.Handle => Handle;
}

public interface IFromNativeIndexHandle<T> where T : IFromNativeIndexHandle<T>
{
    /// <summary>Wraps <paramref name="handle"/>. The wrapper's lifetime and ownership are exactly those of the handle.</summary>
    internal static abstract T FromHandle(FaissIndexHandle handle);

    internal static virtual T FromPointer(IntPtr ptr, bool ownsHandle = true)
        => T.FromHandle(new FaissIndexHandle(ptr, ownsHandle));

    /// <summary>
    /// Wraps a native sub-object of <paramref name="owner"/> without taking ownership.
    /// The wrapper never frees <paramref name="ptr"/>, and keeps <paramref name="owner"/> alive for as long as it is open, so it cannot dangle.
    /// </summary>
    internal static virtual T FromBorrowed(IntPtr ptr, SafeHandle owner)
    {
        var handle = new FaissBorrowedIndexHandle(ptr, owner);
        try
        {
            return T.FromHandle(handle);
        }
        catch
        {
            // Release the reference on the owner now rather than whenever the handle is finalized.
            handle.Dispose();
            throw;
        }
    }
}

public interface IFromNativeBinaryIndexHandle<T> where T : IFromNativeBinaryIndexHandle<T>
{
    /// <inheritdoc cref="IFromNativeIndexHandle{T}.FromHandle"/>
    internal static abstract T FromHandle(FaissBinaryIndexHandle handle);

    internal static virtual T FromPointer(IntPtr handle, bool ownsHandle = true)
        => T.FromHandle(new FaissBinaryIndexHandle(handle, ownsHandle));

    /// <inheritdoc cref="IFromNativeIndexHandle{T}.FromBorrowed"/>
    internal static virtual T FromBorrowed(IntPtr ptr, SafeHandle owner)
    {
        var handle = new FaissBorrowedBinaryIndexHandle(ptr, owner);
        try
        {
            return T.FromHandle(handle);
        }
        catch
        {
            handle.Dispose();
            throw;
        }
    }
}