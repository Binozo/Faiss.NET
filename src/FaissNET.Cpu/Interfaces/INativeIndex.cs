using Faiss.Interfaces;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Interfaces;

public interface INativeObject
{
    protected internal FaissHandle Handle { get; }
}

public interface INativeObject<out T> : INativeObject where T : FaissHandle
{
    protected internal new T Handle { get; }

    FaissHandle INativeObject.Handle => Handle;
}

/// <inheritdoc cref="IIndex" />
public interface INativeIndex : IIndex, INativeObject<FaissIndexHandle>
{
    protected internal new FaissIndexHandle Handle { get; }

    FaissIndexHandle INativeObject<FaissIndexHandle>.Handle => Handle;
}

/// <inheritdoc cref="IIndex" />
public interface INativeBinaryIndex : IIndex, INativeObject<FaissBinaryIndexHandle>
{
    protected internal new FaissBinaryIndexHandle Handle { get; }

    FaissBinaryIndexHandle INativeObject<FaissBinaryIndexHandle>.Handle => Handle;
}

public interface IFromNativeIndexHandle<T> where T : IFromNativeIndexHandle<T>
{
    [Obsolete("Use FromPointer instead")]
    internal static abstract T FromHandle(FaissIndexHandle handle);

    internal static virtual T FromPointer(IntPtr ptr, bool ownsHandle = true)
        => T.FromHandle(new FaissIndexHandle(ptr, ownsHandle));
}

public interface IFromNativeBinaryIndexHandle<T> where T : IFromNativeBinaryIndexHandle<T>
{
    [Obsolete("Use FromPointer instead")]
    internal static abstract T FromHandle(FaissBinaryIndexHandle handle);

    internal static virtual T FromPointer(IntPtr handle, bool ownsHandle = true)
        => T.FromHandle(new FaissBinaryIndexHandle(handle, ownsHandle));
}