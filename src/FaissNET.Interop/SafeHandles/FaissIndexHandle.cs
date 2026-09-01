namespace Faiss.Interop.SafeHandles;

using Microsoft.Win32.SafeHandles;

using NativeMethods;

internal interface IFaissRelease
{
    static abstract void Release(IntPtr handle);
}

public abstract class FaissHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    protected FaissHandle(bool ownsHandle) : base(ownsHandle) { }

    protected FaissHandle(IntPtr preexistingHandle, bool ownsHandle) : base(ownsHandle)
        => SetHandle(preexistingHandle);
}

internal readonly struct IndexRelease : IFaissRelease
{
    public static void Release(IntPtr handle) => Native.faiss_Index_free(handle);
}

public class FaissIndexHandle : FaissHandle
{
    public FaissIndexHandle() : base(true)
    { }

    internal FaissIndexHandle(IntPtr preexistingHandle, bool ownsHandle = true)
        : base(preexistingHandle, ownsHandle) { }

    protected override bool ReleaseHandle()
    {
        IndexRelease.Release(handle);
        return true;
    }
}

internal sealed class FaissIndexHandle<T> : FaissIndexHandle where T : struct, IFaissRelease
{
    public FaissIndexHandle()
    { }

    internal FaissIndexHandle(IntPtr preexistingHandle, bool ownsHandle = true)
        : base(preexistingHandle, ownsHandle) { }

    protected override bool ReleaseHandle()
    {
        T.Release(handle);
        return true;
    }
}

internal readonly struct BinaryIndexRelease : IFaissRelease
{
    public static void Release(IntPtr handle) => Native.faiss_IndexBinary_free(handle);
}

public class FaissBinaryIndexHandle : FaissHandle
{
    public FaissBinaryIndexHandle() :  base(true)
    { }

    internal FaissBinaryIndexHandle(IntPtr preexistingHandle, bool ownsHandle = true)
        : base(preexistingHandle, ownsHandle) { }

    protected override bool ReleaseHandle()
    {
        BinaryIndexRelease.Release(handle);
        return true;
    }
}

internal class FaissBinaryIndexHandle<T> : FaissBinaryIndexHandle where T : struct, IFaissRelease
{
    public FaissBinaryIndexHandle()
    { }

    internal FaissBinaryIndexHandle(IntPtr preexistingHandle, bool ownsHandle = true)
        : base(preexistingHandle, ownsHandle) { }

    protected override bool ReleaseHandle()
    {
        T.Release(handle);
        return true;
    }
}