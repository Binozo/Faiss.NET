using Faiss.Interfaces;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Interfaces;

/// <inheritdoc/>
public interface INativeBinaryIndex : IBinaryIndex
{
    protected internal FaissIndexBinaryHandle Handle { get; }
}

public interface INativeBinaryIndex<TSelf> : INativeBinaryIndex
    where TSelf : INativeBinaryIndex<TSelf>
{
}

public interface IFromNativeBinaryHandle<TSelf>
    where TSelf : INativeBinaryIndex<TSelf>
{
    internal static abstract TSelf FromHandle(IntPtr handle);
}