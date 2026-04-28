using Faiss.Interfaces;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Interfaces;

/// <inheritdoc/>
public interface INativeIndex : IIndex
{
    protected internal FaissIndexHandle Handle { get; }
}

public interface INativeIndex<TSelf> : INativeIndex
    where TSelf : INativeIndex<TSelf>
{
}

public interface IFromNativeHandle<TSelf>
    where TSelf : INativeIndex<TSelf>
{
    internal static abstract TSelf FromHandle(IntPtr handle);
}
