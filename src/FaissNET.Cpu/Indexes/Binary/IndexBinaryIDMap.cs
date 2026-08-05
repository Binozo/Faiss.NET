using Faiss.Cpu.Factory;
using Faiss.Cpu.Interfaces;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Binary;

public sealed class IndexBinaryIDMap<T> : MappedBinaryIndex<IndexBinaryIDMap<T>, T>, IFromNativeBinaryIndexHandle<IndexBinaryIDMap<T>> where T : IIDSequentialBinaryIndex, IBinaryIndex, IFromNativeBinaryIndexHandle<T>
{
    public IndexBinaryIDMap(string description, int dimension) : this(CreateHandle($"IDMap,{description}", dimension))
    {
    }

    private IndexBinaryIDMap(FaissBinaryIndexHandle handle) : base(handle)
    {
    }

    private static FaissBinaryIndexHandle CreateHandle(string description, int dimension) => BinaryIndexFactory.Create<IndexBinaryIDMap<T>>(description, dimension).NativeHandle;

    static IndexBinaryIDMap<T> IFromNativeBinaryIndexHandle<IndexBinaryIDMap<T>>.FromHandle(FaissBinaryIndexHandle handle) => new(handle);
}