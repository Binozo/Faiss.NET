using Faiss.Cpu.Selectors;
using Faiss.Cpu.Indexes;
using Faiss.Cpu.Indexes.Sharding;
using Faiss.Cpu.Interfaces;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Models;
using Faiss.Gpu.Interop.NativeMethods;

namespace Faiss.Gpu.Indexes;

public class GpuIndexShards<TGpu, TCpu> : IndexShards, IGpuDevicesSquadIndex<TCpu>, IGpuIVFIndex, IGpuScalarQuantizerIndex
    where TCpu : INativeIndex, IFromNativeIndexHandle<TCpu>
    where TGpu : FloatIndex, INativeIndex, IFromNativeIndexHandle<TGpu>, IGpuDeviceIndex<TCpu>
{

    private GpuIndexShards(FaissIndexHandle handle) : base(handle)
    {
    }

    internal static GpuIndexShards<TGpu, TCpu> FromPointer(IntPtr handle) => new(new FaissIndexHandle<IndexShardsRelease>(handle, ownsHandle: true));

    /// <inheritdoc cref="IndexShards.this[int]"/>
    public new TGpu this[int index] => (TGpu)base[index];

    private protected override INativeIndex CreateView(IntPtr member) => TGpu.FromBorrowed(member, NativeHandle);

    public override void AddIndex(INativeIndex index)
    {
        if (index is not TGpu)
            throw new ArgumentException($"Only {typeof(TGpu).Name} shards can be added.", nameof(index));

        base.AddIndex(index);
        QuantizerType = GpuScalarQuantizerLimits.Combine(QuantizerType, index);
    }

    /// <summary>The quantizer of the scalar quantizer shards.</summary>
    internal QuantizerType? QuantizerType { get; private set; }

    QuantizerType? IGpuScalarQuantizerIndex.QuantizerType
    {
        get => QuantizerType;
        set => QuantizerType = value;
    }

    internal override void ThrowIfKUnsupported(long k) => GpuScalarQuantizerLimits.ThrowIfKUnsupported(QuantizerType, Dimensions, k);

    bool IGpuIVFIndex.Uses32BitIds { get; set; }

    /// <inheritdoc />
    public override void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids)
    {
        GpuIVFIDs.ThrowIfNot32Bit(this, xids);
        base.Add(count, vectors, xids);
    }

    public override long RemoveIds(IDSelector selector)
    {
        throw new NotSupportedException("Removing IDs is not supported on the gpu.");
    }

    /// <summary>The device of every shard, in shard order.</summary>
    public int[] DeviceIds
    {
        get
        {
            var added = false;
            NativeHandle.DangerousAddRef(ref added);
            try
            {
                var deviceIds = new int[Count];
                for (var i = 0; i < deviceIds.Length; i++)
                    deviceIds[i] = GpuNative.faiss_GpuIndex_getDevice_ptr(Native.faiss_IndexShards_at(NativeHandle, i));

                return deviceIds;
            }
            finally
            {
                if (added)
                    NativeHandle.DangerousRelease();
            }
        }
    }
}