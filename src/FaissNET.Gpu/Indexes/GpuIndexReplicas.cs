using Faiss.Cpu.Indexes;
using Faiss.Cpu.Indexes.Replicas;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Selectors;
using Faiss.Gpu.Interop.NativeMethods;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Gpu.Indexes;

public class GpuIndexReplicas<TGpu, TCpu> : IndexReplicas<TGpu>, IGpuDevicesSquadIndex<TCpu>, IGpuIVFIndex, IGpuScalarQuantizerIndex
    where TCpu : INativeIndex, IFromNativeIndexHandle<TCpu>
    where TGpu : FloatIndex, INativeIndex, IFromNativeIndexHandle<TGpu>, IGpuDeviceIndex<TCpu>
{
    private GpuIndexReplicas(long dimensions) : base(dimensions, true, true)
    {
        // Not callable
    }

    private GpuIndexReplicas(FaissIndexHandle handle) : base(handle)
    {
    }

    internal static GpuIndexReplicas<TGpu, TCpu> FromPointer(IntPtr handle) => new(new FaissIndexHandle<IndexReplicasRelease>(handle, ownsHandle: true));

    public override void AddReplica(TGpu replica)
    {
        base.AddReplica(replica);
        QuantizerType = GpuScalarQuantizerLimits.Combine(QuantizerType, replica);
    }

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

    /// <summary>The quantizer of the scalar quantizer replicas.</summary>
    internal QuantizerType? QuantizerType { get; private set; }

    QuantizerType? IGpuScalarQuantizerIndex.QuantizerType
    {
        get => QuantizerType;
        set => QuantizerType = value;
    }

    internal override void ThrowIfKUnsupported(long k) => GpuScalarQuantizerLimits.ThrowIfKUnsupported(QuantizerType, Dimensions, k);

    bool IGpuIVFIndex.Uses32BitIds { get; set; }
    
    /// <summary>The device of every replica, in replica order.</summary>
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
                    deviceIds[i] = GpuNative.faiss_GpuIndex_getDevice_ptr(Native.faiss_IndexReplicas_at(NativeHandle, i));

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