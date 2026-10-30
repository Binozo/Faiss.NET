using Faiss.Cpu.Interfaces;
using Faiss.Gpu.Cloning;
using Faiss.Interop.NativeMethods;

namespace Faiss.Gpu.Interop;

/// <summary>Enforces the id range of <see cref="IndicesOptions.Bit32"/> storage.</summary>
internal static class Bit32Ids
{
    /// <summary>
    /// Rejects an IVF source holding an id outside the 32-bit range.
    /// </summary>
    public static unsafe void ThrowIfSourceHoldsWideIds(INativeIndex cpuIndex, GpuClonerOptions options)
    {
        if (options.StorageMode != IndicesOptions.Bit32 || cpuIndex is not IIVFIndex ivf)
            return;

        var handle = cpuIndex.Handle;
        long[] ids = [];

        for (var list = 0; list < ivf.Nlist; list++)
        {
            var size = (int)Native.faiss_IndexIVF_get_list_size(handle, (nuint)list);
            if (size == 0)
                continue;

            if (ids.Length < size)
                ids = new long[size];

            fixed (long* pIds = ids)
                Native.faiss_IndexIVF_invlists_get_ids(handle, (nuint)list, (IntPtr)pIds);

            foreach (var id in ids.AsSpan(0, size))
            {
                if (id is < int.MinValue or > int.MaxValue)
                    throw new ArgumentException($"The index holds id {id}, which does not fit the 32-bit id storage of {nameof(IndicesOptions)}.{nameof(IndicesOptions.Bit32)}.", nameof(cpuIndex));
            }
        }
    }

    /// <summary>Makes a device IVF index transferred with 32-bit id storage reject wider ids added later.</summary>
    public static void Mark(object gpuIndex, GpuClonerOptions options)
    {
        if (options.StorageMode == IndicesOptions.Bit32 && gpuIndex is IGpuIVFIndex ivf)
            ivf.Uses32BitIds = true;
    }
}
