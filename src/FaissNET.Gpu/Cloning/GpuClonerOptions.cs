using System.Runtime.InteropServices;
using Faiss.Exceptions;
using Faiss.Gpu.Interop.NativeMethods;
using Faiss.Gpu.Interop.SafeHandles;
using Faiss.Interop.Errors;

namespace Faiss.Gpu.Cloning;

/// <summary>
/// Provides options that control how a Faiss index is cloned onto a GPU.
/// </summary>
/// <remarks>
/// These settings affect memory layout, quantization precision, and
/// index ID storage mode on the device.
/// </remarks>
public class GpuClonerOptions : IDisposable
{
    private readonly SafeHandle _handle;

    internal GpuClonerOptionsHandle NativeHandle => (GpuClonerOptionsHandle)_handle;

    protected SafeHandle GetUnderlyingHandle() => _handle;

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuClonerOptions"/> class
    /// with default cloning settings.
    /// </summary>
    /// <exception cref="FaissException">
    /// Thrown when the native options allocation fails.
    /// </exception>
    public GpuClonerOptions()
    {
        FaissErrorHandler.ThrowIfError(
            GpuNative.faiss_GpuClonerOptions_new(out IntPtr ptr)
        );

        _handle = new GpuClonerOptionsHandle(ptr);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="GpuClonerOptions"/> class
    /// using an existing native handle.
    /// </summary>
    /// <param name="handle">The native handle to wrap.</param>
    /// <remarks>
    /// This constructor is intended for use by derived types such as
    /// <see cref="GpuMultipleClonerOptions"/>.
    /// </remarks>
    protected GpuClonerOptions(SafeHandle handle)
    {
        _handle = handle;
    }

    /// <summary>
    /// Gets or sets whether to use 16-bit floating-point storage for vectors on the GPU.
    /// </summary>
    /// <value>
    /// <c>true</c> to use FP16 storage; <c>false</c> to use FP32.
    /// </value>
    /// <remarks>
    /// Enabling this reduces GPU memory usage by half but may slightly affect accuracy.
    /// It affects intermediate calculations for GPU IVF-PQ and storage for IVF scalar quantizer indexes.
    /// It does not affect <c>GpuIndexIVFFlat</c> storage.
    /// </remarks>
    public bool UseFloat16
    {
        get => GpuNative.faiss_GpuClonerOptions_useFloat16(NativeHandle) != 0;
        set => GpuNative.faiss_GpuClonerOptions_set_useFloat16(NativeHandle, value ? 1 : 0);
    }

    /// <summary>
    /// Gets or sets whether the coarse quantizer uses 16-bit floating-point storage.
    /// </summary>
    /// <value>
    /// <c>true</c> if the coarse quantizer is stored in FP16; otherwise, <c>false</c>.
    /// </value>
    public bool UseFloat16CoarseQuantizer
    {
        get => GpuNative.faiss_GpuClonerOptions_useFloat16CoarseQuantizer(NativeHandle) != 0;
        set => GpuNative.faiss_GpuClonerOptions_set_useFloat16CoarseQuantizer(NativeHandle, value ? 1 : 0);
    }

    /// <summary>
    /// Gets or sets whether to precompute distance tables for faster search.
    /// </summary>
    /// <value>
    /// <c>true</c> to use precomputed tables; otherwise, <c>false</c>.
    /// </value>
    /// <remarks>
    /// This trades a small amount of additional VRAM for improved search throughput.
    /// </remarks>
    public bool UsePrecomputed
    {
        get => GpuNative.faiss_GpuClonerOptions_usePrecomputed(NativeHandle) != 0;
        set => GpuNative.faiss_GpuClonerOptions_set_usePrecomputed(NativeHandle, value ? 1 : 0);
    }

    /// <summary>
    /// Gets or sets whether vectors are stored in transposed layout.
    /// </summary>
    /// <value>
    /// <c>true</c> to store data transposed; otherwise, <c>false</c>.
    /// </value>
    /// <remarks>
    /// This only affects <c>GpuIndexFlat</c>. Transposed layout can improve
    /// memory coalescing for certain query patterns.
    /// </remarks>
    public bool StoreTransposed
    {
        get => GpuNative.faiss_GpuClonerOptions_storeTransposed(NativeHandle) != 0;
        set => GpuNative.faiss_GpuClonerOptions_set_storeTransposed(NativeHandle, value ? 1 : 0);
    }

    /// <summary>
    /// Gets or sets whether verbose logging is enabled during cloning.
    /// </summary>
    /// <value>
    /// <c>true</c> to enable verbose output; otherwise, <c>false</c>.
    /// </value>
    public bool Verbose
    {
        get => GpuNative.faiss_GpuClonerOptions_verbose(NativeHandle) != 0;
        set => GpuNative.faiss_GpuClonerOptions_set_verbose(NativeHandle, value ? 1 : 0);
    }

    /// <summary>
    /// Gets or sets how vector indices are stored on the GPU.
    /// </summary>
    /// <value>
    /// One of the <see cref="IndicesOptions"/> values.
    /// </value>
    /// <remarks>
    /// This controls the memory footprint and lookup behavior for index IDs.
    /// Use <see cref="IndicesOptions.Cpu"/> to minimize GPU memory,
    /// or <see cref="IndicesOptions.Bit64"/> for full 64-bit index storage.
    /// </remarks>
    public IndicesOptions StorageMode
    {
        get => (IndicesOptions)GpuNative.faiss_GpuClonerOptions_indicesOptions(NativeHandle);
        set => GpuNative.faiss_GpuClonerOptions_set_indicesOptions(NativeHandle, (int)value);
    }

    /// <summary>
    /// Gets or sets the number of vectors to reserve in the inverted file lists.
    /// </summary>
    /// <value>
    /// The number of vectors to pre-allocate. <c>0</c> means no pre-allocation.
    /// </value>
    /// <remarks>
    /// Pre-allocating avoids repeated GPU memory reallocations during
    /// incremental index building.
    /// </remarks>
    public long ReserveVecs
    {
        get => GpuNative.faiss_GpuClonerOptions_reserveVecs(NativeHandle).Value.ToInt64();
        set
        {
            var clong = Marshal.SizeOf<CLong>() == sizeof(int)
                ? new CLong(checked((int)value))
                : new CLong(new IntPtr(value));
            GpuNative.faiss_GpuClonerOptions_set_reserveVecs(NativeHandle, clong);
        }
    }

    /// <summary>
    /// Releases the native cloning options held by this instance.
    /// </summary>
    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}