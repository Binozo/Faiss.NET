using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Interfaces;

internal interface IKeyLayout
{
    KeyLayout KeyLayout { get; }
}

/// <summary>
/// How an index interprets the keys of <c>Reconstruct</c> and <c>ComputeResidual</c>, and whether faiss validates a
/// batch reconstruct before entering its OpenMP loop.
/// </summary>
/// <remarks>
/// As faiss's default <c>Index::reconstruct_n</c> and every <c>Index::compute_residual_n</c> call the per-key method inside
/// an OpenMP region a key the index rejects there cannot propagate and aborts the process. The binding therefore must
/// never pass such a call a key it cannot prove valid.
/// </remarks>
/// <param name="KeysAreIds">Keys are user ids (any 64-bit value) rather than positions <c>0..TotalCount-1</c>.</param>
/// <param name="NativeChecksBatchReconstruct">The index's own <c>reconstruct_n</c> rejects invalid ranges before any OpenMP loop.</param>
internal readonly record struct KeyLayout(bool KeysAreIds, bool NativeChecksBatchReconstruct)
{
    /// <summary>Positions; faiss's default <c>reconstruct_n</c> (HNSW, NSG, RefineFlat, Replicas). The default for undeclared types.</summary>
    public static KeyLayout Positions => new(KeysAreIds: false, NativeChecksBatchReconstruct: false);

    /// <summary>Positions; <c>IndexFlatCodes::reconstruct_n</c> checks the range first (Flat, PQ, LSH, SQ).</summary>
    public static KeyLayout CheckedPositions => new(KeysAreIds: false, NativeChecksBatchReconstruct: true);

    /// <summary>User ids; faiss's default <c>reconstruct_n</c> (IDMap, IDMap2).</summary>
    public static KeyLayout Ids => new(KeysAreIds: true, NativeChecksBatchReconstruct: false);

    /// <summary>User ids; <c>IndexIVF::reconstruct_n</c> checks the range first and walks the lists without a direct map.</summary>
    public static KeyLayout CheckedIds => new(KeysAreIds: true, NativeChecksBatchReconstruct: true);

    public static KeyLayout Of(INativeIndex index) => index is IKeyLayout declared ? declared.KeyLayout : Positions;

    /// <summary>
    /// Derives the layout of a native index whose concrete type is unknown to the binding (factory, deserialization).
    /// Only id-keyed types need detecting; every other type is positional, and <see cref="Positions"/> is safe for all
    /// of them because the range check makes faiss's own check redundant rather than wrong.
    /// </summary>
    public static KeyLayout OfNative(IntPtr index)
    {
        while (true)
        {
            // IndexIDMap2 derives from IndexIDMap, so one cast covers both.
            if (Native.faiss_IndexIDMap_cast(index) != IntPtr.Zero)
                return Ids;

            if (Native.faiss_IndexIVF_cast(index) != IntPtr.Zero)
                return CheckedIds;

            // IndexPreTransform passes keys and reconstruct_n straight to its sub-index.
            if (Native.faiss_IndexPreTransform_cast(index) == IntPtr.Zero)
                return Positions;

            index = Native.faiss_IndexPreTransform_index_ptr(index);
        }
    }
}
