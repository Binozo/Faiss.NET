namespace Faiss.Tests.Infrastructure;

/// <summary>
/// The faiss' <c>c_api</c> is missing some c apis that are required for full functionality.
/// </summary>
public static class NativeSupport
{
    /// <summary><c>faiss_IndexReplicas_set_own_indices</c> in pending PR #5511 <see href="https://github.com/facebookresearch/faiss/pull/5511" />  needed by the <see cref="Faiss.Cpu.Indexes.Replicas.IndexReplicas{T}"/> constructor.</summary>
    public const string ReplicasOwnIndices = "Requires faiss_IndexReplicas_set_own_indices, PR #5511 not merged yet.";

    /// <summary><c>faiss_IndexRefineFlat_base_index</c> in pending PR #5514 <see href="https://github.com/facebookresearch/faiss/pull/5514" /> needed by the <see cref="Faiss.Cpu.Indexes.Refine.IndexRefineFlat{T}"/> constructor.</summary>
    public const string RefineFlatBaseIndex = "Requires faiss_IndexRefineFlat_base_index, PR #5514 not merged yet.";
}
