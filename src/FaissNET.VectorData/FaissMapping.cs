using System;
using Faiss.Cpu.Factory;
using Faiss.Cpu.Indexes.Approximate;
using Faiss.Cpu.Indexes.Binary;
using Faiss.Cpu.Indexes.Factory;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Indexes.Mapped;
using Faiss.Cpu.Indexes.Transform;
using Faiss.Cpu.Transforms;
using Faiss.Interfaces;
using Faiss.Models;
using Microsoft.Extensions.VectorData;
using Microsoft.Extensions.VectorData.ProviderServices;

namespace FaissNet.VectorData;

internal static class FaissMapping
{
    const int Nlist = 24;

    public static IIndex MapIndex(VectorPropertyModel vector) => vector.DistanceFunction switch
    {
        DistanceFunction.DotProductSimilarity => vector.IndexKind switch
        {
            null or IndexKind.Flat => new IndexIDMap2<IndexFlatIP>(new IndexFlatIP(vector.Dimensions), takeOwnership: true),
            IndexKind.Hnsw => new IndexIDMap2<IndexHNSW>(new IndexHNSW(vector.Dimensions, metricType: MetricType.InnerProduct), takeOwnership: true),
            IndexKind.IvfFlat => new IndexIDMap2<IndexIVFFlat<IndexFlatIP>>(new IndexIVFFlat<IndexFlatIP>(new IndexFlatIP(vector.Dimensions), nlist: Nlist, ownQuantizer: true), takeOwnership: true),
            IndexKind.QuantizedFlat => new IndexIDMap2<IndexScalarQuantizer>(new IndexScalarQuantizer(vector.Dimensions, qt: QuantizerType.QT_fp16, metric: MetricType.InnerProduct), takeOwnership: true),
            _ => throw new NotSupportedException($"Index kind '{vector.IndexKind}' is not supported."),
        },
        DistanceFunction.EuclideanSquaredDistance or DistanceFunction.EuclideanDistance => vector.IndexKind switch
        {
            null or IndexKind.Flat => new IndexIDMap2<IndexFlatL2>(new IndexFlatL2(vector.Dimensions), takeOwnership: true),
            IndexKind.Hnsw => new IndexIDMap2<IndexHNSW>(new IndexHNSW(vector.Dimensions, metricType: MetricType.L2), takeOwnership: true),
            IndexKind.IvfFlat => new IndexIDMap2<IndexIVFFlat<IndexFlatL2>>(new IndexIVFFlat<IndexFlatL2>(new IndexFlatL2(vector.Dimensions), nlist: Nlist, ownQuantizer: true), takeOwnership: true),
            IndexKind.QuantizedFlat => new IndexIDMap2<IndexScalarQuantizer>(new IndexScalarQuantizer(vector.Dimensions, qt: QuantizerType.QT_fp16, metric: MetricType.L2), takeOwnership: true),
            _ => throw new NotSupportedException($"Index kind '{vector.IndexKind}' is not supported."),
        },
        DistanceFunction.CosineSimilarity or DistanceFunction.CosineDistance => vector.IndexKind switch
        {
            null or IndexKind.Flat => new IndexIDMap2<IndexPreTransform<IndexFlatIP>>(new IndexPreTransform<IndexFlatIP>(new IndexFlatIP(vector.Dimensions), new NormalizationTransform(vector.Dimensions), ownSubIndex: true), takeOwnership: true),
            IndexKind.Hnsw => new IndexIDMap2<IndexPreTransform<IndexHNSW>>(new IndexPreTransform<IndexHNSW>(new IndexHNSW(vector.Dimensions, metricType: MetricType.InnerProduct), new NormalizationTransform(vector.Dimensions), ownSubIndex: true), takeOwnership: true),
            IndexKind.IvfFlat => new IndexIDMap2<IndexPreTransform<IndexIVFFlat<IndexFlatIP>>>(new IndexPreTransform<IndexIVFFlat<IndexFlatIP>>(new IndexIVFFlat<IndexFlatIP>(new IndexFlatIP(vector.Dimensions), nlist: Nlist, ownQuantizer: true), new NormalizationTransform(vector.Dimensions), ownSubIndex: true), takeOwnership: true),
            IndexKind.QuantizedFlat => new IndexIDMap2<IndexPreTransform<IndexScalarQuantizer>>(new IndexPreTransform<IndexScalarQuantizer>(new IndexScalarQuantizer(vector.Dimensions, qt: QuantizerType.QT_fp16, metric: MetricType.InnerProduct), new NormalizationTransform(vector.Dimensions), ownSubIndex: true), takeOwnership: true),
            _ => throw new NotSupportedException($"Index kind '{vector.IndexKind}' is not supported."),
        },
        DistanceFunction.NegativeDotProductSimilarity => new IndexIDMap2<IndexFlatIP>(new IndexFlatIP(vector.Dimensions), takeOwnership: true),
        DistanceFunction.ManhattanDistance => vector.IndexKind switch
        {
            null or IndexKind.Flat => new IndexIDMap2<IndexFlat>(new IndexFlat(vector.Dimensions, MetricType.L1), takeOwnership: true),
            IndexKind.Hnsw => new IndexIDMap2<IndexHNSWFlat>(new IndexHNSWFlat(vector.Dimensions, metricType: MetricType.L1), takeOwnership: true),
            _ => throw new NotSupportedException($"Index kind '{vector.IndexKind}' is not supported."),
        },
        DistanceFunction.HammingDistance => vector.IndexKind switch
        {
            null or IndexKind.Flat => new IndexBinaryIDMap2<IndexBinaryFlat>("BFlat", vector.Dimensions * 8),
            IndexKind.Hnsw => new IndexBinaryIDMap2<IndexBinaryHNSW>("BHNSW32", vector.Dimensions * 8),
            IndexKind.IvfFlat => new IndexBinaryIDMap2<IndexBinaryIVF>($"BIVF{Nlist}",  vector.Dimensions * 8),
            _ => throw new NotSupportedException($"Index kind '{vector.IndexKind}' is not supported."),
        },
        null => new IndexIDMap2<IndexFlatL2>(new IndexFlatL2(vector.Dimensions), takeOwnership: true),
        _ => new IndexIDMap2<GenericFloatIndex>(IndexFactory.Create<GenericFloatIndex>(vector.DistanceFunction, vector.Dimensions), takeOwnership: true),
    };

    public static float PreProcessIndexSearchResult(string? distanceFunction, float distance) => distanceFunction switch
    {
        DistanceFunction.EuclideanDistance => MathF.Sqrt(MathF.Max(distance, 0f)),
        DistanceFunction.CosineDistance => 1f - distance,
        DistanceFunction.NegativeDotProductSimilarity => -distance,
        _ => distance,
    };
}