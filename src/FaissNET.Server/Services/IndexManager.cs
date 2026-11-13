using System.Collections.Concurrent;
using Faiss.Cpu.Factory;
using Faiss.Cpu.Indexes.Factory;
using Faiss.Cpu.Interfaces;
using Faiss.Interfaces;
using Faiss.Server.Models;
using ITrainableIndex = Faiss.Interfaces.ITrainableIndex;

namespace Faiss.Server.Services;

record IndexEntry : IDisposable
{
    public readonly ReaderWriterLockSlim LockSlim = new();
    public readonly IIndex Index;

    public DateTimeOffset CreatedAt;
    public DateTimeOffset UpdatedAt;

    public IndexEntry(IIndex index)
    {
        Index = index;

        CreatedAt = DateTimeOffset.UtcNow;
        UpdatedAt = DateTimeOffset.UtcNow;
    }

    public void Dispose() => Index.Dispose();
}

public class IndexManager : IDisposable
{
    private readonly ConcurrentDictionary<Guid, IndexEntry> _indexes = new();

    public IReadOnlyList<IndexInfo> List()
    {
        var indexes = _indexes.ToArray();
        var infos = new IndexInfo[indexes.Length];

        for (int i = 0; i < indexes.Length; i++)
        {
            var index = indexes[i].Value;
            index.LockSlim.EnterReadLock();

            try
            {
                infos[i] = new IndexInfo
                {
                    Id = indexes[i].Key,
                    FactoryString = index.Index switch
                    {
                        IFactoryDescribableFloatIndex describableFloatIndex => describableFloatIndex.GetFactoryDescription(),
                        IFactoryDescribableBinaryIndex describableBinaryIndex => describableBinaryIndex.GetFactoryDescription(),
                        _ => null
                    },
                    Dimension = index.Index.Dimensions,
                    MetricType = index.Index.Metric,
                    NTotal = index.Index.TotalCount,
                    IsTrained = index.Index is ITrainableIndex { IsTrained: true },
                    CreatedAt = index.CreatedAt,
                    UpdatedAt = index.UpdatedAt,
                };
            }
            finally
            {
                index.LockSlim.ExitReadLock();
            }
        }
        
        return infos;
    }

    public IndexInfo Create(string description, int dimension)
    {
        Guid guid = Guid.NewGuid();
        var newIndex = IndexFactory.Create<GenericFloatIndex>(description, dimension);

        try
        {
            var indexEntry = new IndexEntry(newIndex);
            var metric = newIndex.Metric;
            var isTrained = newIndex.IsTrained;

            while (!_indexes.TryAdd(guid, indexEntry))
            {
                guid = Guid.NewGuid();
            }

            return new IndexInfo
            {
                Id = guid,
                FactoryString = description,
                Dimension = dimension,
                MetricType = metric,
                NTotal = 0,
                IsTrained = isTrained,
                CreatedAt = indexEntry.CreatedAt,
                UpdatedAt = indexEntry.UpdatedAt
            };
        }
        catch
        {
            newIndex.Dispose();
            throw;
        }
    }

    public void Dispose() => _indexes.Values.ToList().ForEach(index => index.Dispose());
}