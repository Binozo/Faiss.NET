using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Faiss.Cpu.Indexes.Factory;
using Faiss.Cpu.Indexes.Mapped;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Selectors;
using Faiss.Exceptions;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;
using Microsoft.Extensions.VectorData.ProviderServices;

namespace FaissNet.VectorData;

public sealed class FaissCollection<TKey, TRecord> : VectorStoreCollection<TKey, TRecord>
    where TKey : notnull
    where TRecord : class
{
    private readonly FaissVectorStore _store;
    private readonly CollectionModel _model;
    private readonly FaissCollectionOptions _options;
    private readonly VectorStoreCollectionMetadata _metadata;

    public override string Name { get; }

    public FaissCollection(FaissVectorStore store, string name, FaissCollectionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);

        _store = store;
        _options = options ?? new FaissCollectionOptions();
        Name = name;

        IEmbeddingGenerator? generator = options?.EmbeddingGenerator ?? store.DefaultEmbeddingGenerator;
        _model = typeof(TRecord) == typeof(Dictionary<string, object?>)
            ? new FaissModelBuilder().BuildDynamic(options?.Definition ?? throw new ArgumentException("A dynamic collection requires a definition.", nameof(options)), generator)
            : new FaissModelBuilder().Build(typeof(TRecord), typeof(TKey), options?.Definition, generator);

        _metadata = new VectorStoreCollectionMetadata
        {
            VectorStoreSystemName = "faiss",
            CollectionName = name,
        };
    }

    public override Task<bool> CollectionExistsAsync(CancellationToken cancellationToken = default)
        => Task.FromResult(_store.TryGetState(Name, out _));

    public override Task EnsureCollectionExistsAsync(CancellationToken cancellationToken = default)
    {
        var vector = _model.VectorProperty;

        var state = _store.GetOrAddState(Name, () =>
        {
            try
            {
                var mappedIndex = FaissMapping.MapIndex(vector);

                if (mappedIndex is IndexIDMap2<GenericFloatIndex> || vector.IndexKind == IndexKind.Hnsw)
                {
                    return new AppendOnlyFloatFaissCollectionState(mappedIndex as IIDMappedFloatIndex)
                    {
                        IndexKind = vector.IndexKind ?? FaissMapping.DefaultIndexKind,
                        DistanceFunction = vector.DistanceFunction ?? FaissMapping.DefaultDistanceFunction
                    };
                }

                if (vector.DistanceFunction is DistanceFunction.HammingDistance)
                {
                    // Binary
                    throw new NotImplementedException();
                }

                return new FloatFaissCollectionState(mappedIndex as IIDMappedFloatIndex)
                {
                    IndexKind = vector.IndexKind ?? FaissMapping.DefaultIndexKind,
                    DistanceFunction = vector.DistanceFunction ?? FaissMapping.DefaultDistanceFunction
                };
            }
            catch (FaissException exception)
            {
                throw Wrap(exception, "EnsureCollectionExists");
            }
        });

        if (state.Index.Dimensions != vector.Dimensions)
            throw new VectorStoreException($"Collection '{Name}' already exists with {state.Index} dimensions, but this record type declares {vector.Dimensions}.");

        return Task.CompletedTask;
    }

    public override Task EnsureCollectionDeletedAsync(CancellationToken cancellationToken = default)
    {
        _store.RemoveState(Name);
        return Task.CompletedTask;
    }

    public override Task UpsertAsync(TRecord record, CancellationToken cancellationToken = default)
        => UpsertAsync([record], cancellationToken);

    public override async Task UpsertAsync(IEnumerable<TRecord> records, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(records);

        var state = GetState();

        foreach (var batch in records.Chunk(_options.BatchSize))
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            float[][] vectors = await GetVectorsAsync(batch, state, cancellationToken).ConfigureAwait(false);
            long[] ids = new long[vectors.Length];
            HashSet<long> toBeDeletedIds = new HashSet<long>();
            
            state.Lock.EnterWriteLock();
            try
            {
                for (int i = 0; i < batch.Length; i++)
                {
                    TRecord record = batch[i];
                    object key = GetOrGenerateKey(record);

                    if (state.KeyToId.TryGetValue(key, out long existingId))
                    {
                        toBeDeletedIds.Add(existingId);
                    }
                    
                    long id = state.NextId++;
                    ids[i] = id;
                }

                if (state is not AppendOnlyFloatFaissCollectionState && toBeDeletedIds.Count > 0 && state.Index is IIDRemovableFloatIndex removableFloatIndex)
                {
                    using var removeSelector = new IDSelectorBatch(toBeDeletedIds.ToArray());
                    long numRemoved = removableFloatIndex.RemoveIds(removeSelector);
                    // Throw if numRemoved < toBeDeletedIds.Count? Or just ignore?
                }
                // TODO: Also support binary

                foreach (var beDeletedId in toBeDeletedIds)
                {
                    state.Records.Remove(beDeletedId);
                }
                
                var buffer = ArrayPool<float>.Shared.Rent(vectors.Length * state.Index.Dimensions);
                try
                {
                    for (int i = 0; i < vectors.Length; i++)
                        Array.Copy(vectors[i], 0, buffer, i, state.Index.Dimensions);
                    
                    if (state.Index is IIDMappedFloatIndex mappedFloatIndex)
                        mappedFloatIndex.Add(vectors.Length, buffer, ids);
                    else
                        throw new NotSupportedException(); // TODO: Also support binary
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(buffer);
                }
                
                
                for (int i = 0; i < batch.Length; i++)
                {
                    TRecord record = batch[i];
                    object key = GetOrGenerateKey(record);

                    long id = ids[i];
                    state.KeyToId[key] = id;

                    var data = new Dictionary<string, object?>();
                    foreach (var property in _model.DataProperties)
                        data[property.StorageName] = property.GetValueAsObject(record);
                    state.Records[id] = new StoredRecord { Key = key, Data = data };
                }
            }
            catch (FaissException exception)
            {
                throw Wrap(exception, "Upsert");
            }
            finally
            {
                state.Lock.ExitWriteLock();
            }
        }
    }

    private async Task<float[][]> GetVectorsAsync(IReadOnlyList<TRecord> records, FaissCollectionState state, CancellationToken cancellationToken)
    {
        var vectorProperty = _model.VectorProperty;
        var vectors = new float[records.Count][];

        if (!NeedsGeneration(vectorProperty))
        {
            for (int i = 0; i < records.Count; i++)
                vectors[i] = ToFaissVector(_model.VectorProperty.GetValueAsObject(records[i]), state);
            return vectors;
        }

        IReadOnlyList<Embedding> embeddings = await vectorProperty
            .GenerateEmbeddingsAsync(records.Select(record => vectorProperty.GetValueAsObject(record)), cancellationToken)
            .ConfigureAwait(false);

        for (int i = 0; i < records.Count; i++)
            vectors[i] = ToFaissVector(embeddings[i], state);

        return vectors;
    }

    private static bool NeedsGeneration(VectorPropertyModel vectorProperty)
        => vectorProperty.Type != vectorProperty.EmbeddingType;
    public override Task DeleteAsync(TKey key, CancellationToken cancellationToken = default)
    {
        var state = GetState();

        state.Lock.EnterWriteLock();
        try
        {
            if (state.KeyToId.Remove(key, out long id))
            {
                state.Records.Remove(id);

                if (state is AppendOnlyFloatFaissCollectionState appendOnlyFloatFaissCollectionState)
                {
                    appendOnlyFloatFaissCollectionState.DeletedRecords.Add(id);
                }
                else
                {
                    using var selector = new IDSelectorBatch([id]);
                    if(state.Index is IIDRemovableFloatIndex index)
                        index.RemoveIds(selector);
                    else
                        throw new NotSupportedException("Cannot remove ids from index");
                }

            }
        }
        catch (FaissException exception)
        {
            throw Wrap(exception, "Delete");
        }
        finally
        {
            state.Lock.ExitWriteLock();
        }

        return Task.CompletedTask;
    }

    public override async IAsyncEnumerable<VectorSearchResult<TRecord>> SearchAsync<TInput>(TInput searchValue, int top, VectorSearchOptions<TRecord>? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(top, 1);

        var state = GetState();

        options ??= new();
        _ = _model.GetVectorPropertyOrSingle(options);

        Func<TRecord, bool>? predicate = options.Filter?.Compile();
        object queryVector = searchValue;
        
        if (searchValue is not (ReadOnlyMemory<float> or float[] or Embedding<float>))
        {
            var vectorProperty = _model.GetVectorPropertyOrSingle(options);

            if (vectorProperty.EmbeddingGenerationDispatcher is null)
                throw new NotSupportedException($"Cannot search with a value of type '{typeof(TInput).Name}': no embedding generator is configured. Pass a float vector, or configure an IEmbeddingGenerator.");

            queryVector = await vectorProperty.GenerateEmbeddingAsync(searchValue, cancellationToken).ConfigureAwait(false);
        }

        float[] query = ToFaissVector(queryVector, state);

        bool higherIsBetter = state.DistanceFunction is DistanceFunction.CosineSimilarity or DistanceFunction.DotProductSimilarity;

        List<VectorSearchResult<TRecord>> result = new(capacity: top);
        
        state.Lock.EnterReadLock();
        try
        {
            HashSet<long>? allowedIds = null;
            if (predicate is not null)
            {
                allowedIds = new HashSet<long>();
                foreach (long id in state.Records.Keys)
                {
                    if(predicate(ToRecord(state, id, includeVectors: false)))
                        allowedIds.Add(id);
                }
            }
            
            HashSet<long>? disallowedIds = null;
            if (state is AppendOnlyFloatFaissCollectionState appendOnlyFloatFaissCollectionState)
                disallowedIds = appendOnlyFloatFaissCollectionState.DeletedRecords;

            int count = top + options.Skip;
            List<(long, float)> hits = new(top); // Label, Distance/Score

            if (state.Index is IParamsFloatSearchIndex paramsFloatSearchIndex)
            {
                Span<float> distances = stackalloc float[count];
                Span<long> labels = stackalloc long[count];

                if (allowedIds is not null && disallowedIds is not null)
                {
                    using var allowedBatch = new IDSelectorBatch(allowedIds.ToArray());
                    using var disallowedBatch = new IDSelectorBatch(disallowedIds.ToArray());
                    using var notDisallowed = new IDSelectorNot(disallowedBatch);
                    using var selector = new IDSelectorAnd(allowedBatch, notDisallowed);
                    using var searchParameters = new SearchParameters(selector);
                    paramsFloatSearchIndex.SearchWithParams(1, query, top, searchParameters, distances, labels);
                }
                if (allowedIds is not null)
                {
                    using var allowedBatch = new IDSelectorBatch(allowedIds.ToArray());
                    using var searchParameters = new SearchParameters(allowedBatch);
                    paramsFloatSearchIndex.SearchWithParams(1, query, top, searchParameters, distances, labels);
                }

                if (disallowedIds is not null)
                {
                    using var disallowedBatch = new IDSelectorBatch(disallowedIds.ToArray());
                    using var notDisallowed = new IDSelectorNot(disallowedBatch);
                    using var searchParameters = new SearchParameters(notDisallowed);
                    paramsFloatSearchIndex.SearchWithParams(1, query, top, searchParameters, distances, labels);
                }
                else
                    paramsFloatSearchIndex.Search(1, query, top, distances, labels);

                for (int i = options.Skip; i < labels.Length; i++)
                {
                    float score = FaissMapping.PreProcessIndexSearchResult(state.DistanceFunction, distances[i]);
                    if(labels[i] == -1 || !state.Records.ContainsKey(labels[i]))
                        continue;
                    if(options.ScoreThreshold is not null && (higherIsBetter ? score < options.ScoreThreshold : score > options.ScoreThreshold))
                        continue;

                    hits.Add((labels[i], score));
                }
            }
            else
                throw new NotSupportedException(); // TODO: Add binary support

            foreach (var hit in hits)
                result.Add(new VectorSearchResult<TRecord>(ToRecord(state, hit.Item1, options.IncludeVectors), hit.Item2));
        }
        catch (FaissException exception)
        {
            throw Wrap(exception, "Search");
        }
        finally
        {
            state.Lock.ExitReadLock();
        }

        foreach (var vectorSearchResult in result)
            yield return vectorSearchResult;
    }

    public override Task<TRecord?> GetAsync(TKey key, RecordRetrievalOptions? options = null, CancellationToken cancellationToken = default)
    {
        var state = GetState();

        state.Lock.EnterReadLock();
        try
        {
            TRecord? record = state.KeyToId.TryGetValue(key, out long id)
                ? ToRecord(state, id, options?.IncludeVectors ?? false)
                : null;

            return Task.FromResult(record);
        }
        catch (FaissException exception)
        {
            throw Wrap(exception, "Get");
        }
        finally
        {
            state.Lock.ExitReadLock();
        }
    }
    public override async IAsyncEnumerable<TRecord> GetAsync(Expression<Func<TRecord, bool>> filter, int top, FilteredRecordRetrievalOptions<TRecord>? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(filter);
        ArgumentOutOfRangeException.ThrowIfLessThan(top, 1);

        options ??= new();
        var state = GetState();

        Func<TRecord, bool> predicate = filter.Compile();

        List<TRecord> results;
        
        state.Lock.EnterReadLock();
        try
        {
            IEnumerable<(long Id, TRecord Record)> matches = state.Records.Keys
                .Select(id => (Id: id, Record: ToRecord(state, id, includeVectors: false)))
                .Where(match => predicate(match.Record));

            if (options.OrderBy is not null)
            {
                IOrderedEnumerable<(long Id, TRecord Record)>? ordered = null;

                foreach (var sort in options.OrderBy(new()).Values)
                {
                    Func<TRecord, object?> selector = sort.PropertySelector.Compile();

                    ordered = (ordered, sort.Ascending) switch
                    {
                        (null, true) => matches.OrderBy(match => selector(match.Record)),
                        (null, false) => matches.OrderByDescending(match => selector(match.Record)),
                        (_, true) => ordered.ThenBy(match => selector(match.Record)),
                        (_, false) => ordered.ThenByDescending(match => selector(match.Record)),
                    };
                }

                matches = ordered ?? matches;
            }
            else
                matches = matches.OrderBy(match => match.Id);

            results = matches
                .Skip(options.Skip)
                .Take(top)
                .Select(match => options.IncludeVectors ? ToRecord(state, match.Id, includeVectors: true) : match.Record)
                .ToList();
        }
        catch (FaissException exception)
        {
            throw Wrap(exception, "Get");
        }
        finally
        {
            state.Lock.ExitReadLock();
        }

        foreach (var result in results)
            yield return result;
    }

    private FaissCollectionState GetState()
        => _store.TryGetState(Name, out var state)
            ? state
            : throw new VectorStoreException($"Collection '{Name}' does not exist. Call EnsureCollectionExistsAsync first.");

    private object GetOrGenerateKey(TRecord record)
    {
        var keyProperty = _model.KeyProperty;
        object? key = keyProperty.GetValueAsObject(record);

        bool isUnset = key is null || key is Guid guid && guid == Guid.Empty;

        if (isUnset && keyProperty.IsAutoGenerated && keyProperty.Type == typeof(Guid))
        {
            key = Guid.NewGuid();
            keyProperty.SetValueAsObject(record, key);
        }

        return key ?? throw new ArgumentException("The record's key must not be null.");
    }

    private static float[] ToFaissVector(object? value, FaissCollectionState state)
    {
        ReadOnlyMemory<float> memory = value switch
        {
            ReadOnlyMemory<float> m => m,
            float[] a => a,
            Embedding<float> e => e.Vector,
            null => throw new ArgumentException("The vector must not be null."),
            _ => throw new NotSupportedException($"Unsupported vector type '{value.GetType().Name}'. Pass a float vector."),
        };

        if (memory.Length != state.Index.Dimensions)
            throw new ArgumentException($"Expected a vector with {state.Index.Dimensions} dimensions, got {memory.Length}.");

        return memory.ToArray();
    }

    private TRecord ToRecord(FaissCollectionState state, long id, bool includeVectors)
    {
        StoredRecord stored = state.Records[id];
        TRecord record = _model.CreateRecord<TRecord>();

        _model.KeyProperty.SetValueAsObject(record, stored.Key);

        foreach (var property in _model.DataProperties)
            if (stored.Data.TryGetValue(property.StorageName, out object? value))
                property.SetValueAsObject(record, value);

        if (includeVectors)
        {
            var vectorProperty = _model.VectorProperty;

            if (NeedsGeneration(vectorProperty))
                throw new NotSupportedException("IncludeVectors is not supported when the vector property holds the embedding input (e.g. a string).");

            float[] vector;
            if (state.Index is IReconstructFloatIndex index)
                vector = index.Reconstruct(id);
            else
                throw new NotSupportedException();

            Type type = Nullable.GetUnderlyingType(vectorProperty.Type) ?? vectorProperty.Type;

            object value =
                type == typeof(float[]) ? vector :
                type == typeof(Embedding<float>) ? new Embedding<float>(vector) :
                new ReadOnlyMemory<float>(vector);

            vectorProperty.SetValueAsObject(record, value);
        }

        return record;
    }

    private VectorStoreException Wrap(FaissException exception, string operationName)
        => new("Call to vector store failed.", exception)
        {
            VectorStoreSystemName = _metadata.VectorStoreSystemName,
            CollectionName = Name,
            OperationName = operationName,
        };

    public override object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);

        if (serviceKey is not null) 
            return null;

        if (serviceType == typeof(VectorStoreCollectionMetadata)) 
            return _metadata;
        
        return serviceType.IsInstanceOfType(this) 
            ? this 
            : null;
    }
}