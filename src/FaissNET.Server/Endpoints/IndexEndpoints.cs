using Faiss.Server.Models;
using Faiss.Server.Services;
using Microsoft.AspNetCore.Http.HttpResults;

namespace Faiss.Server.Endpoints;

public static class IndexEndpoints
{
    public static void MapIndexEndpoints(this IEndpointRouteBuilder endpoints)
    {
        var group = endpoints.MapGroup("/api/v1/indexes").WithTags("Indexes");
        
        group.MapGet("/", ListIndexes);
        group.MapPost("/", CreateIndex);
    }
    
    static Ok<IReadOnlyList<IndexInfo>> ListIndexes(IndexManager manager) 
        => TypedResults.Ok(manager.List());

    static Ok<IndexInfo> CreateIndex(CreateIndexRequest createIndexRequest, IndexManager manager, CancellationToken cancellationToken)
        => TypedResults.Ok(manager.Create(createIndexRequest.Description, createIndexRequest.Dimension));
}