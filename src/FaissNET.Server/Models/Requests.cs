using System.ComponentModel.DataAnnotations;

namespace Faiss.Server.Models;

public sealed record CreateIndexRequest
{
    [Required, StringLength(256)]
    public string Description { get; init; } = "";

    [Required, Range(1, 2048)]
    public int Dimension { get; init; } = 0;
}