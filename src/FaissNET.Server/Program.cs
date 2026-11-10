using System.Text.Json;
using System.Text.Json.Serialization;
using Faiss.Models;
using Faiss.Server;
using Faiss.Server.Endpoints;
using Faiss.Server.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddProblemDetails();

// Add services to the container.
// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();

builder.Services.AddSingleton<IndexManager>();

builder.Services.ConfigureHttpJsonOptions(o =>
{
    o.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower;
    o.SerializerOptions.DictionaryKeyPolicy  = JsonNamingPolicy.SnakeCaseLower;
    o.SerializerOptions.Converters.Add(new JsonStringEnumConverter<MetricType>(JsonNamingPolicy.SnakeCaseLower));
    o.SerializerOptions.TypeInfoResolverChain.Insert(0, AppJsonContext.Default);
});

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseDeveloperExceptionPage();
    app.MapOpenApi();
}
else
{
    app.UseExceptionHandler();
}

app.MapIndexEndpoints();

app.Run();