FROM alpine:3.24 AS faiss-builder

RUN apk add --no-cache \
    cmake \
    g++ \
    make \
    openblas-dev \
    lapack-dev \
    git

WORKDIR /build

COPY faiss ./faiss
COPY patches/faiss ./patches/faiss
RUN git apply --ignore-whitespace --directory=faiss patches/faiss/*.patch

WORKDIR /build/faiss
RUN cmake -B build . \
    -DFAISS_ENABLE_C_API=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_OPT_LEVEL=dd \
    -DFAISS_USE_LTO=ON \
    -DFAISS_ENABLE_MKL=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DFAISS_ENABLE_EXTRAS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_INSTALL_RPATH='$ORIGIN'

RUN make -C build -j $(nproc) faiss_c

FROM mcr.microsoft.com/dotnet/sdk:11.0-alpine-aot AS build
ARG TARGETARCH
WORKDIR /server

COPY src/FaissNET.Linux/Faiss.Linux.csproj src/FaissNET.Linux/
COPY src/FaissNET.Native/Faiss.Native.csproj src/FaissNET.Native/

COPY src/FaissNET.Core/Faiss.csproj src/FaissNET.Core/
COPY src/FaissNET.Interop/Faiss.Interop.csproj src/FaissNET.Interop/
COPY src/FaissNET.Cpu/Faiss.Cpu.csproj src/FaissNET.Cpu/

COPY src/FaissNET.Server/Faiss.Server.csproj src/FaissNET.Server/
RUN dotnet restore src/FaissNET.Server/Faiss.Server.csproj -a $TARGETARCH

COPY src/FaissNET.Linux/ src/FaissNET.Linux/
COPY src/FaissNET.Native/ src/FaissNET.Native/

COPY src/FaissNET.Core/ src/FaissNET.Core/
COPY src/FaissNET.Interop/ src/FaissNET.Interop/
COPY src/FaissNET.Cpu/ src/FaissNET.Cpu/

COPY src/FaissNET.Server/ src/FaissNET.Server/

RUN dotnet publish src/FaissNET.Server/Faiss.Server.csproj -c Release -a $TARGETARCH -p:FaissRuntime=Linux -p:FaissBackend=Cpu -p:ContinuousIntegrationBuild=true --no-restore -o /server/app

FROM mcr.microsoft.com/dotnet/runtime-deps:11.0-alpine AS final

RUN apk add --no-cache libgomp openblas

WORKDIR /app
COPY --from=build /server/app .
COPY --from=faiss-builder /build/faiss/build/c_api/libfaiss_c.so /build/faiss/build/faiss/libfaiss.so ./

RUN mkdir -p /app/data && chown $APP_UID /app/data
VOLUME /app/data

USER $APP_UID
EXPOSE 8080
ENTRYPOINT ["./FaissNET.Server"]