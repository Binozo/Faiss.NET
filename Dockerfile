FROM ubuntu:24.04 AS faiss-builder

RUN --mount=type=cache,id=apt,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,id=apt,sharing=locked,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    make \
    libopenblas-dev \
    liblapack-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY faiss ./faiss

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

FROM mcr.microsoft.com/dotnet/sdk:9.0-noble AS test

ARG TARGETARCH

RUN --mount=type=cache,id=apt,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,id=apt,sharing=locked,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libgomp1 \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src ./src
COPY Faiss.NET.slnx .
COPY README.md .
COPY LICENSE .

COPY --from=faiss-builder /build/faiss/build/c_api/libfaiss_c.so /tmp/
COPY --from=faiss-builder /build/faiss/build/faiss/libfaiss.so /tmp/

RUN set -eux; \
    case ${TARGETARCH} in \
      amd64) RID=x64 ;; \
      arm64) RID=arm64 ;; \
      *) echo "Unsupported architecture: ${TARGETARCH}" && exit 1 ;; \
    esac && \
    mkdir -p /app/src/FaissNET.Linux/runtimes/linux-${RID}/native/ && \
    cp /tmp/libfaiss_c.so /tmp/libfaiss.so /app/src/FaissNET.Linux/runtimes/linux-${RID}/native/

CMD ["dotnet", "test", "src/FaissNET.Tests/Faiss.Tests.csproj", "-c", "Release"]