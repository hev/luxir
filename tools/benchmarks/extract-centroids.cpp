// Copyright 2020-2026 Yonik Seeley and Luxir contributors
// SPDX-License-Identifier: Apache-2.0
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>

// Output a uint64 byte count, coarse float32 centroids, then PQ centroids.
// Luxir stores an ordinary FAISS header before its custom list payload.
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::unique_ptr<faiss::Index> index(faiss::read_index(argv[1]));
  auto* ivf = dynamic_cast<faiss::IndexIVF*>(index.get());
  auto* flat = ivf ? dynamic_cast<faiss::IndexFlat*>(ivf->quantizer) : nullptr;
  if (!flat) throw std::runtime_error("Expected IVF with flat coarse quantizer");
  uint64_t bytes = (uint64_t)flat->ntotal * flat->d * sizeof(float);
  std::cout.write((const char*)&bytes, sizeof(bytes));
  std::cout.write((const char*)flat->get_xb(), bytes);
  if (auto* pq = dynamic_cast<faiss::IndexIVFPQ*>(ivf)) {
    std::cout.write((const char*)pq->pq.centroids.data(), pq->pq.centroids.size() * sizeof(float));
  }
  return std::cout ? 0 : 1;
}
