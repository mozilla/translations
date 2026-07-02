#pragma once

#include "common/shape.h"
#include "common/types.h"
#include "graph/chainable.h"

#include <cstdint>
#include <fstream>
#include <string>

namespace marian {

// Reference-trace recorder: writes a record for every node's freshly-computed
// value in forward-execution order, producing a complete oracle of the graph's
// intermediate tensors for parity testing against the Rust reimplementation.
//
// Enabled only when the MARIAN_TRACE environment variable names an output path;
// otherwise every entry point is a cheap no-op, so normal runs are untouched.
//
// See inference-rs/build-plan.md (step 1).
//
// Two files are written:
//
//   <MARIAN_TRACE>       the binary trace consumed by the Rust reader
//   <MARIAN_TRACE>.txt   a human-readable manifest: one line per node with its
//                        id, op type, name, dtype and shape (no tensor data),
//                        for eyeballing the graph without a parser.
//
// Binary file layout (little-endian, native):
//
//   header : magic "MTRC" (4 bytes) | version:u32
//   record : id:u64
//            type:      len:u32 + utf8 bytes
//            name:      len:u32 + utf8 bytes
//            value_type:u64                     (marian::Type enum value)
//            shape:     rank:u32 + dims:i32[rank]
//            children:  count:u32 + child_ids:u64[count]
//            data:      byte_len:u64 + raw_bytes[byte_len]
//
// Records are appended in execution order; the file is a flat stream with no
// index. byte_len is the logical size (shape.elements() * sizeOf(type)), which
// excludes the tensor allocator's 256-byte alignment padding.
class TraceRecorder {
public:
  // Process-wide singleton. Constructed lazily; reads MARIAN_TRACE once.
  static TraceRecorder& instance();

  bool enabled() const { return enabled_; }

  // Append one record for the just-computed value of `node`. No-op when
  // disabled. Safe to call for every node in ExpressionGraph::forward().
  void record(const Expr& node);

private:
  TraceRecorder();

  void writeString(const std::string& s);
  template <typename T>
  void writePod(const T& v) {
    out_.write(reinterpret_cast<const char*>(&v), sizeof(T));
  }

  bool enabled_{false};
  size_t index_{0};  // running node counter, shown in the manifest
  std::ofstream out_;      // binary trace
  std::ofstream manifest_;  // human-readable sidecar
};

}  // namespace marian
