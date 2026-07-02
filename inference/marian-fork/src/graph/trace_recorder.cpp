#include "graph/trace_recorder.h"

#include "common/logging.h"
#include "tensors/tensor.h"

#include <cstdlib>
#include <sstream>

namespace marian {

static const uint32_t TRACE_VERSION = 1;

TraceRecorder& TraceRecorder::instance() {
  static TraceRecorder recorder;
  return recorder;
}

TraceRecorder::TraceRecorder() {
  const char* path = std::getenv("MARIAN_TRACE");
  if(!path || path[0] == '\0')
    return;

  out_.open(path, std::ios::binary | std::ios::trunc);
  ABORT_IF(!out_, "MARIAN_TRACE set but trace file '{}' could not be opened", path);

  const std::string manifestPath = std::string(path) + ".txt";
  manifest_.open(manifestPath, std::ios::trunc);
  ABORT_IF(!manifest_, "trace manifest '{}' could not be opened", manifestPath);

  out_.write("MTRC", 4);
  writePod(TRACE_VERSION);

  manifest_ << "# Reference trace manifest for " << path << "\n"
            << "# One line per node in forward-execution order. Tensor data is\n"
            << "# in the binary trace, not here; only shapes are described.\n"
            << "# columns: index  id  op  dtype  shape  bytes  children  name\n";

  enabled_ = true;
  LOG(info, "Reference-trace recording enabled -> {}", path);
}

void TraceRecorder::writeString(const std::string& s) {
  writePod(static_cast<uint32_t>(s.size()));
  if(!s.empty())
    out_.write(s.data(), static_cast<std::streamsize>(s.size()));
}

void TraceRecorder::record(const Expr& node) {
  if(!enabled_)
    return;

  Tensor val = node->val();
  if(!val)
    return;  // node produced no value (should not happen post-forward)

  // The oracle runs natively on CPU, so the tensor memory is host-readable and
  // no device copy is needed. Guard rather than silently emit garbage if this
  // is ever run on a non-CPU backend.
  ABORT_IF(val->getDeviceId().type != DeviceType::cpu,
           "Reference-trace recording only supports CPU tensors (node id {})",
           node->getId());

  const Type type = val->type();
  const Shape& shape = val->shape();
  const size_t byteLen = shape.elements() * sizeOf(type);

  writePod(static_cast<uint64_t>(node->getId()));
  writeString(node->type());
  writeString(node->name());
  writePod(static_cast<uint64_t>(type));

  writePod(static_cast<uint32_t>(shape.size()));
  for(size_t i = 0; i < shape.size(); ++i)
    writePod(static_cast<int32_t>(shape[static_cast<int>(i)]));

  const std::vector<Expr>& children = node->children();
  writePod(static_cast<uint32_t>(children.size()));
  for(const auto& child : children)
    writePod(static_cast<uint64_t>(child->getId()));

  writePod(static_cast<uint64_t>(byteLen));
  if(byteLen > 0)
    out_.write(reinterpret_cast<const char*>(val->memory()->data()),
               static_cast<std::streamsize>(byteLen));

  // Human-readable manifest line (data-free).
  std::ostringstream dims;
  dims << "[";
  for(size_t i = 0; i < shape.size(); ++i)
    dims << (i ? "," : "") << shape[static_cast<int>(i)];
  dims << "]";

  std::ostringstream kids;
  kids << "[";
  for(size_t i = 0; i < children.size(); ++i)
    kids << (i ? "," : "") << children[i]->getId();
  kids << "]";

  manifest_ << index_ << "  id=" << node->getId() << "  op=" << node->type()
            << "  dtype=" << type << "  shape=" << dims.str() << "  bytes=" << byteLen
            << "  children=" << kids.str();
  if(!node->name().empty())
    manifest_ << "  name=" << node->name();
  manifest_ << "\n";

  ++index_;
}

}  // namespace marian
