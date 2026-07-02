//! Reader for the reference-trace format written by the C++ engine's
//! `TraceRecorder` (see `inference/marian-fork/src/graph/trace_recorder.h`).
//!
//! A trace is a flat stream of node records in forward-execution order — the
//! oracle the Rust ops are validated against (build-plan.md, step 2). This
//! module parses that stream into [`TraceRecord`]s and exposes them as
//! per-node fixtures, including typed views of the raw tensor bytes and a
//! resolver from a node to the records that fed it.
//!
//! Binary layout (little-endian):
//!
//! ```text
//! header : magic "MTRC" (4 bytes) | version:u32
//! record : id:u64
//!          type:      len:u32 + utf8 bytes
//!          name:      len:u32 + utf8 bytes
//!          value_type:u64                     (marian::Type enum value)
//!          shape:     rank:u32 + dims:i32[rank]
//!          children:  count:u32 + child_ids:u64[count]
//!          data:      byte_len:u64 + raw_bytes[byte_len]
//! ```

use std::fmt;
use std::path::Path;

/// Magic bytes at the start of every trace file.
const MAGIC: &[u8; 4] = b"MTRC";
/// Trace format version this reader understands.
const SUPPORTED_VERSION: u32 = 1;

/// Tensor element type, mirroring the values of marian's `enum class Type`
/// (`inference/marian-fork/src/common/types.h`). The raw value encodes both a
/// class (float/signed/unsigned/intgemm) and the element size in its low byte.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DType {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    Float64,
    /// intgemm-quantized int8 weights (architecture-agnostic layout).
    Intgemm8,
    /// intgemm-quantized int16 weights.
    Intgemm16,
}

impl DType {
    /// Decode a marian `Type` enum value. Errors on values this reader does not
    /// model (e.g. the FBGEMM packed types, which never appear on the CPU
    /// gemmology path we trace).
    pub fn from_raw(raw: u64) -> Result<DType, TraceError> {
        Ok(match raw {
            0x101 => DType::Int8,
            0x102 => DType::Int16,
            0x104 => DType::Int32,
            0x108 => DType::Int64,
            0x201 => DType::UInt8,
            0x202 => DType::UInt16,
            0x204 => DType::UInt32,
            0x208 => DType::UInt64,
            0x402 => DType::Float16,
            0x404 => DType::Float32,
            0x408 => DType::Float64,
            0x4101 => DType::Intgemm8,
            0x4102 => DType::Intgemm16,
            other => return Err(TraceError::UnknownDType(other)),
        })
    }

    /// Size of one element in bytes (the low byte of the marian type value).
    pub fn element_size(self) -> usize {
        match self {
            DType::Int8 | DType::UInt8 | DType::Intgemm8 => 1,
            DType::Int16 | DType::UInt16 | DType::Float16 | DType::Intgemm16 => 2,
            DType::Int32 | DType::UInt32 | DType::Float32 => 4,
            DType::Int64 | DType::UInt64 | DType::Float64 => 8,
        }
    }

    /// Short human name matching the manifest / marian's `operator<<`.
    pub fn name(self) -> &'static str {
        match self {
            DType::Int8 => "int8",
            DType::Int16 => "int16",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::UInt8 => "uint8",
            DType::UInt16 => "uint16",
            DType::UInt32 => "uint32",
            DType::UInt64 => "uint64",
            DType::Float16 => "float16",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Intgemm8 => "intgemm8",
            DType::Intgemm16 => "intgemm16",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// One node's recorded value: its identity, shape, and raw tensor bytes.
#[derive(Clone, Debug)]
pub struct TraceRecord {
    /// Node id within the graph it was recorded from. Ids are reused across the
    /// encoder/decoder forward passes, so an id is unique only within a pass —
    /// use execution order (record position) to disambiguate.
    pub id: u64,
    /// Op type, e.g. `"intgemmAffine"`, `"layer_normalization"`, `"param"`.
    pub op_type: String,
    /// Node name (often `"none"` for intermediates; set for parameters).
    pub name: String,
    /// Element type of the recorded tensor.
    pub dtype: DType,
    /// Tensor shape, most-significant dimension first (marian `Shape`).
    pub shape: Vec<i32>,
    /// Node ids of this node's inputs, in argument order.
    pub children: Vec<u64>,
    /// Raw little-endian tensor bytes, `num_elements * dtype.element_size()`.
    pub data: Vec<u8>,
}

impl TraceRecord {
    /// Number of tensor elements (product of the shape dimensions).
    pub fn num_elements(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    /// View the raw bytes as `f32`. Errors unless the dtype is `Float32`.
    pub fn to_f32(&self) -> Result<Vec<f32>, TraceError> {
        self.decode(DType::Float32, |b| {
            f32::from_le_bytes([b[0], b[1], b[2], b[3]])
        })
    }

    /// View the raw bytes as `i8`. Accepts both `Int8` and `Intgemm8` (the
    /// latter is int8 data in an intgemm-specific tile layout).
    pub fn to_i8(&self) -> Result<Vec<i8>, TraceError> {
        if self.dtype != DType::Int8 && self.dtype != DType::Intgemm8 {
            return Err(TraceError::DTypeMismatch {
                requested: "int8",
                actual: self.dtype,
            });
        }
        Ok(self.data.iter().map(|&b| b as i8).collect())
    }

    /// View the raw bytes as `i32`. Errors unless the dtype is `Int32`.
    pub fn to_i32(&self) -> Result<Vec<i32>, TraceError> {
        self.decode(DType::Int32, |b| {
            i32::from_le_bytes([b[0], b[1], b[2], b[3]])
        })
    }

    /// View the raw bytes as `u32`. Errors unless the dtype is `UInt32`. This is
    /// marian's `IndexType`, used by gather ops (`rows`/`cols`) for indices.
    pub fn to_u32(&self) -> Result<Vec<u32>, TraceError> {
        self.decode(DType::UInt32, |b| {
            u32::from_le_bytes([b[0], b[1], b[2], b[3]])
        })
    }

    fn decode<T, F>(&self, expected: DType, convert: F) -> Result<Vec<T>, TraceError>
    where
        F: Fn(&[u8]) -> T,
    {
        if self.dtype != expected {
            return Err(TraceError::DTypeMismatch {
                requested: expected.name(),
                actual: self.dtype,
            });
        }
        let size = expected.element_size();
        Ok(self.data.chunks_exact(size).map(convert).collect())
    }
}

/// A parsed reference trace: node records in forward-execution order.
#[derive(Clone, Debug)]
pub struct Trace {
    /// Format version the file declared.
    pub version: u32,
    /// All node records, in the order they executed.
    pub records: Vec<TraceRecord>,
}

impl Trace {
    /// Read and parse a trace file from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Trace, TraceError> {
        let bytes = std::fs::read(path.as_ref()).map_err(TraceError::Io)?;
        Trace::from_bytes(&bytes)
    }

    /// Parse a trace from an in-memory byte buffer.
    pub fn from_bytes(bytes: &[u8]) -> Result<Trace, TraceError> {
        let mut cursor = Cursor::new(bytes);

        let magic = cursor.take(4)?;
        if magic != MAGIC {
            return Err(TraceError::BadMagic);
        }
        let version = cursor.u32()?;
        if version != SUPPORTED_VERSION {
            return Err(TraceError::UnsupportedVersion(version));
        }

        let mut records = Vec::new();
        while !cursor.at_end() {
            let id = cursor.u64()?;
            let op_type = cursor.string()?;
            let name = cursor.string()?;
            let dtype = DType::from_raw(cursor.u64()?)?;

            let rank = cursor.u32()? as usize;
            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                shape.push(cursor.i32()?);
            }

            let num_children = cursor.u32()? as usize;
            let mut children = Vec::with_capacity(num_children);
            for _ in 0..num_children {
                children.push(cursor.u64()?);
            }

            let byte_len = cursor.u64()? as usize;
            let data = cursor.take(byte_len)?.to_vec();

            records.push(TraceRecord {
                id,
                op_type,
                name,
                dtype,
                shape,
                children,
                data,
            });
        }

        Ok(Trace { version, records })
    }

    /// Number of recorded nodes.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the trace has no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Resolve the input records of the node at `index` by matching each child
    /// id to the most recent earlier record with that id. "Most recent earlier"
    /// is the correct rule because a node always runs after its inputs, and ids
    /// repeat across forward passes — the nearest prior occurrence is the one
    /// that actually fed this node.
    ///
    /// Errors if a child id has no earlier record (which would indicate a
    /// malformed trace).
    pub fn inputs(&self, index: usize) -> Result<Vec<&TraceRecord>, TraceError> {
        let record = self
            .records
            .get(index)
            .ok_or(TraceError::IndexOutOfRange(index))?;

        let mut inputs = Vec::with_capacity(record.children.len());
        for &child_id in &record.children {
            let found = self.records[..index]
                .iter()
                .rev()
                .find(|candidate| candidate.id == child_id)
                .ok_or(TraceError::MissingChild {
                    node_index: index,
                    child_id,
                })?;
            inputs.push(found);
        }
        Ok(inputs)
    }
}

/// A minimal little-endian byte cursor over the trace buffer.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Cursor<'a> {
        Cursor { bytes, pos: 0 }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], TraceError> {
        let end = self.pos.checked_add(n).ok_or(TraceError::Truncated)?;
        let slice = self.bytes.get(self.pos..end).ok_or(TraceError::Truncated)?;
        self.pos = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, TraceError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i32(&mut self) -> Result<i32, TraceError> {
        Ok(self.u32()? as i32)
    }

    fn u64(&mut self) -> Result<u64, TraceError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn string(&mut self) -> Result<String, TraceError> {
        let len = self.u32()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| TraceError::InvalidUtf8)
    }
}

/// Errors from reading or interpreting a trace.
#[derive(Debug)]
pub enum TraceError {
    Io(std::io::Error),
    /// The file did not start with the `MTRC` magic.
    BadMagic,
    /// The file declared a version this reader does not support.
    UnsupportedVersion(u32),
    /// A record ran past the end of the buffer.
    Truncated,
    /// A string field was not valid UTF-8.
    InvalidUtf8,
    /// A `value_type` field held a marian type value this reader does not model.
    UnknownDType(u64),
    /// A typed view was requested for the wrong dtype.
    DTypeMismatch {
        requested: &'static str,
        actual: DType,
    },
    /// `inputs()` was called with an out-of-range index.
    IndexOutOfRange(usize),
    /// A node referenced a child id with no earlier record.
    MissingChild {
        node_index: usize,
        child_id: u64,
    },
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TraceError::Io(e) => write!(f, "i/o error reading trace: {e}"),
            TraceError::BadMagic => write!(f, "not a trace file (missing MTRC magic)"),
            TraceError::UnsupportedVersion(v) => write!(f, "unsupported trace version {v}"),
            TraceError::Truncated => write!(f, "trace ended mid-record (truncated)"),
            TraceError::InvalidUtf8 => write!(f, "trace string field was not valid UTF-8"),
            TraceError::UnknownDType(raw) => write!(f, "unknown marian type value {raw:#x}"),
            TraceError::DTypeMismatch { requested, actual } => {
                write!(f, "requested {requested} view of a {actual} tensor")
            }
            TraceError::IndexOutOfRange(i) => write!(f, "record index {i} out of range"),
            TraceError::MissingChild {
                node_index,
                child_id,
            } => write!(
                f,
                "node at index {node_index} references child id {child_id} with no earlier record"
            ),
        }
    }
}

impl std::error::Error for TraceError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid trace in memory so parsing is tested without a
    /// multi-hundred-MB fixture on disk.
    fn synthetic_trace() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&SUPPORTED_VERSION.to_le_bytes());

        // Record 0: a float32 "param" with id 7, shape [2, 2], no children.
        let write_record = |b: &mut Vec<u8>,
                            id: u64,
                            op: &str,
                            name: &str,
                            dtype: u64,
                            shape: &[i32],
                            children: &[u64],
                            data: &[u8]| {
            b.extend_from_slice(&id.to_le_bytes());
            b.extend_from_slice(&(op.len() as u32).to_le_bytes());
            b.extend_from_slice(op.as_bytes());
            b.extend_from_slice(&(name.len() as u32).to_le_bytes());
            b.extend_from_slice(name.as_bytes());
            b.extend_from_slice(&dtype.to_le_bytes());
            b.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for &d in shape {
                b.extend_from_slice(&d.to_le_bytes());
            }
            b.extend_from_slice(&(children.len() as u32).to_le_bytes());
            for &c in children {
                b.extend_from_slice(&c.to_le_bytes());
            }
            b.extend_from_slice(&(data.len() as u64).to_le_bytes());
            b.extend_from_slice(data);
        };

        let param: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        write_record(&mut b, 7, "param", "F0::W", 0x404, &[2, 2], &[], &param);

        // Record 1: an intgemmAffine consuming node 7, float32 output shape [2].
        let out: Vec<u8> = [10.0f32, 20.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        write_record(&mut b, 8, "intgemmAffine", "none", 0x404, &[2], &[7], &out);

        b
    }

    #[test]
    fn parses_header_and_records() {
        let trace = Trace::from_bytes(&synthetic_trace()).unwrap();
        assert_eq!(trace.version, 1);
        assert_eq!(trace.len(), 2);

        let param = &trace.records[0];
        assert_eq!(param.id, 7);
        assert_eq!(param.op_type, "param");
        assert_eq!(param.name, "F0::W");
        assert_eq!(param.dtype, DType::Float32);
        assert_eq!(param.shape, vec![2, 2]);
        assert_eq!(param.num_elements(), 4);
        assert!(param.children.is_empty());
        assert_eq!(param.to_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn resolves_inputs_by_backward_search() {
        let trace = Trace::from_bytes(&synthetic_trace()).unwrap();
        let inputs = trace.inputs(1).unwrap();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].id, 7);
        assert_eq!(inputs[0].op_type, "param");
    }

    #[test]
    fn wrong_dtype_view_errors() {
        let trace = Trace::from_bytes(&synthetic_trace()).unwrap();
        assert!(matches!(
            trace.records[0].to_i8(),
            Err(TraceError::DTypeMismatch { .. })
        ));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = synthetic_trace();
        bytes[0] = b'X';
        assert!(matches!(
            Trace::from_bytes(&bytes),
            Err(TraceError::BadMagic)
        ));
    }

    #[test]
    fn rejects_truncated_record() {
        let full = synthetic_trace();
        // Cut mid-way through the records so a length runs past the end.
        let truncated = &full[..full.len() - 3];
        assert!(matches!(
            Trace::from_bytes(truncated),
            Err(TraceError::Truncated)
        ));
    }

    #[test]
    fn dtype_sizes() {
        assert_eq!(DType::Float32.element_size(), 4);
        assert_eq!(DType::Int8.element_size(), 1);
        assert_eq!(DType::Intgemm8.element_size(), 1);
        assert_eq!(DType::from_raw(0x4101).unwrap(), DType::Intgemm8);
        assert!(matches!(
            DType::from_raw(0x800),
            Err(TraceError::UnknownDType(0x800))
        ));
    }
}
