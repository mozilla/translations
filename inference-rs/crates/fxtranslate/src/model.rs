//! Reader for the marian binary model format (`*.intgemm.alphas.bin`), the
//! format `translator-cli` loads (`marian-fork/src/common/binary.cpp`).
//!
//! Why the Rust engine needs this: the reference **trace** records each int8
//! weight `B` in gemmology's CPU-specific *packed* tile layout, which is an
//! opaque, architecture-dependent blob. The **model file**, by contrast, stores
//! `B` in a plain, architecture-agnostic layout — logical `int8`, already
//! transposed, with the quantization multiplier appended
//! (`integer_common.h:prepareAndTransposeB`). marian only repacks into the tile
//! layout in memory at load time, so to run the int8 GEMM we read the logical
//! weights straight from the model and never touch the packed layout.
//!
//! Binary layout (little-endian):
//!
//! ```text
//! header  : version:u64 (== 1) | num_items:u64
//! headers : num_items × { name_len:u64, type:u64, shape_len:u64, data_len:u64 }
//! names   : num_items × name_len bytes (NUL-padded)
//! shapes  : num_items × shape_len × i32
//! gap     : offset:u64 | offset padding bytes (aligns data to 256 bytes)
//! data    : num_items × data_len bytes
//! ```
//!
//! For an `intgemm8` weight of `E` logical elements, its `data` block is `E`
//! bytes of `int8`, then a 4-byte `f32` quantization multiplier, then padding.

use std::fmt;
use std::fs::File;
use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use crate::trace::{DType, TraceError};

/// Expected value of the leading version word.
const BINARY_FILE_VERSION: u64 = 1;

/// A tensor's bytes: either owned on the heap (default `Model::load`) or a view
/// into a memory-mapped model file (`--mmap` / [`Model::load_mmapped`]). Mapped
/// bytes are backed by the shared `Mmap`, so they cost no heap and their pages
/// are file-backed (clean, reclaimable) rather than dirty anonymous copies.
/// Derefs to `[u8]`, so all accessors are agnostic to which it is.
#[derive(Clone)]
pub enum Bytes {
    Owned(Vec<u8>),
    Mapped {
        map: Arc<Mmap>,
        off: usize,
        len: usize,
    },
}

impl Deref for Bytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Bytes::Owned(v) => v,
            Bytes::Mapped { map, off, len } => &map[*off..*off + *len],
        }
    }
}

impl fmt::Debug for Bytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self {
            Bytes::Owned(_) => "owned",
            Bytes::Mapped { .. } => "mapped",
        };
        write!(f, "Bytes::{kind}({} bytes)", self.len())
    }
}

/// One named tensor from the model file.
#[derive(Clone, Debug)]
pub struct ModelItem {
    pub name: String,
    pub dtype: DType,
    /// Logical shape as declared in the file. Note: for the transposed int8
    /// weights the *data* is laid out as the transpose of this shape (see
    /// [`ModelItem::int8_transposed`]).
    pub shape: Vec<i32>,
    /// The item's full data block, including any appended quant multiplier and
    /// padding for int8 weights.
    pub data: Bytes,
}

impl ModelItem {
    /// Number of logical tensor elements (product of the shape dims).
    pub fn num_elements(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    /// The `int8` weight values as `i8`, i.e. the first `num_elements()` bytes
    /// of the data block (excluding the appended quant multiplier and padding).
    ///
    /// The values are stored *transposed*: a weight that is logically
    /// `[K, N]` (used as `A[M,K] · W[K,N]`) is laid out row-major as `[N, K]`,
    /// so `W[k, n] == int8_transposed()[n * K + k]`.
    pub fn int8_transposed(&self) -> Result<&[i8], TraceError> {
        if self.dtype != DType::Intgemm8 && self.dtype != DType::Int8 {
            return Err(TraceError::DTypeMismatch {
                requested: "int8",
                actual: self.dtype,
            });
        }
        let n = self.num_elements();
        let bytes = &self.data[..n];
        // SAFETY-free reinterpret: i8 and u8 have identical layout.
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i8, n) })
    }

    /// The quantization multiplier appended right after an int8 weight's data.
    pub fn quant_mult(&self) -> Result<f32, TraceError> {
        if self.dtype != DType::Intgemm8 && self.dtype != DType::Int8 {
            return Err(TraceError::DTypeMismatch {
                requested: "int8",
                actual: self.dtype,
            });
        }
        let n = self.num_elements();
        let b = self.data.get(n..n + 4).ok_or(TraceError::Truncated)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Decode a float32 item's data as `f32`.
    pub fn to_f32(&self) -> Result<Vec<f32>, TraceError> {
        if self.dtype != DType::Float32 {
            return Err(TraceError::DTypeMismatch {
                requested: "float32",
                actual: self.dtype,
            });
        }
        Ok(self.data[..self.num_elements() * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }
}

/// A parsed marian binary model.
#[derive(Clone, Debug)]
pub struct Model {
    pub items: Vec<ModelItem>,
}

/// Per-item layout from the file header: `(name, type_raw, shape, data_offset,
/// data_len)`, where `data_offset` is the byte offset of the data block.
type Layout = Vec<(String, u64, Vec<i32>, usize, usize)>;

impl Model {
    /// Read and parse a model file from disk, copying each tensor to the heap.
    pub fn load(path: impl AsRef<Path>) -> Result<Model, TraceError> {
        let bytes = std::fs::read(path.as_ref()).map_err(TraceError::Io)?;
        Model::from_bytes(&bytes)
    }

    /// Memory-map a model file; tensors become views into the mapping (no heap
    /// copy, file-backed pages). The mapping lives as long as any item holds it.
    pub fn load_mmapped(path: impl AsRef<Path>) -> Result<Model, TraceError> {
        let file = File::open(path.as_ref()).map_err(TraceError::Io)?;
        // SAFETY: the file is opened read-only and the mapping is treated as
        // immutable for the whole lifetime of the Model — we never write it, and
        // the model files are not concurrently mutated during a run.
        let map = Arc::new(unsafe { Mmap::map(&file).map_err(TraceError::Io)? });
        Model::from_mmap(map)
    }

    /// Look up an item by exact name.
    pub fn get(&self, name: &str) -> Option<&ModelItem> {
        self.items.iter().find(|it| it.name == name)
    }

    /// Parse the header/name/shape/gap section and locate each data block.
    fn parse_layout(bytes: &[u8]) -> Result<Layout, TraceError> {
        let mut c = Reader::new(bytes);

        let version = c.u64()?;
        if version != BINARY_FILE_VERSION {
            return Err(TraceError::UnsupportedVersion(version as u32));
        }
        let num_items = c.u64()? as usize;

        let mut headers = Vec::with_capacity(num_items);
        for _ in 0..num_items {
            let name_len = c.u64()? as usize;
            let type_raw = c.u64()?;
            let shape_len = c.u64()? as usize;
            let data_len = c.u64()? as usize;
            headers.push((name_len, type_raw, shape_len, data_len));
        }

        // Names (NUL-padded within name_len bytes).
        let mut names = Vec::with_capacity(num_items);
        for &(name_len, ..) in &headers {
            let raw = c.take(name_len)?;
            let end = raw.iter().position(|&b| b == 0).unwrap_or(name_len);
            names.push(String::from_utf8_lossy(&raw[..end]).into_owned());
        }

        // Shapes.
        let mut shapes = Vec::with_capacity(num_items);
        for &(_, _, shape_len, _) in &headers {
            let mut shape = Vec::with_capacity(shape_len);
            for _ in 0..shape_len {
                shape.push(c.i32()?);
            }
            shapes.push(shape);
        }

        // Alignment gap: a byte count followed by that many padding bytes.
        let offset = c.u64()? as usize;
        c.take(offset)?;

        // Data blocks: record each block's absolute offset (bounds-checked).
        let mut out = Layout::with_capacity(num_items);
        for i in 0..num_items {
            let (_, type_raw, _, data_len) = headers[i];
            let off = c.pos;
            c.take(data_len)?; // advance + bounds-check
            out.push((
                std::mem::take(&mut names[i]),
                type_raw,
                std::mem::take(&mut shapes[i]),
                off,
                data_len,
            ));
        }
        Ok(out)
    }

    /// Parse a model from an in-memory buffer (tensors copied to the heap).
    pub fn from_bytes(bytes: &[u8]) -> Result<Model, TraceError> {
        let layout = Model::parse_layout(bytes)?;
        let mut items = Vec::with_capacity(layout.len());
        for (name, type_raw, shape, off, len) in layout {
            items.push(ModelItem {
                name,
                dtype: DType::from_raw(type_raw)?,
                shape,
                data: Bytes::Owned(bytes[off..off + len].to_vec()),
            });
        }
        Ok(Model { items })
    }

    /// Parse a model from a memory mapping (tensors are views into it).
    fn from_mmap(map: Arc<Mmap>) -> Result<Model, TraceError> {
        let layout = Model::parse_layout(&map)?;
        let mut items = Vec::with_capacity(layout.len());
        for (name, type_raw, shape, off, len) in layout {
            items.push(ModelItem {
                name,
                dtype: DType::from_raw(type_raw)?,
                shape,
                data: Bytes::Mapped {
                    map: map.clone(),
                    off,
                    len,
                },
            });
        }
        Ok(Model { items })
    }
}

/// A minimal little-endian byte cursor. (Local to this module rather than shared
/// with `trace` because the field widths differ — the model uses u64 lengths.)
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Reader<'a> {
        Reader { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], TraceError> {
        let end = self.pos.checked_add(n).ok_or(TraceError::Truncated)?;
        let slice = self.bytes.get(self.pos..end).ok_or(TraceError::Truncated)?;
        self.pos = end;
        Ok(slice)
    }

    fn u64(&mut self) -> Result<u64, TraceError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn i32(&mut self) -> Result<i32, TraceError> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "model with {} items", self.items.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny valid model in memory: one f32 bias and one int8 weight.
    fn synthetic_model() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&BINARY_FILE_VERSION.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes()); // num_items

        // Item 0: f32 "bias" shape [1,2], data_len 8.
        // Item 1: int8 "W" shape [2,2], data_len 8 (4 int8 + 4-byte quantMult).
        let write_header = |b: &mut Vec<u8>, name: &str, ty: u64, shape_len: u64, data_len: u64| {
            b.extend_from_slice(&(name.len() as u64).to_le_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&shape_len.to_le_bytes());
            b.extend_from_slice(&data_len.to_le_bytes());
        };
        write_header(&mut b, "bias", 0x404, 2, 8);
        write_header(&mut b, "W", 0x4101, 2, 8);

        b.extend_from_slice(b"bias");
        b.extend_from_slice(b"W");

        for d in [1i32, 2] {
            b.extend_from_slice(&d.to_le_bytes());
        }
        for d in [2i32, 2] {
            b.extend_from_slice(&d.to_le_bytes());
        }

        // No alignment padding in this synthetic file.
        b.extend_from_slice(&0u64.to_le_bytes());

        // bias data: [1.5, -2.0]
        b.extend_from_slice(&1.5f32.to_le_bytes());
        b.extend_from_slice(&(-2.0f32).to_le_bytes());
        // W data: int8 [1, -2, 3, -4] then quantMult 0.5
        b.extend_from_slice(&[1u8, (-2i8) as u8, 3, (-4i8) as u8]);
        b.extend_from_slice(&0.5f32.to_le_bytes());

        b
    }

    #[test]
    fn parses_items() {
        let model = Model::from_bytes(&synthetic_model()).unwrap();
        assert_eq!(model.items.len(), 2);

        let bias = model.get("bias").unwrap();
        assert_eq!(bias.dtype, DType::Float32);
        assert_eq!(bias.shape, vec![1, 2]);
        assert_eq!(bias.to_f32().unwrap(), vec![1.5, -2.0]);

        let w = model.get("W").unwrap();
        assert_eq!(w.dtype, DType::Intgemm8);
        assert_eq!(w.shape, vec![2, 2]);
        assert_eq!(w.int8_transposed().unwrap(), &[1, -2, 3, -4]);
        assert_eq!(w.quant_mult().unwrap(), 0.5);
    }

    #[test]
    fn missing_item_is_none() {
        let model = Model::from_bytes(&synthetic_model()).unwrap();
        assert!(model.get("nope").is_none());
    }

    #[test]
    fn rejects_bad_version() {
        let mut bytes = synthetic_model();
        bytes[0] = 9;
        assert!(matches!(
            Model::from_bytes(&bytes),
            Err(TraceError::UnsupportedVersion(9))
        ));
    }
}
