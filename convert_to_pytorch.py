"""Convert MediaPipe's face_landmarker.task bundle into a PyTorch checkpoint.

The .task file is a zip holding three TFLite models (face detector, face landmarks
detector, blendshapes) plus the face-geometry pipeline metadata. Each TFLite graph is
parsed from its flatbuffer and stored as a list of ops + float32 weights that
`face_mesh_torch.TFLiteGraph` executes with plain PyTorch ops. float16 weights are
dequantized here, and conv kernels are re-laid-out to PyTorch's OIHW.

Does not import mediapipe. Needs the `tflite` flatbuffer schema package (dev dependency).

Usage:
    python convert_to_pytorch.py [models/face_landmarker.task] [models/face_landmarker_torch.pt]
"""

import struct
import sys
import zipfile
from pathlib import Path

import flatbuffers
import numpy as np
import tflite
import torch
from tflite.BuiltinOperator import BuiltinOperator
from tflite.BuiltinOptions import BuiltinOptions

OP_NAMES = {v: k for k, v in BuiltinOperator.__dict__.items() if not k.startswith("_")}
NP_DTYPES = {0: np.float32, 1: np.float16, 2: np.int32, 3: np.uint8, 4: np.int64, 9: np.int8}
PADDING = {0: "SAME", 1: "VALID"}
ACTIVATIONS = {0: None, 1: "RELU", 2: "RELU_N1_TO_1", 3: "RELU6", 4: "TANH"}

SUPPORTED_OPS = {
    "ADD", "SUB", "MUL", "DIV", "SQUARED_DIFFERENCE", "CONV_2D", "DEPTHWISE_CONV_2D",
    "MAX_POOL_2D", "PAD", "RELU", "PRELU", "LOGISTIC", "NEG", "SQRT", "RSQRT", "RESHAPE",
    "CONCATENATION", "MEAN", "SUM", "TRANSPOSE", "STRIDED_SLICE",
}


def _options(op, cls):
    table = op.BuiltinOptions()
    opts = cls()
    opts.Init(table.Bytes, table.Pos)
    return opts


class TFLiteReader:
    def __init__(self, data: bytes):
        self.data = data
        self.model = tflite.Model.GetRootAsModel(data, 0)
        self.graph = self.model.Subgraphs(0)

    def tensor_name(self, i):
        return self.graph.Tensors(i).Name().decode()

    def tensor_shape(self, i):
        t = self.graph.Tensors(i)
        return t.ShapeAsNumpy().tolist() if t.ShapeLength() else []

    def tensor_constant(self, i):
        t = self.graph.Tensors(i)
        buf = self.model.Buffers(t.Buffer())
        if buf is None or buf.DataLength() == 0:
            return None
        arr = np.frombuffer(buf.DataAsNumpy().tobytes(), dtype=NP_DTYPES[t.Type()])
        return arr.reshape(self.tensor_shape(i))

    def op_name(self, op):
        code = self.model.OperatorCodes(op.OpcodeIndex())
        builtin = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
        if builtin == BuiltinOperator.CUSTOM:
            return "CUSTOM:" + code.CustomCode().decode()
        return OP_NAMES[builtin]

    def metadata_buffer(self, name: bytes):
        for i in range(self.model.MetadataLength()):
            md = self.model.Metadata(i)
            if md.Name() == name:
                return self.model.Buffers(md.Buffer()).DataAsNumpy().tobytes()
        return None


def _fb_field(table: flatbuffers.table.Table, index: int):
    return table.Offset(4 + 2 * index)


def _fb_vector_of_tables(table, index):
    off = _fb_field(table, index)
    if not off:
        return []
    start, n = table.Vector(off), table.VectorLen(off)
    return [flatbuffers.table.Table(table.Bytes, table.Indirect(start + 4 * k)) for k in range(n)]


def _fb_float_vector(table, index):
    off = _fb_field(table, index)
    if not off:
        return []
    start, n = table.Vector(off), table.VectorLen(off)
    return [table.Get(flatbuffers.number_types.Float32Flags, start + 4 * k) for k in range(n)]


def read_input_normalization(reader: TFLiteReader):
    """Read (mean, std) of the image input from the TFLite model metadata flatbuffer.

    Path: ModelMetadata.subgraph_metadata[0].input_tensor_metadata[0].process_units[*]
    -> NormalizationOptions{mean, std}.
    """
    buf = reader.metadata_buffer(b"TFLITE_METADATA")
    if buf is None:
        return None
    root = flatbuffers.table.Table(buf, flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, 0))
    subgraph = _fb_vector_of_tables(root, 3)[0]
    tensor = _fb_vector_of_tables(subgraph, 2)[0]
    for unit in _fb_vector_of_tables(tensor, 4):
        type_off = _fb_field(unit, 0)
        if type_off and unit.Get(flatbuffers.number_types.Uint8Flags, unit.Pos + type_off) == 1:
            norm = flatbuffers.table.Table(unit.Bytes, 0)
            unit.Union(norm, _fb_field(unit, 1))
            return _fb_float_vector(norm, 0), _fb_float_vector(norm, 1)
    return None


def convert_graph(reader: TFLiteReader):
    """Turn a TFLite subgraph into a serializable op list + float32 tensors (PyTorch layouts)."""
    g = reader.graph
    tensors = {}  # buffer name -> torch.Tensor
    constants = {}  # tflite tensor index -> np.ndarray
    for i in range(g.TensorsLength()):
        value = reader.tensor_constant(i)
        if value is not None:
            constants[i] = value

    def const_ref(tid, array=None):
        name = f"t{tid}"
        if name not in tensors:
            arr = constants[tid] if array is None else array
            tensors[name] = torch.from_numpy(np.ascontiguousarray(arr).astype(np.float32))
        return name

    def operand(tid):
        if tid in constants:
            return {"const": const_ref(tid)}
        return {"tensor": int(tid)}

    nodes = []
    for k in range(g.OperatorsLength()):
        op = g.Operators(k)
        name = reader.op_name(op)
        ins = [int(t) for t in op.InputsAsNumpy()] if op.InputsLength() else []
        outs = [int(t) for t in op.OutputsAsNumpy()]

        if name == "DEQUANTIZE":
            if ins[0] not in constants:
                raise NotImplementedError("DEQUANTIZE of a non-constant tensor")
            constants[outs[0]] = constants[ins[0]].astype(np.float32)
            continue
        if name not in SUPPORTED_OPS:
            raise NotImplementedError(f"Unsupported TFLite op: {name}")

        node = {"op": name, "out": outs[0], "out_shape": reader.tensor_shape(outs[0])}
        opt_type = op.BuiltinOptionsType()

        if name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            weight = constants[ins[1]].astype(np.float32)
            if name == "CONV_2D":
                o = _options(op, tflite.Conv2DOptions)
                weight = weight.transpose(0, 3, 1, 2)  # OHWI -> OIHW
            else:
                o = _options(op, tflite.DepthwiseConv2DOptions)
                weight = weight.transpose(3, 0, 1, 2)  # 1HW(C*M) -> (C*M)1HW
            bias = None
            if len(ins) > 2 and ins[2] >= 0:
                bias = const_ref(ins[2])
            node.update(
                input=operand(ins[0]),
                weight=_store(tensors, f"w{k}", weight),
                bias=bias,
                stride=[o.StrideH(), o.StrideW()],
                dilation=[o.DilationHFactor(), o.DilationWFactor()],
                padding=PADDING[o.Padding()],
                act=ACTIVATIONS[o.FusedActivationFunction()],
            )
        elif name == "MAX_POOL_2D":
            o = _options(op, tflite.Pool2DOptions)
            node.update(
                input=operand(ins[0]),
                kernel=[o.FilterHeight(), o.FilterWidth()],
                stride=[o.StrideH(), o.StrideW()],
                padding=PADDING[o.Padding()],
                act=ACTIVATIONS[o.FusedActivationFunction()],
            )
        elif name == "PAD":
            node.update(input=operand(ins[0]), paddings=constants[ins[1]].astype(int).tolist())
        elif name in ("ADD", "SUB", "MUL", "DIV"):
            cls = {"ADD": tflite.AddOptions, "SUB": tflite.SubOptions,
                   "MUL": tflite.MulOptions, "DIV": tflite.DivOptions}[name]
            act = None
            if opt_type != BuiltinOptions.NONE:
                act = ACTIVATIONS[_options(op, cls).FusedActivationFunction()]
            node.update(a=operand(ins[0]), b=operand(ins[1]), act=act)
        elif name in ("SQUARED_DIFFERENCE", "PRELU"):
            node.update(a=operand(ins[0]), b=operand(ins[1]))
        elif name in ("RELU", "LOGISTIC", "NEG", "SQRT", "RSQRT", "RESHAPE"):
            node.update(input=operand(ins[0]))
        elif name == "CONCATENATION":
            o = _options(op, tflite.ConcatenationOptions)
            node.update(inputs=[operand(t) for t in ins], axis=o.Axis(),
                        act=ACTIVATIONS[o.FusedActivationFunction()])
        elif name in ("MEAN", "SUM"):
            o = _options(op, tflite.ReducerOptions)
            axes = np.atleast_1d(constants[ins[1]]).astype(int).tolist()
            node.update(input=operand(ins[0]), axes=axes, keep_dims=bool(o.KeepDims()))
        elif name == "TRANSPOSE":
            node.update(input=operand(ins[0]), perm=constants[ins[1]].astype(int).tolist())
        elif name == "STRIDED_SLICE":
            o = _options(op, tflite.StridedSliceOptions)
            if o.EllipsisMask() or o.NewAxisMask():
                raise NotImplementedError("STRIDED_SLICE ellipsis/new-axis masks")
            node.update(
                input=operand(ins[0]),
                begin=constants[ins[1]].astype(int).tolist(),
                end=constants[ins[2]].astype(int).tolist(),
                strides=constants[ins[3]].astype(int).tolist(),
                begin_mask=o.BeginMask(),
                end_mask=o.EndMask(),
                shrink_axis_mask=o.ShrinkAxisMask(),
            )
        nodes.append(node)

    spec = {
        "inputs": [int(t) for t in g.InputsAsNumpy()],
        "input_shapes": [reader.tensor_shape(int(t)) for t in g.InputsAsNumpy()],
        "outputs": [int(t) for t in g.OutputsAsNumpy()],
        "output_names": [reader.tensor_name(int(t)) for t in g.OutputsAsNumpy()],
        "nodes": nodes,
    }
    return spec, tensors


def _store(tensors, name, array):
    tensors[name] = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    return name


def _read_varint(buf, pos):
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return result, pos


def _parse_proto(buf):
    """Minimal protobuf wire-format parser: returns {field_number: [values]}."""
    fields, pos = {}, 0
    while pos < len(buf):
        key, pos = _read_varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _read_varint(buf, pos)
        elif wire == 1:
            value, pos = buf[pos:pos + 8], pos + 8
        elif wire == 2:
            n, pos = _read_varint(buf, pos)
            value, pos = buf[pos:pos + n], pos + n
        elif wire == 5:
            value, pos = buf[pos:pos + 4], pos + 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        fields.setdefault(field, []).append((wire, value))
    return fields


def _proto_floats(entries):
    out = []
    for wire, value in entries:
        if wire == 5:
            out.append(struct.unpack("<f", value)[0])
        else:  # packed
            out.extend(struct.unpack(f"<{len(value) // 4}f", value))
    return out


def read_geometry_metadata(buf: bytes):
    """Parse GeometryPipelineMetadata (canonical mesh + Procrustes landmark weights)."""
    meta = _parse_proto(buf)
    mesh = _parse_proto(meta[1][0][1])
    if mesh.get(1, [(0, 0)])[0][1] != 0:
        raise ValueError("expected VERTEX_PT canonical mesh")
    vertices = np.array(_proto_floats(mesh[3]), dtype=np.float32).reshape(-1, 5)  # x y z u v
    weights = np.zeros(len(vertices), dtype=np.float32)
    for _, ref in meta.get(2, []):
        r = _parse_proto(ref)
        weights[r[1][0][1]] = struct.unpack("<f", r[2][0][1])[0]
    input_source = meta.get(3, [(0, 1)])[0][1]
    if input_source not in (0, 1):
        raise ValueError("only FACE_LANDMARK_PIPELINE geometry metadata is supported")
    return {
        "canonical_landmarks": torch.from_numpy(np.ascontiguousarray(vertices[:, :3].T)),
        "landmark_weights": torch.from_numpy(weights),
    }


def convert(task_path: Path, out_path: Path):
    with zipfile.ZipFile(task_path) as zf:
        files = {n: zf.read(n) for n in zf.namelist()}

    checkpoint = {"format": "face_landmarker_torch/1", "models": {}}
    for key, fname in [
        ("detector", "face_detector.tflite"),
        ("landmarks", "face_landmarks_detector.tflite"),
        ("blendshapes", "face_blendshapes.tflite"),
    ]:
        reader = TFLiteReader(files[fname])
        spec, tensors = convert_graph(reader)
        norm = read_input_normalization(reader) if key != "blendshapes" else None
        if norm is not None:
            spec["normalization"] = {"mean": [float(v) for v in norm[0]], "std": [float(v) for v in norm[1]]}
        checkpoint["models"][key] = {"spec": spec, "tensors": tensors}
        n_params = sum(t.numel() for t in tensors.values())
        print(f"{fname}: {len(spec['nodes'])} ops, {n_params:,} weights, "
              f"normalization={spec.get('normalization')}")

    checkpoint["geometry"] = read_geometry_metadata(files["geometry_pipeline_metadata_landmarks.binarypb"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    here = Path(__file__).parent
    task = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "models" / "face_landmarker.task"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "models" / "face_landmarker_torch.pt"
    convert(task, out)
