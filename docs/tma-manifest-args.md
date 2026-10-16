# Design note: TMA tensor maps as manifest arguments

Status: proposal, not implemented.

A CUDA kernel that uses TMA (`cp.async.bulk.tensor`, `UTMALDG` in SASS) needs a
`CUtensorMap` built on the host with `cuTensorMapEncodeTiled`. A manifest
cannot run host code, so today a CUDA candidate cannot use TMA at all. cuTile
and Triton are not affected: they build their descriptors themselves. This
note proposes one more argument type, built by arena from data in the
manifest, so the manifest stays data and not code.

## The argument

```json
{"tensor_map": "a", "shape": ["M", "K"], "box": [128, 8],
 "elem": "f32", "swizzle": "128B", "interleave": "none",
 "l2_promotion": "128B", "oob_fill": "zero"}
```

| Field | Meaning |
| --- | --- |
| `tensor_map` | A category buffer or declared scratch name. Its device pointer is the global address. |
| `shape` | Global dims as expressions, row-major (slowest first), like `cutile_array`. Arena reverses them for the driver, which wants the fastest dim first. |
| `strides` | Optional, in elements, row-major. Default contiguous. Arena converts to bytes and drops the innermost one, as the driver expects. |
| `box` | Box dims as expressions, same order as `shape`. |
| `elem` | `f32`, `f16`, `bf16`, `i32`, `u8` ... Must match the buffer's dtype. |
| `swizzle` | `none`, `32B`, `64B`, `128B`. |
| `interleave` | `none`, `16B`, `32B`. Default `none`. |
| `l2_promotion` | `none`, `64B`, `128B`, `256B`. Default `none`. |
| `oob_fill` | `zero` (`CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`, fills zeros) or `nan` (`NAN_REQUEST_ZERO_FMA`). Default `zero`. |

Arena encodes it into a 64-byte aligned `CUtensorMap` and pushes its 128
bytes as one parameter. The kernel declares it as
`const __grid_constant__ CUtensorMap tmap`, passed by value. The driver's
parameter sizes already checked by `pack_args` must report 128 bytes for that
slot; anything else is the usual "argument N has the wrong size" error.
`PackedArgs` stores bytes in a plain vector, so the encode must target an
aligned temporary that is then copied in. The map embeds the buffer address,
so arena re-encodes it, outside the timed op, whenever a buffer is
reallocated (new sizes, verification buffers).

## Validation (in `pack_args`, so `--plan` and `pgaio.py check` catch it)

- Rank 1 to 5; `box`, `shape` and `strides` of the same rank.
- Global address 16-byte aligned (32 with `interleave: 32B`). cudaMalloc
  buffers are; a future offset argument would have to keep it so.
- Every byte stride a multiple of 16 and below 2^40. For f32 this means the
  row length in elements is a multiple of 4: `K % 4 == 0` for A.
- Each box dim between 1 and 256. With `interleave: none`, the inner box dim
  times the element size a multiple of 16 bytes, and no larger than the
  swizzle span (128 bytes for `128B`).
- `elem` matches the buffer's dtype in the category.
- Then call `cuTensorMapEncodeTiled` and report its error code if it still
  refuses, naming the manifest entry.

A shape that breaks a rule is an error, not a silent fallback; use `when` to
send unaligned sizes to another launch.

## What `--plan` prints

The resolved values, one object per tensor map argument:

```json
{"tensor_map": "a", "elem": "f32", "dims": [4096, 4096],
 "strides_bytes": [16384], "box": [128, 8], "swizzle": "128B",
 "interleave": "none", "l2_promotion": "128B", "oob_fill": "zero",
 "param_bytes": 128, "ok": true}
```

`--plan` allocates nothing, so it encodes with a stand-in 256-byte aligned
address. That still runs the driver's own checks; the real encode happens
before the first launch.

## DSLs

cuTile lowers loads to TMA from `cutile_array` arguments and Triton builds
descriptors from pointers and strides, so neither needs this type. It is for
CUDA C++ candidates, and possibly a Triton kernel taking a host-built
descriptor, if Triton's AOT signature allows it.

## Open questions

- Should `box` and `swizzle` come from source constants (`// arena: BM=128`)
  only, so the kernel and the map cannot disagree?
- Multicast and im2col (`cuTensorMapEncodeIm2col`) are out of scope; is there
  a category that needs them?
- Does the PGAIO SASS diff need to flag `UTMALDG` without a matching tensor
  map argument (a kernel that builds its own map on the device)?
