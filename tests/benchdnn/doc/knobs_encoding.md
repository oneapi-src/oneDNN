# Sparse Encodings

**Benchdnn** supports the same sparse encodings as the library does (memory::sparse_encoding
enum). If an unsupported sparse encoding is specified, an error will be reported.
The following sparse encodings are supported:

| Sparse encoding | Description
| :---            | :---
| csr             | Compressed Sparse Row (CSR) encoding

## Usage
```
    --encoding=ENCODING[+SPARSITY]:ENCODING[+SPARSITY]:ENCODING[+SPARSITY]
```

The colon-separated encodings correspond to the source, weights and destination
tensors respectively.
