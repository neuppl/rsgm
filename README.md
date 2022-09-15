# Summary

A small library for importing Bayesian networks into Rust

## Usage

### Converter

The `converter` directory contains a `bif_to_json.py` file that converts an
arbitrary `bif` file into the JSON format that can be read by the library.
To use it, first install the `pgmpy` python package. Then, it can be invoked as:

```
python bif_to_json.py input.bif > output.json
```

### rsgm

The RSGM library contains tools for parsing and manipulating Bayesian networks
from a JSON representation.
