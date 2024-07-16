# Synthdnn

Synthdnn is a suite of scripts for collecting and analyzing oneDNN performance
across a randomly generated data. The general architecture is intended to follow
a data pipeline composed of synthetic problem generation, data collection, and
data analysis. The `synthdnn.py` script provides a command line interface to
these tools. Sample Usage:


Problem Generation:
```
python3 synthdnn.py <primitive> [sampling controls] -b <batch_file>
```
Performance Data Collection:
```
python3 synthdnn.py collect --engine=<engine> --collect <data_kind> -b <batch_file> <benchdnn_file>
```

Problem Generation and Performance Data Collection:
```
python3 synthdnn.py <primitive> [sampling controls] --engine=<engine> --collect <data_kind> <benchdnn_file>
```

Report Generation: Not yet implemented.
```

See `synthdnn.py -h` for additional details.
