# Synthdnn

Synthdnn is a suite of scripts for collecting and analyzing oneDNN performance
across a randomly generated suite of synthetic data. The general architecture is
a standard data pipeline composed of synthetic problem generation, data
collection, data storage, and data analysis. The `collectdnn.py` script provides
a command line interface combining synthetic problem generation and data
collection and data is output in a csv format for storage. The script
`reportdnn.py` can then be used on these files to generate various
reports/plots. Real-time report generation is supported and can be accomplished
via pipes

```
ssh <machine> "<remote_sythdnn_dir>/collectdnn.py <args>" | <local_synthdnn_dir>/reportdnn.py <args>
```

For information on how to use the above scripts, see the output of the
`-h/--help` argument.
