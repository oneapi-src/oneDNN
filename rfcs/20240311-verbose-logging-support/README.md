# Proposal for a Logging Mechanism in oneDNN

## Motivation

This is a proposal to introduce a logging mechanism within the oneDNN framework to direct, save and manage the verbose outputs generated from oneDNN into user-specified logs.
Presently, oneDNN in the verbose mode prints all information directly to the console using `stdout` - with a logging mechanism, the user will be able to create custom logs by controlling what and how verbose information is saved in the user-specified logfiles. 

## Proposal
The proposal is to implement logging support in oneDNN with the help of [`spdlog`]((https://github.com/gabime/spdlog)), a header-only C++ logging library which provides a Python-like formatting API using the bundled [fmt](https://github.com/fmtlib/fmt) lib. The library uses an MIT license, has no cross-component dependencies and has a distinct performance [gain](https://github.com/gabime/spdlog/tree/v1.x?tab=readme-ov-file#benchmarks) over `sprintf` for logging. Key considerations for the proposal are listed as follows:

### 1. Build Options
`spdlog` can be built with oneDNN using a header-only approach by adding the library headers to the build tree. A pre-compiled version is recommended by the authors to reduce compilation overhead but is not required since the implementation will utilize only limited features from the entire scope of `spdlog`. 

### 2. Runtime Logging Controls
A basic requirement for implementing logging support will be to define the environmental control variables which the user can specify to manage oneDNN data logging. 
For the simple case where the logging mechanism involves directly dumping the verbose outputs into a logfile, this can be accomplished with two control variables, one for enabling the data logging (`ENABLE_ONEDNN_LOGGING`) and the other for specifying the logfile path (`ONEDNN_LOGFILE=/path/to/file`).
In this scenario, the data recorded in the logfile mirrors the printed verbose mode information, hence, the logged data can be managed using oneDNN [runtime controls](https://oneapi-src.github.io/oneDNN/dev_guide_verbose.html?highlight=onednn_verbose#run-time-controls) for the verbose mode.
For better logging capabilities, additional runtime controls can be introduced to separately control the logging levels for the saved data. 

### 3. Alignment and Specification of Logging Levels
`spdlog` defines the following levels for data logging in its implementation:
```
#define SPDLOG_LEVEL_TRACE 0
#define SPDLOG_LEVEL_DEBUG 1
#define SPDLOG_LEVEL_INFO 2
#define SPDLOG_LEVEL_WARN 3
#define SPDLOG_LEVEL_ERROR 4
#define SPDLOG_LEVEL_CRITICAL 5
#define SPDLOG_LEVEL_OFF 6 
```

The type of tracing information logged for each of these levels is evident from their names. Comparing with the different [verbose modes](https://oneapi-src.github.io/oneDNN/dev_guide_verbose.html?highlight=onednn_verbose#run-time-controls) defined for oneDNN, these logging levels can be aligned with the verbose modes as follows:

|   | `spdlog` Level        | oneDNN Verbose Mode                                              |
|---|-----------------------|------------------------------------------------------------------|
| 0 | SPDLOG_LEVEL_TRACE    | `all`                                                            |
| 1 | SPDLOG_LEVEL_DEBUG    | `dispatch`, `debuginfo=<level>`, `check`                         |
| 2 | SPDLOG_LEVEL_INFO     | `profile_create`, `profile_exec`, `profile`                      |
| 3 | SPDLOG_LEVEL_WARNING  | ---                                                              |
| 4 | SPDLOG_LEVEL_ERROR    | `error`                                                          |
| 5 | SPDLOG_LEVEL_CRITICAL | -                                                                |
| 6 | SPDLOG_LEVEL_OFF      | `none`                                                           |

With this alignment, the tracing information printed out for each verbose mode can be also logged at the aligned level. Obviously, the logging level here is determined from the value of the `ONEDNN_VERBOSE` variable. 
To exercise better control over the logged data, it is recommended to introduce another variable `ONEDNN_LOG_VERBOSE` which specifies the verbose mode for the logged data and also additionally allows the user to use the [`filter`]((https://oneapi-src.github.io/oneDNN/dev_guide_verbose.html?highlight=onednn_verbose#run-time-controls)) option for the verbose mode to filter the information being logged for supported components and primitives. 

## References

- **spdlog**: https://github.com/gabime/spdlog
- **fmt** lib: https://github.com/fmtlib/fmt 

(EOD)