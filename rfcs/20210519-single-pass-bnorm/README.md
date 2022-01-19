# Proposal to calculate mean & variance in batch normalization(BN) in single pass

## Requester
GPU Enabling team

## Motivation
Using new formula for calculating variance will result in up to 33% performance
improvement in BN forward training operation.

Currently BN performance is far below projections. Projections are based on
assumption that BN reads input tensor once and writes result once. Current
implementation requires BN to read input tensor 3 times. Proposed change
will decrease it to 2 reads. There is no known generic algorithm to decrease it
to 1 read, although such efficiency is possible for special cases where input
tensor is small enough.

## Proposal
Use the following formula to calculate variance:
```python
Var(X) = E(X^2) - (E(X))^2
```
where X = input tensor, E = average

This formula allows hardware to calculate both mean and variance in single pass
over input tensor.
For reference, the canonical formula is:
```python
Var(X) = E((X - Mean(X))^2)
```
The benefit of new formula is 25-33% performance improvement. BN kernel
is memory-bound, with new formula it will need to read input tensor 2 times
(1: mean&var, 2: normalize) instead of 3 times (1: mean, 2: var, 3: normalize)

This formula will be used whenever BN kernel calculates mean and
variance, which means forward propagation in training. However, with
non-standard combination of flags, oneDNN's BN implementation may be
configured to calculate those two values in inference as well.

## Design
Algorithm for calculating variance will be implemented as:
```python
Sum = 0
SumOfSquares = 0
for each x in X:
	Sum += x
	SumOfSquares += (x*x)
Variance = SumOfSquares/N - (Sum/N)^2
Variance = max(0, Variance)
```
where N = number of items in given channel in tensor.

In the formula there's a subtraction of two large values which is vulnerable
to catastrophic cancellation. Rounding errors in the two big numbers may be
large in relation to result of the formula. In order to minimize rounding
errors:
- Sum and SumOfSquares will be stored in fp32 precision, regardless of input
	data type
- Kahan summation algorithm [[#1]][1] will be used

The last line, max( ) is there to make sure rounding errors won't cause
variance to go negative. Further in BN calculations square root is taken
from variance. Negative value would result in NaN.

New formula is considered experimental for now. By default the old one will be
used. There needs to be a way for user to choose formula.

### Enable through build flag and environment variable
A new build flag `DNNL_EXPERIMENTAL` is introduced, disabled by default.
When built with this flag enabled, oneDNN will read environment variable
`DNNL_EXPERIMENTAL_BNORM_STATS_ONE_PASS` and will select BN formula
according to that flag. Default is the slower but numerically stable formula.

Pros:
- API/ABI compatibility
- Simplest to implement and use
- Clearly defined default state

Cons:
- No fine-tuning per instance of BN primitive
- Harder to reproduce bugs related to BN: value of this flag will need to be
 reported in bug reports to allow reproduction.
 - Inconvenient long-term. State of oneDNN will have a dependency which is not
 obvious to users.

## Layer normalization (LN)
BN and LN primitives work according to the same principles and
formulas. Only difference is in dimension across which the normalization is
performed. Therefore LN's performance may benefit from using the new
formula for calculating mean & variance.

## Validation
New formula was tested with RN50 and it trained to SotA. Further tests with
broader set of topologies are needed. Until such tests are performed, new
formula should be optionally available but disabled by default. The point
of this request is to make new formula available for further validation, while
external customers can keep using the old and proven formula.

## References

1. [Kahan summation algorithm][1]

[1]: https://en.wikipedia.org/wiki/Kahan_summation_algorithm
