# oneDNN Design Documents / RFCs

The purpose of the RFC process is to communicate the intent to make
library-wide changes, get the feedback prior to the actual implementation,
increase the transparency on why and how the decisions are made, and improve
the alignment between different teams involved in oneDNN development.

This branch contains design documents (RFCs) that are approved for
implementation in oneDNN.

## Document Style

The design documents are stored in the `rfcs` directory.

- Each RFC is stored in a separate subdirectory
  `rfcs/<YYYMMDD>-descriptive-but-short-name`.

  - There must be a `README.md` file that contains the main RFC itself.

  - The directory can contain other supporting files, such as images,
    tex formulas, and sub-proposals / sub-RFCs.

- The RFC is written in markdown. The width of the text should be limited by
  80 symbols, unless there is a need to violate this rule, e.g. because of
  long links or wide tables.

- The document structure should follow the [RFC template](rfcs/template.md).

  - It is also recommended to read through existing RFCs to better understand
    the general writing style and required elements.

## RFC Ratification Process

Before submitting an RFC, it might be helpful to have a preliminary discussion
on the idea with oneDNN contributors. Regular GitHub issues could be used for
the discussion.

The RFC life-cycle is:

1. A design author writes up a design and opens a PR to the
   [`rfcs`](https://github.com/oneapi-src/oneDNN/tree/rfcs) branch. The PR
   should be labeled as an
   [`RFC`](https://github.com/oneapi-src/oneDNN/labels/RFC) and contain the
   short description of the objective.
   It is quite handy if the PR description contains a link to the RFC
   directory that will automatically render the RFC's `README.md`.

2. RFCs from external parties require a sponsor from oneDNN maintainers to
   guide them to completion (either approval or rejection). The sponsor will be
   assigned to the PR.

3. RFC discussions take place in comments sections in the PRs. RFC owners
   continuously update the design until all the issues are resolved.

4. To accept the RFC, at least two approvals from oneDNN maintainers (excluding
   the author of the proposal) are required. Once general consensus is reached
   and the approvals are in place, the RFC is merged to the branch.
