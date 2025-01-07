# Introduction

This document defines roles available in oneDNN project and includes the current
list of Code Owners and Maintainers for the project.

# Roles and responsibilties

oneDNN project defines three main roles:
 * [Contributor](#contributor)
 * [Code Owner](#code-Owner)
 * [Maintainer](#maintainer)

These roles are merit based. Refer to the corresponding section for specific
requirements and the nomination process.

## Contributor

A Contributor invests time and resources to improve oneDNN. Anyone can become
a Contributor by bringing value in one of the following ways:
  * Answer questions from community members.
  * Submit feedback to design proposals.
  * Review and/or test pull requests.
  * Test releases and report bugs.
  * Contribute code, including bug fixes, features implementations,
and performance optimizations.
  * Contribute design proposals.

Responsibilities:
  * Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
  * Follow the project [contributing guidelines](CONTRIBUTING.md).

Privileges:
  * Eligible to become a Code Owner.

## Code Owner

A Code Owner has responsibility for a specific project component or a functional
area. Code Owners are collectively responsible, with other Code Owners, 
for developing and maintaining their component or functional areas, including
reviewing all changes to their their areas of responsibility and indicating
whether those changes are ready to merge. They have a track record of
contribution and review in the project.

Responsibilities:
  * Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
  * Follow and enforce the project [contributing guidelines](CONTRIBUTING.md).
  * Co-own with other code owners a specific component, including contributing
    bug fixes, implementing features, and performance optimizations.
  * Review pull requests in their specific areas of responsibility.
  * Monitor testing results and flag issues in their specific areas of
    responsibility.
  * Support and guide Contributors.

Requirements:
  * Experience as Contributor for at least 6 months.
  * Commit at least 25% of working time to the project.
  * Track record of accepted code contributions to a specific project component.
  * Track record of contributions to the code review process.
  * Demonstrated in-depth knowledge of the architecture of a specific project
    component.
  * Commits to being responsible for that specific area.

Privileges:
  * PR approval counts towards approval requirements for a specific component.
  * Can promote fully approved Pull Requests to the `main` branch.
  * Can recommend Contributors to become Code Owners.
  * Eligible to become a Maintainer.

The process of becoming a Code Owner is:
1. A Contributor is nominated by opening a PR modifying the MAINTAINERS.md file
including name, Github username, and affiliation.
2. At least two specific component Maintainers approve the PR.

## Maintainer
Maintainers are the most established contributors who are responsible for the 
project technical direction and participate in making decisions about the
strategy and priorities of the project.

Responsibilities:
  * Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
  * Follow and enforce the project [contributing guidelines](CONTRIBUTING.md)
  * Co-own with other component Maintainers on the technical direction of a specific component.
  * Co-own with other Maintainers on the project as a whole, including determining strategy and policy for the project.
  * Suppport and guide Contributors and Code Owners.

Requirements:
  * Experience as a Code Owner for at least 12 months.
  * Commit at least 25% of working time to the project.
  * Track record of major project contributions to a specific project component.
  * Demonstrated deep knowledge of a specific project component.
  * Demonstrated broad knowledge of the project across multiple areas.
  * Commits to using priviledges responsibly for the good of the project.
  * Is able to exercise judgment for the good of the project, independent of
    their employer, friends, or team.

Privileges:
  * Can represent the project in public as a Maintainer.
  * Can promote Pull Requests to release branches and override mandatory
  checks when necessary.
  * Can recommend Code Owners to become Maintainers.

Process of becoming a maintainer:
1. A Maintainer may nominate a current Reviewer to become a new Maintainer by 
opening a PR against MAINTAINERS.md file.
2. A majority of the current Maintainers must then approve the PR.

# Code Owners and Maintainers List

## Core (API, Architecture, Tests)

Team: @oneapi-src/onednn-arch

| Name               | Github ID             | Affiliation       | Role       |
| -----------------  | --------------------- | ----------------- | ---------- |
| Denis Samoilov     | @densamoilov          | Intel Corporation | Maintainer |
| Dmitry Zarukin     | @dzarukin             | Intel Corporation | Maintainer |
| Mourad Gouicem     | @mgouicem             | Intel Corporation | Maintainer |
| Vadim Pirogov      | @vpirogov             | Intel Corporation | Maintainer |
| Ankit Manerikar    | @avmanerikar          | Intel Corporation | Code Owner |
| Stefan Palicki     | @spalicki             | Intel Corporation | Code Owner |

## Graph API

Team: @oneapi-src/onednn-graph

| Name               | Github ID             | Affiliation       | Role       |
| ------------------ | --------------------- | ----------------- | ---------- |
| Tao Lv             | @TaoLv                | Intel Corporation | Maintainer |
| Zhitao Wang        | @wzt1997              | Intel Corporation | Code Owner |
| Jiexin Zheng       | @Jiexin-Zheng         | Intel Corporation | Code Owner |
| Shaojie Cui        | @ShanSimu             | Intel Corporation | Code Owner |
| Yonghao Gu         | @gyhintel             | Intel Corporation | Code Owner |
| Rong Zhang         | @rongzha1             | Intel Corporation | Code Owner |
| Xiang Guo          | @xiang1guo            | Intel Corporation | Code Owner |
| Yixin Bao          | @ElaineBao            | Intel Corporation | Code Owner |

## CPU Engine

### x64

Team: @oneapi-src/onednn-cpu-x64

| Name               | Github ID             | Affiliation       | Role       |
| ------------------ | --------------------- | ----------------- | ---------- |
| Andrey Kalinin     | @ankalinin            | Intel Corporation | Maintainer |
| Tatyana Primak     | @tprimak              | Intel Corporation | Maintainer |
| Alexey Makarevich  | @amakarev             | Intel Corporation | Code Owner |
| David Eberius      | @davideberius         | Intel Corporation | Code Owner |
| Stefan Palicki     | @spalicki             | Intel Corporation | Code Owner |
| Tomasz Czeszun     | @tczeszun             | Intel Corporation | Code Owner |
| Xuxin Zeng         | @xuxinzen             | Intel Corporation | Code Owner |

### AArch64

Team: @oneapi-src/onednn-cpu-aarch64

| Name               | Github ID             | Affiliation       | Role       |
| ------------------ | --------------------- | ----------------- | ---------- |
| Crefeda Rodrigues  | @cfrod                | Arm Ltd           | Code Owner |
| David Svantesson   | @davsva01             | Arm Ltd           | Code Owner |
| Jonathan Deakin    | @jondea               | Arm Ltd           | Code Owner |
| Hamza Butt         | @theComputeKid        | Arm Ltd           | Code Owner |
| Radu Salavat       | @Radu2k               | Arm Ltd           | Code Owner |
| Siddhartha Menon   | @Sqvid                | Arm Ltd           | Code Owner |
| Sunita Nadampalli  | @snadampal            | Amazon.com, Inc.  | Code Owner |

### OpenPOWER (PPC64)

Vacant. Maintained by Core team.

### IBMz (s390x)

Vacant. Maintained by Core team.

### RISC-V

Vacant. Maintained by Core team.

### Loongarch64

Vacant. Maintained by Core team.

## GPU Engine

### Intel

Team: @oneapi-src/onednn-gpu-intel

| Name               | Github ID             | Affiliation       | Role       |
| ------------------ | --------------------- | ----------------- | ---------- |
| Eugene Chereshnev  | @echeresh             | Intel Corporation | Maintainer |
| Konstantin Arturov | @karturov             | Intel Corporation | Maintainer |
| Peter Caday        | @petercad             | Intel Corporation | Maintainer |
| Andy Kassen        | @atkassen             | Intel Corporation | Code Owner |
| Daniel Youssif     | @dyoussif             | Intel Corporation | Code Owner |
| Haleema Sadia      | @h-sadia              | Intel Corporation | Code Owner |
| Andrey Guskov      | @hidefromkgb          | Intel Corporation | Code Owner |
| Gallagher Pryor    | @pv-pterab-s          | Intel Corporation | Code Owner |
| Kealan Barbieri    | @kealan-barbieri      | Intel Corporation | Code Owner |
| Roy Oursler        | @rjoursler            | Intel Corporation | Code Owner |
| Simon Ewing        | @Simonsays095         | Intel Corporation | Code Owner |
| Sergey Kazakov     | @skazakov1            | Intel Corporation | Code Owner |
| Stefen Yurkevich   | @syurkevi             | Intel Corporation | Code Owner |
| Umar Arshad        | @umar456              | Intel Corporation | Code Owner |

### NVIDIA, AMD, and generic GPU

Teams:
* @oneapi-src/onednn-gpu-nvidia
* @oneapi-src/onednn-gpu-amd
* @oneapi-src/onednn-gpu-generic

| Name               | Github ID             | Affiliation       | Role       |
| ------------------ | --------------------- | ----------------- | ---------- |
| Anton Mitkov       | @ShanoToni            | Codeplay Software | Code Owner |
| Mehdi Goli         | @mehdi-goli           | Codeplay Software | Code Owner |
| Svetlozar Georgiev | @sgeor255             | Codeplay Software | Code Owner |

## Support functions

### Documentation

Team: @oneapi-src/onednn-doc

| Name               | Github ID             | Affiliation       | Role       |
| ------------------ | --------------------- | ----------------- | ---------- |
| Vadim Pirogov      | @vpirogov             | Intel Corporation | Maintainer |
| Ranu Kundu         | @ranukund             | Intel Corporation | Code Owner |
| Tao Lv             | @TaoLv                | Intel Corporation | Code Owner |

### DevOps

Team: @oneapi-src/onednn-devops

| Name               | Github ID             | Affiliation       | Role       |
| ------------------ | --------------------- | ----------------- | ---------- |
| Sergey Razumovskiy | @srazumov             | Intel Corporation | Maintainer |
| Vadim Pirogov      | @vpirogov             | Intel Corporation | Maintainer |
| Hamza Butt         | @theComputeKid        | Arm Ltd           | Code Owner |

### Release management

| Name               | Github ID             | Affiliation       | Role       |
| ------------------ | --------------------- | ----------------- | ---------- |
| Tatyana Primak     | @tprimak              | Intel Corporation | Maintainer |
| Vadim Pirogov      | @vpirogov             | Intel Corporation | Maintainer |
