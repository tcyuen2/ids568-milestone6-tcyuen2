# Data Versioning with DVC

Data Version Control (DVC) is an open-source tool for versioning datasets and
machine learning experiments alongside code. It solves the problem that Git
alone is poorly suited for managing large data files, which can be gigabytes
or terabytes in size.

## How DVC Tracks Data Versions

DVC does not store large files in Git. Instead, when you run dvc add
path/to/dataset, DVC computes a content hash of the file, moves the file to
a local cache, and creates a small metadata file with the .dvc extension in
your working directory. This .dvc file contains the hash and is committed to
Git.

To share data with collaborators, DVC uses a remote storage backend such as
S3, Google Cloud Storage, Azure Blob Storage, SSH, or a shared filesystem.
Running dvc push uploads the actual data to the remote. Running dvc pull
downloads the data that matches the .dvc file in the current Git checkout.

When you check out a different Git branch or commit, DVC can restore the
corresponding version of the data by hash. This gives you reproducibility:
any commit in Git history corresponds to an exact snapshot of both code and
data.

## DVC Pipelines

DVC supports defining pipelines in a dvc.yaml file. Each stage declares its
inputs (dependencies), outputs, and the command to run. DVC tracks when
dependencies change and reruns only the affected stages, similar to make
but with content-based rather than timestamp-based invalidation.

A typical pipeline might have stages like prepare, featurize, train, and
evaluate. If you change only the training hyperparameters, DVC will reuse
cached outputs from the prepare and featurize stages.

## Experiments and Metrics

DVC supports lightweight experiments via dvc exp run. Each experiment is a
branch-like entity with its own parameter values and output metrics. You can
compare experiments with dvc exp show, which displays a table of parameters
and resulting metrics.

This is useful during model development when you want to try many
hyperparameter configurations without cluttering your Git history with commits
for each attempt.

## Alternatives

LakeFS provides Git-like semantics at the object-store level. Pachyderm offers
data-aware pipelines with built-in lineage tracking. Delta Lake and Apache
Iceberg are table formats that support time travel on big data workloads.
Each addresses a slightly different point in the data versioning design space.
