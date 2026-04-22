# Feature Stores

A feature store is a centralized data management system for storing, serving, and
sharing features used in machine learning models. It solves the problem of
training-serving skew, where the feature values a model sees during training
differ from those it encounters in production. This skew is one of the leading
causes of silent model degradation in deployed systems.

## Core Problems Feature Stores Solve

Training-serving skew occurs when the data pipeline that generates features for
training is different from the one used at inference time. A feature store
provides a single source of truth: the same transformation logic is applied in
both contexts, guaranteeing consistency.

Feature reuse is another problem. Without a feature store, each team rewrites
similar feature engineering logic, duplicating work and creating subtle
inconsistencies. A feature store lets teams publish features once and reuse them
across many models.

Point-in-time correctness is critical for avoiding label leakage. A feature
store tracks the exact value of a feature at the time a prediction was made, so
that historical training sets accurately reflect what the model would have seen
in production.

## Online and Offline Stores

Feature stores typically have two components. The offline store holds large
volumes of historical feature data used for training and batch scoring. It is
optimized for throughput and often backed by a data warehouse or object storage.

The online store holds the most recent feature values and is optimized for
low-latency reads during real-time inference. It is typically backed by a
key-value store like Redis or DynamoDB, with single-digit millisecond lookups.

## Popular Implementations

Feast is an open-source feature store widely used in production. Tecton is a
commercial feature store built by former Uber engineers who worked on Michelangelo.
Databricks Feature Store and Vertex AI Feature Store are cloud-provider options.

The choice of feature store depends on scale, existing infrastructure, and
whether real-time inference is required.
