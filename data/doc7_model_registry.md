# Model Registry

A model registry is a centralized system for storing, versioning, and managing
the lifecycle of machine learning models. It serves as the source of truth for
which models exist, which versions are deployed where, and what metadata is
associated with each model.

## What a Model Registry Stores

For each model, the registry stores the serialized model artifact, a version
identifier, and metadata. Metadata typically includes training data version,
training code version, training hyperparameters, evaluation metrics, model
signature (expected input and output schema), and the identity of who trained
and approved the model.

The registry also tracks lifecycle stage. Common stages are None (just
registered), Staging (under evaluation), Production (serving live traffic),
and Archived (retired). Stage transitions can require approval.

## Relationship to Other MLOps Components

A model registry sits between training pipelines and deployment systems.
Training pipelines register candidate models. Deployment systems pull models
from the registry by version or by stage. This decoupling means training and
serving can evolve independently.

The model registry also connects to monitoring. When a model in production
shows degraded performance, you can trace back to the exact registry entry,
including training data version and training code. This closes the loop
between training and production observability.

## Popular Implementations

MLflow Model Registry is the most widely used open-source option. It
integrates tightly with MLflow Tracking, which logs experiments and metrics.
Models can be registered directly from tracking runs.

Weights and Biases Model Registry is part of the W&B platform and offers
richer collaboration features, including model cards and lineage tracking.

SageMaker Model Registry and Vertex AI Model Registry are cloud-provider
offerings with tight integration into their respective deployment services.

## Model Registry vs Model Store

The terms are sometimes used interchangeably but have a subtle distinction. A
model store is any system that stores model artifacts. A model registry adds
lifecycle management, versioning semantics, and governance on top. Every
registry is a store, but not every store is a registry.

## Governance Benefits

For regulated industries, the registry is where compliance evidence lives. It
records who approved a model for production, what metrics it achieved on
which evaluation set, and a complete history of deployments and rollbacks.
This is essential for audit trails and model risk management.
