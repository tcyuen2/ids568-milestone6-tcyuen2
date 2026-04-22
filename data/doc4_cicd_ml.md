# CI/CD for Machine Learning

Continuous Integration and Continuous Deployment for machine learning extends
traditional software CI/CD with additional stages specific to data and models.
ML CI/CD is fundamentally more complex than software CI/CD because the behavior
of the system depends not just on code but also on data and trained model
artifacts.

## Why ML CI/CD Is More Complex

Traditional CI/CD tests code. If the unit tests pass and integration tests
pass, you ship. ML systems have three independent inputs that can change
behavior: code, data, and model hyperparameters. A pipeline that tests only
code will miss data-induced regressions.

Model behavior is non-deterministic in ways that code usually is not. The same
training code run on the same data can produce slightly different models due
to random initialization, GPU nondeterminism, or parallelism. Tests must
account for this.

Models degrade silently. A traditional bug throws an exception or produces
obviously wrong output. A degraded model produces subtly worse predictions
that only show up in aggregate metrics.

## Components of ML CI/CD

Data validation runs on every pipeline execution. It checks schema conformance,
feature distributions, and known invariants. TensorFlow Data Validation and
Great Expectations are common tools.

Model training is automated and triggered by code changes, data changes, or
scheduled retraining. The pipeline produces a candidate model artifact along
with training metrics.

Model evaluation compares the candidate model against the currently deployed
model on a held-out evaluation set. It checks not just aggregate metrics but
also slice metrics: does the new model do better across all customer segments,
or is it trading gains in one segment for losses in another?

Model deployment promotes the candidate to production if it passes evaluation.
This can be gated by manual approval or fully automated.

## Testing at Multiple Levels

Unit tests verify individual preprocessing functions and model components.
Integration tests verify the full training pipeline runs end-to-end. Behavioral
tests check that the trained model produces sensible predictions on known
examples. Invariance tests verify the model is insensitive to transformations
that should not affect predictions.

Tools like Kubeflow Pipelines, MLflow, and Metaflow provide infrastructure for
orchestrating ML pipelines with these properties.
