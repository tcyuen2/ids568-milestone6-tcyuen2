# Model Monitoring and Drift Detection

Monitoring deployed machine learning models is essential because model quality
degrades over time. Unlike traditional software, which fails in obvious ways,
ML models can silently produce worse predictions while all infrastructure
metrics look healthy.

## Data Drift vs Concept Drift

Data drift is a change in the distribution of the input features. For example,
a fraud detection model might see a new type of transaction pattern that was
rare during training. The model still maps inputs to outputs correctly, but the
inputs themselves have changed.

Concept drift is a change in the relationship between inputs and outputs. The
same input features now imply a different label. For example, customer churn
behavior might shift after a competitor changes their pricing, so the same
customer profile now has a different probability of churning.

Data drift can often be detected without labels by comparing the distribution
of production inputs to training inputs. Concept drift requires ground-truth
labels and is therefore harder to detect in real time.

## Detection Techniques

Population Stability Index (PSI) is a common metric for detecting drift in
individual features. PSI values above 0.1 suggest moderate drift; above 0.25
suggest significant drift.

Kolmogorov-Smirnov tests compare the cumulative distributions of two samples
and are useful for continuous features. Chi-square tests work well for
categorical features. Maximum Mean Discrepancy is a kernel-based method that
detects multivariate distributional differences.

For concept drift, prediction-based monitors track the distribution of model
outputs. If the predicted positive rate suddenly shifts, that may indicate
either data drift or concept drift.

## What to Monitor

Operational metrics (latency, throughput, error rate) are necessary but not
sufficient. You also need input monitoring (feature distributions, missingness,
schema violations), output monitoring (prediction distributions, confidence
distributions), and quality monitoring (accuracy, precision, recall when
ground truth is available).

Alerting should be tiered: page for serving outages, email for drift warnings,
dashboard-only for low-priority anomalies.
