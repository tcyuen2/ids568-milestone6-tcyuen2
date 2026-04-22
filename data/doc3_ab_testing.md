# A/B Testing for Machine Learning Models

A/B testing is the gold standard for evaluating whether a new machine learning
model actually improves outcomes in production. Offline metrics like accuracy
or AUC can be misleading because they do not account for user behavior,
interaction effects, or the specific business objective the model is supposed
to drive.

## Stages of an ML A/B Test

The first stage is hypothesis formulation. You must define the business metric
you want to improve (click-through rate, revenue per user, churn reduction) and
the minimum detectable effect you care about. This determines sample size.

The second stage is traffic splitting. Users are randomly assigned to the
control group (current model) or treatment group (new model). Randomization
must be stable per user to avoid exposing the same user to both versions.

The third stage is guardrail metrics. Besides the primary metric, track secondary
metrics that should not regress. If your new ranking model improves clicks but
tanks revenue, that is a failed experiment.

The fourth stage is statistical analysis. Compute confidence intervals and
p-values for the difference between groups. Be cautious of peeking at results
before the experiment is complete, which inflates false positive rates.

## Common Pitfalls

Sample ratio mismatch occurs when the actual split differs from the intended
split, often due to logging bugs or bot filtering. Always check that the ratio
matches what you configured before trusting results.

Novelty effects can make a new model look better simply because users react to
the change. Run experiments long enough to see steady-state behavior, typically
at least two weeks.

Interference between arms happens when users in different arms can affect each
other, for example in marketplace settings where a change in one arm's ranking
affects inventory seen by the other arm. Cluster randomization can mitigate this.

## Alternatives When A/B Testing Is Impractical

Shadow deployment runs the new model in parallel with the old one without
serving its predictions to users. This is useful for safety validation.

Interleaved experiments mix ranked results from two models within a single user
session, reducing variance compared to between-user tests.
