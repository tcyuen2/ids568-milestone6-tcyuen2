# Model Deployment Strategies

Deploying a machine learning model to production involves more than just
loading it behind an API. The deployment strategy determines how much risk you
take when introducing a new model version and how quickly you can roll back
if something goes wrong.

## Blue-Green Deployment

In a blue-green deployment, two identical production environments are
maintained. The blue environment serves current traffic. The green environment
receives the new model version. Once the new version passes smoke tests,
traffic is switched entirely from blue to green in a single cutover.

The advantage is immediate rollback: if the new version fails, flip traffic
back to blue. The disadvantage is cost: you pay for double infrastructure
during the transition, and you cannot gradually validate the new model on real
traffic before full exposure.

## Canary Deployment

In a canary deployment, the new model version is exposed to a small fraction
of traffic, typically 1 to 5 percent. Key metrics are monitored. If the metrics
look healthy, traffic to the new version is gradually increased: 10 percent,
25 percent, 50 percent, 100 percent. If metrics degrade at any step, traffic
is rolled back.

Canary deployment is the most common strategy for high-stakes ML systems
because it combines gradual exposure with monitoring. The tradeoff is slower
rollout and more complex traffic routing infrastructure.

## Shadow Deployment

In a shadow deployment, the new model receives a copy of production traffic
but its predictions are not returned to users. The predictions are logged for
offline comparison with the current production model. This is useful when you
want to validate that the new model behaves sensibly on real data without any
user impact.

Shadow deployment cannot detect problems caused by serving the new model's
predictions (for example, a feedback loop where user behavior reacts to worse
predictions). It is complementary to, not a replacement for, canary.

## Rolling Deployment

Rolling deployment replaces instances of the old version with the new version
gradually, one at a time or in batches. Unlike canary, there is no explicit
traffic split; instances are replaced until all are running the new version.

This is the default in Kubernetes Deployments. It is cheaper than blue-green
but slower to roll back, because rollback also happens one instance at a time.

## Choosing a Strategy

For low-risk changes (small model updates, non-critical features), rolling
deployment is fine. For high-risk changes (new model architecture, major
retraining), canary is preferred. Blue-green makes sense when you need clean
cutover semantics, for example when schema migrations happen at the same time.
