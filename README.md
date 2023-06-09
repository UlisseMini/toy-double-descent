# toy double descent

Reproducing the core of Anthropic's "[Superposition, Memorization, and Double Descent](https://transformer-circuits.pub/2023/toy-double-descent/index.html)" paper.

They found that for limited training sets overfitting can be seen as *datapoints* being in superposition while *features* aren't:

![](images/features-vs-hidden-vectors.png)

We see the same effect in our reproduction, here for $T=10$:

![](images/T10-feat-vs-data.png)

Here's a cute gif of the evolution over time:

![](images/T10-feat-vs-data.gif)

Henighan et. al also demonstrated *double descent*, the difference

![](images/double-descent.png)

We see the same effect in our reproduction, for $T=1000$ and $n*S = 10$ (expected number of features per datapoint) we get:

(TODO: add result. big runs take a long time on my laptop...)

