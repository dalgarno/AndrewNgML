# Week 2 notes
## Overview
Regression with multiple features.

# Feature normalisation
Hypothesis is now $$ h(x) = \theta_0 + \theta\_1 x\_1 + ... \theta\_n x\_n $$.

Can write this with vector notation: $$\Theta^T \mathbf{x}$$

Feature scaling. Make sure features are on a similar scale. Get every feature approximately in the range $$-1 <= x <= 1 $$. Try and also make features zero mean, except for $$x\_0$$. 

Typically this is done by:

$$ x\_i = \frac{x\_i - \mu\_i}{s\_i} $$

Where $$\mu\_i$$ is the mean of the $$i^{th}$$ feature, and $$s\_i$$ is the standard deviation or range (max-min) of the feature.

# Learning rate
Plot J(\mathbf{\theta}) over each iteration. Should see it converge. If not, learning rate is probably too large. Want to find a learning rate that is not too big or too small.

# Feature selection
Not restricted to just the features available. Consider a house with width and height. We could combine these into one feature: area, by multiplying them together.

# Polynomial regression
Can incorporate polynomials by including features like $$x, x^2, x^3 $$ etc. However, these take on extreme values at large and small values of x. Also, feature normalisation is important as the ranges of the different powers change enormously.

