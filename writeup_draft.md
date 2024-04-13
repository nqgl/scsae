TLDR: SAEs apply some scaling to features to counteract the use of the bias for noise suppression. This scaling means that feature magnitudes are only reproduced for a particular

Here's the technique and motivation:

An observation: The encoder bias somewhat serves two functions:

*   Translation behavior: conditional on a feature activating, it adds a bias to feature activation magnitude.
*   Inhibition: if the pre-bias feature activation plus the bias is less than zero, the feature does not activate

I think this works fine for binary features, but I see a potential issue when features have variable magnitude activations.

Toy Motivating Example
======================

Considering a single feature direction being reconstructed by an autoencoder, here's a toy model of a situation I think it will struggle with.

*   Feature \\(v\_i = s \\hat{v\_i}\\) where \\(\\hat v_i\\) is the unit vector in the feature direction and \\(s_i\\) is the feature's magnitude \[\[ may need to improve this notation? \]\] \[\[probably should just use \\(||v_i||\\) for the magnitude, oops\]\] 
*   We'll assume for simplicity that when this feature is active, all other active features are orthogonal to it, so conditional on \\(v_i\\) being active, the autoencoder does not need to compensate for interference from other features when reconstructing \\(v_i\\)

A standard untied SAE looks like:

\\\[SAE(x) = ReLU((x-b\_d) W\_e + b\_e) W\_d + b_d\\\]

The decoder bias does not have much relevance for this scenario effect, so we will consider SAEs of the form:

\\\[SAE(x) = ReLU(x W\_e + b\_e) W_d\\\]

With input dimension \\(D\\) and dictionary size \\(F\\), let

\\\[W\_e = \\left\[ w\_e^1, w\_e^2, ..., w\_e^F \\right\] \\text{ and } W\_d = \\left\[ w\_d^1, w\_d^2, ..., w\_d^F \\right\]^T\\\]

Looking at just the part of the SAE that reconstructs the \\(v\\) feature:

\\\[SAE\_i(x) = ReLU(x\\cdot w\_e^i + b\_e^i) \\cdot w\_d^i\\\]

**Noise-free solution:**  
If there is never any noise in the direction of \\(v_i\\), then the solution is simple:

\\\[w\_e^i = w\_d^i = \\hat v\_i;\\; b\_e^i = 0\\\]

This works for all activation magnitudes.

With noise, constant activation magnitude:
------------------------------------------

Now, suppose

*   when \\(v_i\\) is active, it activates with magnitude \\(s_i\\)
*   when \\(v_i\\) is not active there may be noise in the direction of \\(v_i\\), up to a maximum magnitude \\(n_i\\) of noise in this direction

To filter out this noise, we need to use a negative inhibitory bias

\\\[b\_e^i = -n\_i ||w_e^i||\\\]

The weight directions are unchanged, but the encoder magnitude will need to change:

\\\[\\frac{w\_e^i}{||w\_e^i||} = w\_d^i = \\hat v\_i\\\]

to reconstruct the activation with magnitude \\(s_i\\), we need:\\(\\)

\\\[x\\cdot w\_e^i + b\_e^i = s_i\\\]\\\[s\_i||w\_e^i|| - n\_i ||w\_e^i|| = s_i\\\]\\\[||w\_e^i|| (s\_i - n\_i) = s\_i\\\]\\\[||w\_e^i|| = \\frac{s\_i}{(s\_i - n\_i)}\\\]

This always filters out the noise, while perfectly reconstructing the feature.

**Numerical example:**

For example, if we have

*   noise threshold \\(n_i = 0.5\\)
*   feature magnitude \\(s_i=1\\)

 we get that 

\\\[||w\_e^i|| = \\frac{s\_i}{(s\_i - n\_i)} = \\frac{1}{1 - 0.5} = 2\\\]\\\[b\_e^i = -n\_i ||w_e^i|| = -0.5 * 2 = -1\\\]

This filters out all the noise, while maintaining \\(SAE\_i(v\_i) = v_i\\)

\\(\\)  
 

Variable magnitudes problem:
----------------------------

1.  What if we use this part of the SAE to detect the feature, but it has variable magnitude?

The derivative wrt. feature scale is not 1 (which would be "nicer", IMO), it is now \\(\\frac{s\_i}{(s\_i - n_i)}\\)

so for some feature activation vector \\(v\_i' = mv\_i\\), when \\(m \\neq 1\\) we no longer get perfect reconstruction:

\\\[SAE\_i(mv\_i) = v\_i + \\frac{(m-1)}{s\_i -n\_i}\\hat v\_i = v\_i + \\frac{m-1}{s\_i - n\_i}v\_i = (1+\\frac{m-1}{s\_i - n\_i}) v_i\\\]

**Numerical example:**

If we process a feature with magnitude 2 using the previous \\(n\_i = 0.5;\\; s\_i=1\\) case, it will produce a feature of size  \\(ReLU(||v\_i'|| ||w\_e^i|| + b_e^i)  = ReLU(2 \\cdot 2 - 1) = 3\\) 

A solution (I think)
====================

I want us to be calculating some function that is capable of filtering out noise while also fulfilling for an above-threshold activations:

*   is capable of filtering out noise
*   \\(SAE\_i(v\_i + q\\hat v\_i) = (1 + q) \\cdot SAE\_i(v_i)\\) \[\[make consistent with the previous \\(SAE\_i(mv\_i)\\) example -- I think adding a component is more elegant and intuitive than scaling by \\(m\\)\]\]
*   \\(SAE\_i(v\_i) = v_i\\)

Our current way of building SAEs is not able to fulfill this, and it is partly because the bias is fulfilling the two roles of inhibition and activation scaling. The solution I have is to use it only for inhibition. Eg,

\\\[SCSAE\_i(x) = \\begin{cases} x \\cdot w\_e^i & \\mbox{if } x\\cdot w\_e^i + b\_e^i > 0 \\\ 0 & \\mbox{otherwise} \\end{cases}\\\]

This has an issue, which is that it's gradient wrt. \\(b_e^i\\) is zero. To deal with this, we computer this function on the forward pass, but backprop as though this were a regular SAE*.

\* I am also experimenting with backpropagating to \\(b_e^i\\) as though it were *multiplied* by the output, and am getting variable success with this -- it learns good solutions more quickly but then often exhibits degenerate behavior as time goes on, eg. U-shaped loss curves.

Results
=======

> Pretty darn good

Training Notes
==============

**Betas:** 

*   Just (0.9, 0.999) seems fine, as per Anthropic's March update [https://transformer-circuits.pub/2024/march-update/index.html](https://transformer-circuits.pub/2024/march-update/index.html)
*   Maybe I will run too b2=0.99, as I had previously observed things that made me think smaller betas were better (though, this may have been just 

**Resampling:**

  
  
**Normalization:**  
I'll use the Anthropic-style l2-norm normalization rather than my old 'divide by standard deviation according to samples from the data' normalization method

Other scattered thoughts/notes
==============================

**Necessity of L1 or some other cause of consistent underestimates**

The gradients that go to the bias need to correspond to a signal appropriate for their "gating" role. Therefore, it is important for the gradient to the bias to be positive when activation was appropriate and negative when the feature should not have activated. This is also somewhat of the case with a normal bias, but easier to get wrong here, since the bias will not be added to the activation.  
In order to achieve this correspondence between gradient direction and gating correctness, it's important that the encoder weights consistently slightly undershoot their ideal activation level. Since we're already using an L1 penalty, we kind of get this by default, but it's interesting to note. I predict that training might not work if I set the L1 coefficient to 0, since the gradients to the bias for a correct activation would be closer to 50/50 for whether they are negative or positive. 

**Resampling:**  
I'm doing the original Anthropic resampling. Neel Nanda suggested this is the current best established method, and it sounds like Anthropic found Ghost Grads don't work as well outside of 1-layer models.  
 

For the SCSAEs I will be reducing the scale of the encoder vector, as at 0.2 this is what happens :) I expected this would be necessary due to the differences in the architecture

![](https://39669.cdn.cke-cs.com/rQvD3VnunXZu34m86e5f/images/097cc5420761fa3ce845229fb86bae8afaa11e4dc2c60210.png)

re-warmup after resample: In addition to resetting Adam, I re-warmup Adam after a resample for 2k steps (2k to match the \\(\\frac{2}{1-\\beta_2}\\) num steps recommended in [https://arxiv.org/pdf/1910.04209.pdf](https://arxiv.org/pdf/1910.04209.pdf) for our largest value of \\(\\beta_2\\))