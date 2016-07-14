---
layout: post
title: "How (not) to invert a 50k x 50k matrix in the JVM"
date: 2020-11-23
published: false
visible: 0
categories: ml systems
---

<p>
I started out with a very simple task last week. I have a relatively small collection of blurry images (3072 pixels) from the cifar data set (50k images). And I want to classify them (from one of 10 classes).
</p>

<img src="/assets/images/cifar_frog.png" width="256" id="cifar_frog">

<p>
I can stack all these images into a $50k \times 3072$ matrix and call it $D$
Since pixel space isn't particularly good for classification, an oracle gives me a featurization function $\Phi_{d}$, this could be anything from a random matrix multiply to a trained convolutional neural network (CNN).
<p>

<p>
The amount of output features is controlled by d and I want to see how for a given featurization method how the output dimension affects classification performance.
For simplicity I will be using the squared loss.
</p>



We can let $A = \Phi_{d}(D)$, note A is now $50000 \times d$, and let $b$ the matrix of one hot encoded labels for our data set.

I want to minimize $\|\|A - b\|\|_{2} + \lambda\|\|w\|\|$

Simple right?

Calculus tells us $w^{*} = (A^{T}A + \lambda I_{d})^{-1}A^{T}b$ and we are done.

And our prediction vector would be $\hat{b} = A(A^{T}A + \lambda I_{d})^{-1}A^{T}b$

Now what problems does this present us if we actually want to compute this quantity? Well if d is very large (perhaps over $100000$), $(XX^{T} + \lambda I_{d})^{-1}$ becomes very hard to compute (since the inverse scales as $d^{3}$).

But we can use a [useful matrix identity](https://people.eecs.berkeley.edu/~stephentu/blog/matrix-analysis/2016/06/03/matrix-inverse-equality.html) to rewrite our problem into
$\hat{b} = AA^{T}(AA^{T} + \lambda I_{50k})^{-1}b$

Now I never have to invert anything greater than 50k right?



<script>
document.getElementById("cifar_frog").className = "";
</script>

