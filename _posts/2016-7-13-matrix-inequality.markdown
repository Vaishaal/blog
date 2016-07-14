---
layout: post
title: "Using a matrix inequality for (small-scale) image classification"
date: 2016-07-13
visible: true 
categories: ml
---

In this post I will walk through a concrete application of [a matrix inequality](http://people.eecs.berkeley.edu/~stephentu/blog/matrix-analysis/2016/06/03/matrix-inverse-equality.html) to speed up the training process of a simple image classification pipeline.

#### Background
I have a relatively small collection of blurry images (32 x 32 rgb pixels) from the cifar data set (50,000 images) from each of 10 classes. The task is to build a
model to classify these images.

For reference, the images look like this:
</p>
<img src="{{site.baseurl}}/assets/images/cifar_frog.png" width="256" id="cifar_frog">
<p>

#### Problem Formulation
We can represent each image as a vector in $R^{32 \times 32 \times 3}$
Then we stack all these images into a $50000 \times 3072$ matrix and call it $X$

We can let Y be a $50000 \times 10$ matrix of corresponding [one hot encoded](http://stackoverflow.com/questions/17469835/one-hot-encoding-for-machine-learning) image labels.

We denote $X_{i}$ to be the $ith$ row of $X$ (or the $ith$ image)

We denote $Y_{i}$ to be the $ith$ row of $Y$ (or the $ith$ image label)

<p>
Now the task is to build a <i>generalizable</i> map from $X_i \to Y_i$
</p>

#### Strawman

What is the dumbest map we can think of?

LINEAR!

That is we want to find a $W$ such that $Wx$ is close to $y$, the label vector.

Particularly we want the maximal index of $Wx$ to be the same as the maximal index
of $y$.

Now I wave my hands and claim that L2 error is a decent metric. So we can formulate
an optimization problem.

<p>
So lets try to minimize
$ \frac{1}{2} \|XW - Y\|_{F}^{2} + \lambda \|W\|^{2}_{F}$
</p>

Note I added a *penalty* term, this is very common.
In the derivation of the solution it will be clear why the penalty
term is necessary. Note the $_F$ simply means I will be using the [Frobenius norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm),
which means I'll treat the matrices XW and Y as large vectors and use the standard euclidean
norm.

#### Strawman solution
We can find the optimum solution with some matrix calculus:

First expand

$ Tr(W^{T}X^{T}XW) - Tr(X^{T}WY) + Tr(Y^{T}Y)  + \lambda Tr(W^{T}W)$

Note I converted the Frobenius norm to a trace

Then take derivative and set to 0.

Note trace and derivative commute

$ Tr(X^{T}XW) - Tr(X^{T}Y) + \lambda I_{d} Tr(W) = 0$

$ Tr(X^{T}XW) +  \lambda Tr(W) = Tr(X^{T}Y)$

Linearity of Trace

$ Tr((X^{T}X +  I_{d}\lambda) W) = Tr(X^{T}Y)$

This is satisfied when:

$W =  (X^{T}X +  I_{d}\lambda)^{-1}X^{T}Y$

Which would be our optimum. Since $d$ in this case is only $3072$ this is a quick and easy computation
that can be done in one line of python.

```
>>> W = scipy.linalg.solve(X, Y)
>>> predictedTestLabels = argmax(Xtest.dot(W), axis=1)
```
####How did it do?
```
>>> (predictedTestLabels == labels)/(1.0*len(labels))
0.23
```

...Ouch
####Strawman++

Unfortunately the raw space  these image-vectors live in isn't very good for
linear classification, so our model will perform poorly. So lets "lift" our data
to a "better" space.

Let $\Phi$ be a featurization function, that will "lift" our data. I've
heard neural networks work well for this task, so I'll let $\Phi$ be a convolutional neural net

I'm lazy so I don't have time to add a lot of layers, so it'll be a one layer conv net,
and pool to 2 x 2.

Furthermore I'm really lazy so I'm not going to train the network.  So $\Phi$ is a
*random*, *single layer* convolutional neural network.

The network will use $6 x 6$ patches, $1024$ filters, a cosine nonlinearity and average pool to 2 x 2
for an output dimension of $4096$.

#### How did it do?

```
>>> W = scipy.linalg.solve(phi(X, seed=0), Y)
>>> predictedTestLabels = argmax(phi(Xtest, seed=0).dot(W), axis=1)
>>> (predictedTestLabels == labels)/(1.0*len(labels))
0.65
```

Holy smokes batman!

thats a big jump. But still very far from state of the art.

####Tinman
Since our $\Phi$ is just a random map, what if we "lift" X
multiple times (independently) and concatenate those representations,
since each one is independent and random.

Let $\Phi_{k}$ be this concatenation map


That is:


```
>>> k = 100
>>> def phik(X):
        return np.hstack(map(lambda i: phi(X, seed=i), range(k)))
```

We can let $A = \Phi_{k}(X)$, note A is now $50000 \times 4096k$


Remember we want to minimize $\|AW - Y\|_{F}^{2} + \lambda\|W\|$

Our previous calculus tells us $W^{*} = (A^{T}A + \lambda I_{4096k})^{-1}A^{T}Y$

And our prediction vector would be $\hat{Y} = A(A^{T}A + \lambda I_{d})^{-1}A^{T}Y$

Even for moderate values of k (perhaps over $25$), $(AA^{T} + \lambda I_{d})^{-1}$ becomes very hard to compute (since the inverse scales as $d^{3}$).

We can finally use the [useful matrix inequality](https://people.eecs.berkeley.edu/~stephentu/blog/matrix-analysis/2016/06/03/matrix-inverse-equality.html)
to rewrite the prediction vector

$\hat{Y} = AA^{T}(AA^{T} + \lambda I_{50k})^{-1}Y$

Thus our new linear model looks like:

$C = A^{T}(AA^{T} + \lambda I_{50k})^{-1}Y$

Now we never have to invert anything greater than $50k \times 50k$ !

####How did it do?

```
>>> A = phik(X)
>>> C = A.t * np.linalg.solve(A.dot(A.t) + lambda * np.eye(n), Y)
>>> predictedTestLabels= np.argmax(phik(Xtest).dot(C), axis=1)
>>> (predictedTestLabels == labels)/(1.0*len(labels))
0.80
```

yay!




