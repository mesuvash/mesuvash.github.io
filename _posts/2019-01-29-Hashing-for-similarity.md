---
layout: post
title: Large scale similarity via hashing
date: 2015-03-15
description: Machine Learning
---
# Locality Sensitive Hashing for Large Scale Similarity

Similarity computation is a very common task in real-world machine learning and data mining problems such as recommender systems,spam detection, online advertising etc. Consider a tweet recommendation problem where one has to find tweets similar to the tweet user previously clicked. This problem becomes extremely challenging when there are billions of tweets created each day.

In this post, we will discuss the two most common similarity metric, namely Jaccard similarity and Cosine similarity; and Locality Sensitive Hashing based approximation of those metrics.



## Jaccard Similarity
Jaccard similarity is one of the most popular metrics used to compute the similarity between two sets. Given set $$A$$ and $$B$$, jaccard similarity is defined as 

$$ 
    jaccard(A, B) = \frac{| A \cap B | }{| A \cup B|}
$$

Let's define tweet as 

$$
 t_1 = \{u_1, u_2\} \\
 t_2 = \{u_2, u_3, u_4\}
$$ 

where each tweet is represented by a set of users who liked the tweet. The jaccard similarity between $$t_1$$ and $$t_2$$
​​is given by

$$ 
    jaccard(t_1, t_2) = \frac{ 1 }{4}
$$


## Cosine similarity

Cosine similarity is a vector space metric where similarity between two vector $$\vec{a}$$ and $$\vec{b}$$ is defined as
$$ 
    cosine(\vec{a}, \vec{b}) = \frac{ \langle \vec{a}, \vec{b} \rangle }{\|\vec{a}\| \|\vec{b}\|} \\
$$
where,  $$\langle . \rangle$$ is a dot product and  $$\| .\|$$ is  $$L_2$$ norm operator.

Jaccard similarity ignores the weights as it’s a set based distance metric for e.g. number of user clicks. Lets define

$$
\begin{aligned}
t_1 & = \{u_1 =2, u_2=1\} \\
t_2 & = \{u_2=2, u_3=8, u_4=3\}
\end{aligned}
$$

where the number indicates the number of times user $$u$$ clicked tweet $$t$$. Now, the tweets can be represented in user's vector space as

<div>
<img class="center-image" src="/assets/img/blogs/lsh/vector_features.png" height="100"/>
</div>

and the cosine similarity is given as
$$ 
    cosine(\vec{t_1}, \vec{t_2})  = \frac{ \langle \vec{t_1}, \vec{t_2} \rangle }{\|\vec{t_1}\| \|\vec{t_2}\|} =  \frac{2}{\sqrt{(5) * (77)}} = 0.1019
$$

#### Challenge
Pairwise similarity scales quadratically $$\Theta(n^2)$$ both in terms of time and space complexity. Hence, finding similar items is very challenging for a large number of items. In the following section, we discuss locality sensitive hashing to address these limitations.

### Locality Sensitive Hashing (LSH)

Hashing is a very widely used technique that assigns pseudo-random value/bucket to objects. Hash functions must be uniform i.e. each bucket is equally likely. Locality Sensitive Hashing(LSH) is a hashing based dimensionality reduction method that preserves item similarity. More precisely, LSH hashes items to kk buckets such that similar items map to the same bucket with high probability.

#### Minhash
Let $$h(*) \in f(*)\rightarrow N$$ be a hash function that maps an object to a positive integer. The minhash is defined as

$$
minhash_h(t = \lbrace u_1, u_2...\rbrace) = argmin_u \ h(u_i)
$$

a function that returns the item with smallest hash value.Now, a k-dimensional minhash is defned by kk hash functions $$\{h_1, h_2, ...., h_k\rbrace$$

Given $$k\text{-}minhash(t)k-minhash(t)$$, jaccard similarity of item $$t_1$$
& $$t_2$$ is defined as

$$
\begin{aligned}
        jaccard(t_1, t_2) & = P\lbrack minhash(t_1) = minhash(t_2)\rbrack \\
        & \approx \frac{\sum_{i}^{} \text{1} \lbrack k\text{-}minhash(t_1)_i = k\text{-}minhash(t_2)_i \rbrack}{k}    
\end{aligned}
$$

<div><img class="center-image" src="/assets/img/blogs/lsh/confused.gif" height="150"></div>

**Minhash explained**

Although theoretical proof is quite rigorous, the key idea is very intuitive. First, let’s define,

> a) How many ways can we shuffle $$U$$ such that $$u_2$$ is the first element?
> 
> Ans: 3!
> 
> b) More generally, how many ways can we shuffle $$U$$ such that $$u \in S$$ is the first element?
> 
> Ans: $$d \times (n-1)!$$

Hashing a set of items is equivalent to generating a random permutation of the set. So, $$minhash_h(t_1) = minhash_h(t_2)$$, only if the hash function $$h$$ assigns smallest value to $$u \in S$$ i.e $$u_2$$. From this observation, we can conclude

$$
\begin{aligned}
    P\lbrack minhash_h(t_1) = minhash_h(t_2)\rbrack = &  \frac{d(n-1)!}{n!} \\
     = & \ \ \frac{d}{n} \\
     = & \frac{|S|}{|U|} \\
     = & \frac{|t_1 \cap t_2|}{|t_1 \cup t_2|}, \ \text{voila!!}
\end{aligned}
$$

**Hence, k-minhash can approximate jaccard similarity.**

### Simhash
Let $$\vec{h_i}$$ be a random vector passing through origin. Let’s define a simhash function for tweet $$t$$

$$
simhash_{\vec{h}}(\vec{t}) = sgn(\langle \vec{t}, \vec{h} \rangle)
$$

as a sign of dot product between $$h$$ and $$t$$.

Given simhash, $$k\text{-}simhash$$ can be defined as

$$
k\text{-}simhash(t) = \lbrack simhash_{\vec{h_1}}(\vec{t}), simhash_{\vec{h_2}}(\vec{t}) ....., simhash_{\vec{h_k}}(\vec{t})\rbrack
$$


Now, the angle between $$t_1$$ and $$t_2$$ is defined as

$$
\begin{aligned}
    \theta & =  (1 - P\lbrack k\text{-}simhash_h(t_1) = k\text{-}simhash_h(t_2)\rbrack) \times \pi \\
    & \approx (1- \frac{\sum_{i=1}^{} \text{1} \lbrack simhash_{\vec{h_i}}(\vec{t_1}) = simhash_{\vec{h_i}}(\vec{t_2}) \rbrack}{k})  \times \pi 
\end{aligned}
$$

<div><img class="center-image" src="/assets/img/blogs/lsh//tellmehow.gif" height="200"></div>

**Simhash explained**

In this section, we discuss the intuition behind approximation of cosine similarity using simhash.

<div><img class="center-image" src="/assets/img/blogs/lsh/dot_hyperplane.svg" height="350"></div>

In the figure above, for the vector $$\vec{t}$$, the pink shaded area corresponds to the half-space where $$simhash_{\vec{*}}(\vec{t}) \gt 1simhash$$, for eg $$simhash_{\vec{h_1}}(\vec{t})$$. On the other hand, the white region corresponds to the half-space where $$simhash_{\vec{*}}(\vec{t}) \lt 0$$,for eg $$simhash_{\vec{h_2}}(\vec{t})$$

<div><img class="center-image" src="/assets/img/blogs/lsh/dot_product_two_vector.svg" height="350"></div>

Lets consider two vector $$t_1$$, $$t_2$$ and $$\theta$$ is an angle between them as shown in figure above. For a randomly drawn a vector hh passing through origin

$$
\lbrack k\text{-}simhash_h(t_1) = k\text{-}simhash_h(t_2)\rbrack 
$$

is true only if vector hh lies in purple or white shaded area (i.e other than pink and blue shaded area). From this observation, we can define

$$
\begin{aligned}
P\lbrack k\text{-}simhash_h(t_1) = k\text{-}simhash_h(t_2)\rbrack  & = (1 - \frac{\text{blue or  pink region}}{2 \times \pi})\\
& = (1 - \frac{\theta}{\pi})\\
\theta & = ( 1 - P\lbrack k\text{-}simhash_h(t_1) = k\text{-}simhash_h(t_2)\rbrack) \times \pi , \ \text{voila!!}


\end{aligned}

$$

**Hence, k-simhash can approximate cosine similarity.**

In the next post, I will discuss how to compute minhash and simhash efficiently in a distributed environment.







<!-- - References
  - https://users.soe.ucsc.edu/~niejiazhong/slides/kumar.pdf
  - https://www.cs.rice.edu/~as143/Doc/Anshumali_Shrivastava.pdf (Page 13) -->