---
layout: post
title: Large scale similarity via hashing
date: 2015-03-15
description: Machine Learning
---
# Locality Sensitive Hashing for Large Scale Similarity

Computing similarity a very common task in real world machine learning and data mining problems such as recommender systems, spam detection,  online advertising etc. Consider a tweet recommendation problem where one has to find top-k tweet similar to the tweet user clicked. This problem becomes extremely challenging when there are billions of tweets created each day. 

Locality Sensitive Hashing(LSH) is a class of approximate similarity computation algorithm. In this blog post we discuss two most common similarity metric, namely Jaccard similarity and Cosine similarity; and LSH based approximation of those metrics.

## Jaccard Similarity
Jaccard similarity is one of the most popular metric used to compute similarity between two sets. Given set $A$ and $B$, jaccard similarity is defined as
$$ 
    jaccard(A, B) = \frac{| A \cap B | }{| A \cup B|}
$$

Let's consider a tweet, $t_1 = \{u_1, u_2\}$  and $t_2 = \{u_2, u_3, u_4\}$, represented by a set of users who liked the tweet. The jaccard similarity between $t_1$ and $t_2$ is 
$$ 
    jaccard(t_1, t_2) = \frac{ 1 }{4}
$$


## Cosine similarity

Cosine similarity is a vector space metric where similarity between two vector $\vec{a}$ and $\vec{b}$ is defined as
$$ 
    cosine(\vec{a}, \vec{b}) = \frac{ \langle \vec{a}, \vec{b} \rangle }{\|\vec{a}\| \|\vec{b}\|} \\
$$
where,  $\langle . \rangle$ is a dot product and  $\| .\|$ is  $L_2$ norm operator.

In other words, cosine similarity is cosine of angle, $\theta$, between two vectors as shown in figure below. 

|![C](../assets/img/blogs/lsh/cosine_similarity.png =220x)|
|:--:| 
| Fig 1: Angle between two vector |

Since the jaccard similarity is a set based metrics, it ignores the associated weights i.e in our example number of times user clicked. Lets define $t_1 = \{u_1 =2, u_2=1\}$  and $t_2 = \{u_2=2, u_3=8, u_4=3\}$, where $u_1 =2$ indicates $u_1$ clicked tweet $t_1$ two times. Now, the tweets can be represented in user's vector space as

![C](../assets/img/blogs/lsh/vector_features.png =220x)

and the cosine similarity is given as
$$ 
    cosine(\vec{t_1}, \vec{t_2})  = \frac{ \langle \vec{t_1}, \vec{t_2} \rangle }{\|\vec{t_1}\| \|\vec{t_2}\|} =  \frac{2}{\sqrt{(5) * (77)}} = 0.1019
$$


## Computing similarity 

- References
  - https://users.soe.ucsc.edu/~niejiazhong/slides/kumar.pdf
  - https://www.cs.rice.edu/~as143/Doc/Anshumali_Shrivastava.pdf (Page 13)