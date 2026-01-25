---
title: "A Succinct Guide To Bidirectional Associative Memory (BAM)"
source: "https://towardsdatascience.com/a-succinct-guide-to-bidirectional-associative-memory-bam-d2f1ac9b868/"
author:
  - "[[Arthur V. Ratz]]"
published: 2022-04-03
created: 2025-11-14
description: "Everything That Readers Need To Know About Bidirectional Associative Memory (BAM), With Samples In Python 3.8 and NumPy"
tags:
  - "clippings"
---
[Skip to content](https://towardsdatascience.com/a-succinct-guide-to-bidirectional-associative-memory-bam-d2f1ac9b868/#wp--skip-link--target)

Everything That Readers Need To Know About Bidirectional Associative Memory (BAM), With Samples In Python 3.8 and NumPy

13 min read

![Photo by Arthur V. Ratz](https://towardsdatascience.com/wp-content/uploads/2022/04/1uVi4yfMwkhO6Y2m_iJydjw.jpeg)

Photo by Arthur V. Ratz

## Introduction

Bidirectional Associative Memory (BAM) is a recurrent neural network (RNN) of a special type, initially proposed by Bart Kosko, in the early 1980s, attempting to overcome several known drawbacks of the auto-associative Hopfield network, and ANNs, that learn associations of data from continuous training.

The BAM is the hetero-associative memory, providing an ability to store the associated data, regardless of its type and structure, without continuous learning. The associations of data are simultaneously stored, only once, prior to recalling them from the BAM’s memory. Normally, the BAM outruns the existing ANNs, providing significantly better performance while recalling the associations. Also, using the BAM perfectly solves the known bitwise XOR problem, compared to the existing ANNs, which makes it possible to store the data, encoded to binary (bipolar) form, in its memory.

Unlike the unidirectional Hopfield network, the BAM is capable of recalling the associations for data, assigned to its either inputs or outputs, bidirectionally. Also, it allows retrieving the correct associations for even incomplete or distorted input data.

The BAM models are rather efficient when deployed as a part of an AI-based decision-making process, inferring the solution to a specific data analysis problem, based on various associations of many interrelated data.

Generally, the BAM model can be used for a vast of applications, such as:

- **Classification and clustering data**
- **Incomplete data augmentation**
- **Recovering damaged or corrupted data**

The BAM models are very useful whenever the varieties of knowledge, acquired by the ANNs, are not enough for processing the data, introduced to an AI for analysis.

For example, the prediction of words missed out from incomplete texts with ANN, basically requires that the associations of words-to-sentences are stored in the ANN’s memory. However, this would incur an incorrect prediction, because the same missing words might occur in more than one incomplete sentence. In this case, using the BAM provides an ability to store and recall all possible associations of the data entities, such as *words-to-sentences*, *sentences-to-words*, *words-to-words*, and *sentences-to-sentences,* and vice versa*.* This, in turn, notably increases the quality of prediction. Associating various entities of data multi-directionally is also recommended for either augmenting or clustering these data.

Finally, the memory capacity of each ANN’s layer is bound to the size of the largest floating-point type: *float64*. For instance, the capacity of a single ANN layer of shape (100×100) is only 1024 bytes. Obviously, the number of the ANN’s layers, as well as the number of neurons, in each of its layers, must be increased to have the ability to store the associations and recall them multi-directionally. This, in turn, negatively impacts the ANN-based memory latency, since its learning and prediction workloads are growing, proportionally to the sizes of the ANN.

Despite this, the high performance of learning and prediction of the BAM provides an ability to allocate as large amounts of memory space as required, beyond the conventional ANN’s memory capacity limits. As well, it’s possible to aggregate multiple BAMs into the memory, having a layered structure.

## Associations Stored In Memory…

The BAM learns the associations of various data, converted to bipolar patterns. A bipolar pattern is a special case of binary vector, the elements of which are the values of **1’s** and **\-1’s**, respectively. Multiple patterns (columns) are arranged into the 2D-pattern maps, along the y-axis of a matrix.x The pattern maps are the embeddings, each column of which corresponds to a specific input and output data item, stored in the BAM’s memory:

![Encoding Data Into Bipolar 2D-Patterns Map of Shape (s x n) | Image by the author](https://towardsdatascience.com/wp-content/uploads/2022/04/1TuS0xMXlK7AogvNj9iSaAg.png)

Encoding Data Into Bipolar 2D-Patterns Map of Shape (s x n) | Image by the author

The BAM model simultaneously learns the associated data from pattern maps, assigned to its corresponding inputs and outputs, respectively:

![Input (X) And Output (Y) Pattern Maps Assigned To The BAM Model | Image by the author](https://towardsdatascience.com/wp-content/uploads/2022/04/13LRtAeBcs796SxqphgYmKg.png)

Input (X) And Output (Y) Pattern Maps Assigned To The BAM Model | Image by the author

In the figure, above, the BAM model learns the associations of data, encoded into the pattern maps 𝙓 and 𝒀 of the shapes (𝙨 x 𝙣) and (𝙨 x 𝙢), respectively. While learning, the values from the input and output patterns are assigned to the corresponding inputs 𝙭ᵢ ∈𝙓 and **o** utputs 𝙮ⱼ∈ 𝒀 of th **e** BAM’s model. In turn, both input and output pattern maps (columns) must have an equal number of patterns (𝙨 ). Since the BAM is a hetero-associative memory model, it has a different number of inputs (𝙣 ) and outputs (𝙢 ), and, thus, the different number of rows in those pattern maps, 𝙓, and 𝒀, correspondingly.

## BAM’s Topology And Structure

Generally, the BAM is a primitive neural network (NN), consisting of only input and output layers, interconnected by the synaptic weights:

![The BAM Model's NN-Topology (4 x 3) | Image by the author](https://towardsdatascience.com/wp-content/uploads/2022/04/1sCXGKFXRzjRAXfZnayC69g.png)

The BAM Model’s NN-Topology (4 x 3) | Image by the author

In the figure, above, the neurons (𝒏) of the input layer perform no output computation, feedforwarding the inputs 𝑿 to the memory cells (𝒎) in the BAM’s output layer. Its sizes basically depend on the quantities of inputs and outputs of the BAM (e.g., the dimensions of the input and output pattern maps). The NN, shown above, of the shape (4 x 3), consists of 4 neurons and 3 memory cells in its input and output layers, respectively.

Unlike the conventional ANNs, the output layer of BAM consists of memory cells that perform the BAM’s outputs computation. Each memory cell computes its output 𝒀, based on the multiple inputs 𝑿, forwarded to the cell by the synaptic weights 𝑾, each one is of specific strength.

The outputs of memory cells correspond to specific values of patterns, being recalled:

![The Outputs (Y) Of The BAM's Memory Cells | Image by the author](https://towardsdatascience.com/wp-content/uploads/2022/04/1QQmpcGv1IIAL0J3-uDSGyQ.png)

The Outputs (Y) Of The BAM’s Memory Cells | Image by the author

The BAM’s outputs are computed as the weighted sum of all BAM’s inputs 𝑾ᵀ𝙓, applied as an argument to the bipolar threshold function. The po ***sitive (***+) or ne ***gative (***\-) value (e.g., 1 **o** r -1**) of** each output 𝒀 is always proportional to the magnitude of the sum of the cell’s weighted inputs.

All synaptic weights, interconnecting the input neurons and memory cells store the fundamental memories, learned by the BAM, based on the Hebbian supervised learning algorithm, discussed, below.

## Learning Algorithm

Algebraically, the ability of BAMs to store and recall associations solely relies on the bidirectional property of matrices, studied by T. Kohonen and J. Anderson, in the mids of 1950s:

> *The inner dot product of two matrices 𝑿 and 𝒀, which gives a correlation matrix 𝑾 is said to be bidirectionally stable, as the result of recollection and feedback of 𝑿 and 𝒀 into 𝑾.*

Generally, it means that the corresponding rows and columns of both matrices 𝑿 and 𝒀, as well as their scalar vector products, equally contribute to the values of 𝑾, a *s* the recollections:

![](https://towardsdatascience.com/wp-content/uploads/2022/04/1F77IWp_tjpuJraj6AU6TsA.png)

Since that, each vector of the matrices 𝑿 and 𝒀 can be easily recalled by taking an inner dot product 𝑾𝑿ₖ or 𝑾 𝒀ₖ, discussed, below.

In this case, 𝑾 is a matrix, which infers the correlation of the 𝑿ₖ or 𝒀ₖ vectors. An entire learning process is much similar to kee\_ **ping (𝙬** *ᵢⱼ≠ 0) **or** disc* **arding (𝙬ᵢⱼ** \_=0) t **he** \_s\_pecific weights, that interconnect input neurons and memory cells of the BAM.

The values of synaptic weights 𝙬ᵢⱼ∈ **\*\* 𝑾 can be either greater than, less, or equal to 0**. **The BAM keeps only those synaptic weights, which values are either positive or negative. The synaptic weights, equal to 0**,\*\* are simply discarded while initializing the model.

The bidirectional matrix property was formulated by Donald Hebb as the famous supervised learning rule, which is fundamental for the Hopfield network, and other associative memory models.

According to the Hebbian algorithm, all associated patterns are stored in the BAM’s memory space, only once, while initializing the matrix 𝑾ₙₓₘ of the synaptic weights (e.g., memory storage space). The matrix 𝑾ₙₓₘ is obtained by taking an inner dot product of the input 𝑿ₚₓₙ and output 𝒀ₚₓₘ pattern maps (i.e., matrices), respectively:

![](https://towardsdatascience.com/wp-content/uploads/2022/04/1mki-ipaTcuswF5VXWMG9Bg.png)

Alternatively, the weights matrix 𝑾ₙₓₘ can be obtained as the sum of outer Kronecker products of the corresponding patterns 𝙭ₖ and 𝙮ₖ, from the 𝑿ₚₓₙ, 𝒀ₚₓₘ matrices.

![](https://towardsdatascience.com/wp-content/uploads/2022/04/1TZxyonXI5Wvg4v5JbgQ2kg.png)

Unlike the inner dot product of 𝑿 and 𝒀, the outer products of (𝙭ₖ,𝙮ₖ) ∈ 𝑿,**𝒀** vectors yield matrices 𝙬ₖ ∈ 𝑾 (i.**e**., the fundamental memories). Finally, all of the matrices 𝙬ₖ are combined into a single matrix 𝑾, by obtaining their sum, pointwise.

An entire process of the BAM’s synaptic weights intialization is illustrated in the figure, below:

![Learning BAM Model Based On The Hebbian Algorithm | Image by the author](https://towardsdatascience.com/wp-content/uploads/2022/04/1C2mPivPJ7Hw3t-_YWbWlVQ.png)

Learning BAM Model Based On The Hebbian Algorithm | Image by the author

However, the correlation matrix 𝑾 provides the most stable and consistent memory storage, for the patterns, which are the orthogonal vectors. The product of orthogonal vectors in 𝑿,𝒀 gives the symmetric correlation matrix 𝑾, in which the fundamental memories of associations, learned from the pattern maps 𝑿 and 𝒀, respectively.

Thus, it’s highly recommended to perform the orthogonalization of 𝑿 and 𝒀 pattern maps, prior to training the model. Since the 𝑿 and 𝒀 are bipolar, the orthogonalization of 𝒀 can be easily done as: 𝒀 = -𝑿. By taking the negative -𝑿, the 𝒀 patterns will have the same values as 𝑿, the sign (+/-) of w **hich** is changed to an opposite.

The BAM’s learning algorithm is trivial and can be easily implemented as 1-line code, in Python 3.8 and NumPy:

```
# Initializes the corellaction weights matrix W 
        
        
          

          # as the inner dot product of X and Y pattern maps
        
        
          

          def learn(x,y):
        
        
          

              return x.T.dot(y)
        
        
          

          

        
        
          

          # Initializes the corellaction weights matrix W as
        
        
          

          # the sum of outer Kronecker products of corresponding
        
        
          

          # patterns x and y from the maps X,Y, respectively.
        
        
          

          def learn_op(x,y):
        
        
          

              return np.sum([np.outer(x,y) for x,y in zip(x,y)],axis=0)
```

[view raw](https://gist.github.com/arthurratz/aba81d48f36c4f2c13ed1ce96fc7984b/raw/90c9d8bd372450611faa712bb67860f28e7421ad/bam_learning.py) [bam\_learning.py](https://gist.github.com/arthurratz/aba81d48f36c4f2c13ed1ce96fc7984b#file-bam_learning-py) hosted with ❤ by [GitHub](https://github.com/)

## Recalling Patterns From The Memory

Since the input and output pattern maps have been already stored in the BAM’s memory, the associations for specific patterns are recalled by computing the BAM’s full output, within a number of iterations, until the correct association for an input pattern 𝑿 was retrieved.

To compute the full memory output, in each of the iterations 𝞭-iterations, 𝒌=𝟏..𝞭, the BAM’s inputs 𝙭ᵢ of the pattern 𝙓 are multiplied by the corresponding synaptic weights 𝙬ᵢⱼ∈ 𝙒. Al **g** ebraically, this is done by taking an inner product of the weights matrix 𝙒-transpose and the input pattern 𝙓. Then, the weighted sum 𝙒 ᵀ𝙓 of the BAM’s inputs is applied to the bipolar threshold function 𝒀=𝙁(𝙒 ᵀ𝙓), to com\_ **p** *ute ea* **c** \_h memory cell’s output, which corresponds to a specific value 1 or -1, of the **o** utpu **t** pattern 𝙮ⱼ∈ 𝒀, being recall **ed**. This computation is similar to feedforwarding the inputs through each layer of the conventional ANN.

The bipolar threshold function 𝙁 is just the same as the classical threshold function with only a difference that its value for the positive (+) or 0 **\*\* inputs is** 1, **and negative (-), unless otherwise. In this case, the positive input of 𝙁 also includes zero:\*\***

![Bipolar Threshold Function Y=F(X)](https://towardsdatascience.com/wp-content/uploads/2022/04/1gnvQ6XyNiFpyssa_FCmtYA.png)

Bipolar Threshold Function Y=F(X)

The bipolar threshold function’s plot diagram is shown, below:

![Bipolar Threshold Activation Function | Image by the author](https://towardsdatascience.com/wp-content/uploads/2022/04/1rgjAoUtNe6-bgJGviLxwaQ.png)

Bipolar Threshold Activation Function | Image by the author

A simple fragment of the code, demonstrating the BAM’s memory cells activation is listed below:

```
# Bipolar threshold activation function
        
        
          

          def bipolar_th(x):
        
        
          

              return 1 if x >= 0 else -1
        
        
          

          # Applies the bipolar_th(x) function to the sum 
        
        
          

          # of weighted inputs of all BAM's memory cells.
        
        
          

          def activate(x):
        
        
          

              return np.vectorize(bipolar_th)(x)
```

[view raw](https://gist.github.com/arthurratz/ca48396e327549ce7ae7da3d9ab35d2b/raw/c9eb40f44dc8e27d521e06502d8f302de9600598/activation_fn.py) [activation\_fn.py](https://gist.github.com/arthurratz/ca48396e327549ce7ae7da3d9ab35d2b#file-activation_fn-py) hosted with ❤ by [GitHub](https://github.com/)

Unlike the conventional ANNs, the memory cells in the output layer also compute the new inputs 𝙓ₖ₊₁ for the next iteration 𝙠+𝟏. Then, it performs a check on whether the current 𝙓 and new inputs 𝙓ₖ₊₁ are not equal (𝙓≠𝙓ₖ₊₁). If not, it assigns the new inputs to the BAM, proceeding to compute the output 𝒀 in the next iteration. Otherwise, if 𝙓=𝙓ₖ₊₁, then it means that the output 𝒀 is the correct association for the input 𝙓, and the algorithm has converged, returning the pattern 𝒀 from the procedure. Also, the BAM provides an ability to recall the associations for either inputs 𝙓 and outputs 𝒀, by computing the memory outputs, bidirectionally, based on the same algorithm, being discussed.

The patterns recalling process is illustrated in the figure, below:

![A Memory Cell Output (Y) Computation | Image by the author](https://towardsdatascience.com/wp-content/uploads/2022/04/1RPc3IKov8jj6mTn2PJj1Ng.png)

A Memory Cell Output (Y) Computation | Image by the author

**An algorithm for recalling the association from the BAM’s memory is listed below:**

Let 𝑾ₙₓₘ— the correlation weights matrix (i.e., the BAM memory storage space), 𝑿ₙ— an input pattern of the length 𝒏, 𝒀ₘ— an output pattern of the length 𝒎:

Recall an association 𝒀 for an input pattern 𝑿, previously stored in the BAM’s memory:

**For each of the 𝞭-iterations,𝟏 *≤* 𝞭** ≤ 𝒕**,****do the following:**

1. Initialize the BAM’s inputs with the input vector 𝑿₁←𝑿.
2. Compute the BAM output vector 𝒀ₖ, as an inner dot product of the weights matrix 𝑾 ᵀ **tr** anspose, and the input vector 𝑿ₖ, for the 𝒌-th iteration:
![](https://towardsdatascience.com/wp-content/uploads/2022/04/1mvbMZVuxCNpaWYkHGggDxQ.png)

1. Obtain the new input vector 𝑿ₖ₊₁ for the next (𝒌＋𝟏)-th iteration, such as:
![](https://towardsdatascience.com/wp-content/uploads/2022/04/15oadIEKegFHOxIyqyn_ifQ.png)

1. Check if new and existing vectors 𝑿ₖ₊₁≠𝑿ₖ, are NO **T t** he same:

If **not**, return to step 1, to compute the output 𝒀ₖ₊₁, for the (𝒌＋𝟏)-th iteration, or proceed to the next step 5, unless otherwise.

1. Return the output vector 𝒀←𝒀ₖ from the procedure, as the correct association for the input 𝑿 vector.
2. Proceed with steps 2–4, until convergence.

A fragment of code in Python 3.8 and NumPy, implementing the patterns prediction algorithm is listed, below:

```
# Recalls an association Y for the input pattern X, bidirectionally:
        
        
          

          def recall(w,x,d='out'):
        
        
          

              end_of_recall = False; \
        
        
          

                  y_pred = None; x_eval = y_pred
        
        
          

              # Compute the BAM output until the existing inputs x
        
        
          

              # are not equal to the new inputs x_eval (x != x_eval)
        
        
          

              while end_of_recall == False:
        
        
          

                  # Compute the output y_pred of all memory cells, activated
        
        
          

                  # by the bipolar threshold function F(X): [ w^T*x - forward, w*x - backwards ]
        
        
          

                  y_pred = activate(w.T.dot(x) \
        
        
          

                      if d == 'out' else w.dot(x))
        
        
          

                  # Compute the new inputs x_eval for the next iteration:
        
        
          

                  # [ w*y - forward, w^T*y - backwards]
        
        
          

                  x_eval = activate(w.dot(y_pred) \
        
        
          

                      if d == 'out' else w.T.dot(y_pred))
        
        
          

                  # Check if x and x_eval are not the same. 
        
        
          

                  # If not, assign the new inputs x_eval to x
        
        
          

                  x,end_of_recall = x_eval,np.all(np.equal(x,x_eval))
        
        
          

          

        
        
          

              return y_pred  # Return the output pattern Y, recalled from the BAM.
```

[view raw](https://gist.github.com/arthurratz/4d134722f8a5d119942d5e2ced3d522d/raw/b67acb4e5edb524af9cc372751cc83a50e1b657a/predict_pattern.py) [predict\_pattern.py](https://gist.github.com/arthurratz/4d134722f8a5d119942d5e2ced3d522d#file-predict_pattern-py) hosted with ❤ by [GitHub](https://github.com/)

## Memory Evaluation (Testing)

The code, listed below, demonstrates the BAM evaluation (testing) process. It builds the BAM model of a specific shape \_(patterns×neurons×memory *cells)*, and generates 1D-array, consisting of the **1** and **\-1** values. Then, it reshapes the array into the input 2D-patterns map 𝑿. To obtain the patterns map 𝒀, it performs the orthogonalization of 𝑿, such as: 𝒀=−𝑿. Next, the correlation matrix is computed to store the associations of patterns in 𝑿 and 𝒀 into the BAM’s memory.

Finally, it performs the model evaluation for each input pattern 𝙭ᵢ ∈ **𝙓** it recalls an association from the BAM’s memory, bidirectionally. When an association y⃗ₚ for the input pattern x⃗ was recalled from the memory, it does the consistency check whether the output y⃗ₚ target y⃗ pattern are identical, displaying the results for each of the input patterns 𝙭ᵢ ∈ 𝙓**:**

```
# The BAM model of 8*10^3 inputs, 5*10^3 memory cells, with memory capacity - 20 patterns
        
        
          

          

        
        
          

          patterns = 20; neurons = 8000; mm_cells = 5500
        
        
          

          

        
        
          

          # Generate input (X) and output (Y) patterns maps of shapes (patterns x neurons) and (patterns by mm_cells)
        
        
          

          X = np.array([1 if x > 0.5 else -1 for x in np.random.rand(patterns*neurons)],dtype=np.int8)
        
        
          

          

        
        
          

          # Orthogonalize the input patterns (X) into the corresponding output patterns (Y) 
        
        
          

          Y = np.array(-X[:patterns*mm_cells],dtype=np.int8)
        
        
          

          

        
        
          

          # Reshape patterns into the input and output 2D-pattern maps X and Y
        
        
          

          X = np.reshape(X,(patterns,neurons))
        
        
          

          Y = np.reshape(Y,(patterns,mm_cells))
        
        
          

          

        
        
          

          # Learn the BAM model with the associations of the input and output patterns X and Y
        
        
          

          W = learn_op(X,Y) # W - the correlation weights matrix (i.e., the BAM's memory storage space)
        
        
          

          

        
        
          

          print("Recalling the associations (Y) for the input patterns (X):\n")
        
        
          

          

        
        
          

          # Recall an association (Y) for each input (X) and target output (Y') patterns, from X,Y
        
        
          

          for x,y in zip(X,Y):
        
        
          

              y_pred = recall(W,x,'out') # y_pred - the predicted pattern Y
        
        
          

              # Check if the target and predicted patterns (Y) are identical, and display the results
        
        
          

              print("x =",x,"target =",y,"y =",-y_pred," :",np.any(-y_pred != y))
        
        
          

          

        
        
          

          print("\r\nRecalling the associations (X) for the output patterns (Y):\n")
        
        
          

          

        
        
          

          # Recall an association (X) for each output (Y) and target input (X) patterns, from X,Y
        
        
          

          for x,y in zip(X,Y):
        
        
          

              x_pred = recall(W,y,d='in') # x_pred - the predicted pattern X
        
        
          

              # Check if the target and predicted patterns (X) are identical, and display the results
        
        
          

              print("y =",y,"target =",x,"x =",-x_pred," :",np.any(-x_pred != x))
```

[view raw](https://gist.github.com/arthurratz/8963bfd8fbac74a8d43bf96916a239ca/raw/745c986dfd67d6a87c00162121bd11b50550afb1/bam_evaluation.py) [bam\_evaluation.py](https://gist.github.com/arthurratz/8963bfd8fbac74a8d43bf96916a239ca#file-bam_evaluation-py) hosted with ❤ by [GitHub](https://github.com/)

## Predicting Incomplete Patterns

One of the BAM model’s advantages is an ability to predict the correct associations for by **\>30** incomplete or damaged inputs. The fragment of code, below, demonstrates the incomplete patterns prediction. It randomly selects an input pattern from the patterns map 𝑿, and distors it, replacing its multiple values with an arbitrary 1 **\*\* or** \-1\*\*. It applies the poison(…) function, implemented in the code, below, to distort the randomly selected pattern x⃗, the association for which is being recalled. Finally, it predicts an associated pattern, performing the consistency check if the target and predicted associations, y⃗ and y⃗ₚ are identical. If so, the correct association for the distorted pattern x⃗ has been recalled:

```
# Distorts an input pattern map X
        
        
          

          def poison(x,ratio=0.33,distort='yes'):
        
        
          

              p_fn = [ lambda x: 0 if np.random.rand() > 0.5 else x,
        
        
          

                       lambda x: 1 if np.random.rand() > 0.5 else -1, ]
        
        
          

          

        
        
          

              x_shape = np.shape(x); x = np.reshape(x,-1)
        
        
          

              return np.reshape(np.vectorize(p_fn[distort == 'yes'])(x),x_shape)
        
        
          

          

        
        
          

          # Predicting a randomly distorted pattern
        
        
          

          print("\r\nPredicting a randomly distorted pattern X:\r\n")
        
        
          

          

        
        
          

          # Select a pattern from X, randomly
        
        
          

          pattern_n = np.random.randint(0,np.size(X,axis=0))
        
        
          

          

        
        
          

          # Distort the input pattern with random 1's or -1's
        
        
          

          x_dist = poison(X[pattern_n],distort='yes')
        
        
          

          

        
        
          

          # Predict a correct association for the random pattern X
        
        
          

          y_pred = recall(W,x_dist)
        
        
          

          

        
        
          

          # Display the results
        
        
          

          print("Output:\r\n")
        
        
          

          print("x =",x,"target =",y,"y =",y_pred,":",np.any(y[pattern_n] != y_pred),"\r\n")
```

[view raw](https://gist.github.com/arthurratz/2ec10790c6f799bd0838302e9fef43b8/raw/81eab3a3ee37b84a90d681dc144413d2ee9f8f35/predict_incomplete.py) [predict\_incomplete.py](https://gist.github.com/arthurratz/2ec10790c6f799bd0838302e9fef43b8#file-predict_incomplete-py) hosted with ❤ by [GitHub](https://github.com/)

## Conclusion

Despite its agility to store and recall the associations without continuous learning, the BAM model might become less efficient for some applications, due to the inability to recall the correct associations for data of certain types. According to the latest research, predicting associations for various incomplete or corrupted data basically requires the BAMs of enormously huge sizes, consisting of large amounts of memory cells. There are also several issues while recalling the correct associations for a small number of patterns, being stored. Obviously, the existing variations of the BAM models must undergo improvements to be used as persistent memory storage, for a vast of applications. Although, the workarounds to the following issues are still in process.

---

## Source Code:

- \_ ["Bidirectional Associative Memory (BAM) In Python 3.8 And NumPy", Jupyter Notebook @ Google Colab](https://colab.research.google.com/drive/1M9oTh4cruwLJBvKs3ijMpY7ptYP_BGXJ) \_
- \_ [Bidirectional Associative Memory, Python 3.8 + NumPy Samples, Visual Studio 2022 Project @ GitHub Repository](https://github.com/arthurratz/bam_associations_intro) \_

## Disclaimer

All images were designed by the author of this story, by using the Draw.io application, [https://www.diagrams.net/](https://www.diagrams.net/)

## References

1. \_ ["Bidirectional associative memory" – From Wikipedia, the free encyclopedia.](https://en.wikipedia.org/wiki/Bidirectional_associative_memory)\_
2. \_ [Adaptive bidirectional associative memories, Bart Kosko, IEEE Transactions on Systems, MAN, And Cybernetics, VOL.18, No. 1, January/February 1988.](https://opg.optica.org/DirectPDFAccess/52A68963-3B21-4588-868A1CEBDFCC7199_30894/ao-26-23-4947.pdf?da=1&id=30894&seq=0&mobile=no)\_
3. *[Pattern Association or Associative Networks, Jugal Kalita University of Colorado at Colorado Springs.](http://www.cs.uccs.edu/~jkalita/work/cs587/2014/05PatternAssoc.pdf)*

---

Written By

Arthur V. Ratz

, , , ,

Towards Data Science is a community publication. Submit your insights to reach our global audience and earn through the TDS Author Payment Program.

[Write for TDS](https://towardsdatascience.com/questions-96667b06af5/)

## Related Articles

- ![](https://towardsdatascience.com/wp-content/uploads/2024/08/0c09RmbCCpfjAbSMq.png)
	## Implementing Convolutional Neural Networks in TensorFlow
	Step-by-step code guide to building a Convolutional Neural Network
	6 min read
- ![Image by author](https://towardsdatascience.com/wp-content/uploads/2024/07/1Zf6XTb6jDQXVOt-N9S_YTg.png)
	Image by author
	## Deep Dive into LSTMs & xLSTMs by Hand ✍️
	Explore the wisdom of LSTM leading into xLSTMs - a probable competition to the present-day LLMs
	13 min read
- ## Speeding Up the Vision Transformer with BatchNorm
	How integrating Batch Normalization in an encoder-only Transformer architecture can lead to reduced training time…
	28 min read
- ![The Math Behind Keras 3 Optimizers: Deep Understanding and Application. Image by DALL-E-3](https://towardsdatascience.com/wp-content/uploads/2024/08/1ZPwekpFJpznH-KcWTW65Vw.png)
	The Math Behind Keras 3 Optimizers: Deep Understanding and Application. Image by DALL-E-3
	## The Math Behind Keras 3 Optimizers: Deep Understanding and Application
	This is a bit different from what the books say.
	9 min read
- ![](https://towardsdatascience.com/wp-content/plugins/ui-kit-core/dist/images/fallback-image.jpg)
	## Latest picks: Time Series Forecasting with Deep Learning and Attention Mechanism
	Your daily dose of data science
	1 min read
- ![Image generated by DALL·E 3](https://towardsdatascience.com/wp-content/uploads/2023/12/15FM14YZopRvGK9baJR0OtQ.png)
	Image generated by DALL·E 3
	## Stacked Ensembles for Advanced Predictive Modeling With H2O.ai and Optuna
	And how I placed top 10% in Europe’s largest machine learning competition with them!
	15 min read
- ![Figure 2: Multi-panel of coincident vertical reflectivity profiles from a surface radar, CloudSat and the Global Precipitation Measurement mission, along with their respective radar blind zones. Image retrieved from Kidd et al., 2021 (https://doi.org/10.3390/rs13091708).](https://towardsdatascience.com/wp-content/uploads/2024/04/1FcWQedWtnXL5UXV2bL5Tbw.png)
	Figure 2: Multi-panel of coincident vertical reflectivity profiles from a surface radar, CloudSat and the Global Precipitation Measurement mission, along with their respective radar blind zones. Image retrieved from Kidd et al., 2021 (https://doi.org/10.3390/rs13091708).
	## Beyond the Blind Zone
	Inpainting radar gaps with deep learning
	25 min read

Some areas of this page may shift around if you resize the browser window. Be sure to check heading and document order.