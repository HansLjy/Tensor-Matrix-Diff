## Tensor-Matrix-Diff

**T**ensor-**M**atrix-**D**iff(TMD) is a symbolic differentiation system. Unlike the [Symbolic Math Toolbox](https://www.mathworks.com/help/symbolic/symbolic-computations-in-matlab.html) of MATLAB which calculates the derivatives of a tensor function in an element-by-element sense, TMD aims to calculate the derivatives while treating matrices as the minimum unit of operation.

TMD is mainly inspired by a series of Zhihu articles([1st episode](https://zhuanlan.zhihu.com/p/24709748) and [2nd episode](https://zhuanlan.zhihu.com/p/24863977)). We thank the [author](https://www.zhihu.com/people/chang-qu-gui-xia) for his/her great introduction to matrix differentiation.

Here is an example result: the partial derivative of 2D gaussian function to the variance:

![derivative](assets/pFpV.png)

## TODO

Much is still under construction for TMD:

- [ ] Simplification: Some simplifications are easy to implement, for example, the cancellation of identity matrices. However, I could not find a systematic way to do all possible simplifications. Ideally, I would like my simplified expressions to have the "normal form" $$\sum \prod K I \otimes M_i K$$ before I plug it into implementations. The problem is that $M_i$ is possibly not a variable or constant. For example, if $M_i$ is the vectorization of a complex expression, it could lead to problems.
- [ ] Fast evaluation