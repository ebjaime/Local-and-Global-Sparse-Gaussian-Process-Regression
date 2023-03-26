# Study of Gaussian Process (GP) local and global approximations
Gaussian process (GP) models are flexible probabilistic nonparametric models for regression,  classification  and  other  tasks.  Unfortunately  they  suffer  from  computational  intractability  for  large  data  sets.  Over  the  past  decade  there  have  been  many  different  approximations  developed to reduce this cost. Most of these can be termed global approximations, in that they  try to summarize all the training data via a small set of support points. A different approach is  that of local regression, where many local experts account for their own part of space. In this  project we are interested to study the regimes in which these different approaches work well  or  fail,  and  then  apply  a  new  sparse  GP  approximation  which  is  a  combination  of  both  the  global and local approaches, and look extremely promising. 

The ![GPflow](https://github.com/GPflow/GPflow) was modified and is so added to this repository for the proper execution of the programs. All the modifications made are tagged with a *REVIEW* tag.

[CST @ Polimi - 2021/22]
