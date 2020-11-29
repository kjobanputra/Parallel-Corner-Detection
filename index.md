## Parallel Corner Detection
Week 4:
  + We implemented the last step in the algorithm. There are further serial optimizations that we are considering to implement to compare the best possible serial version as well as the best possible parallel version. You can see results in the following link: [Midpoint Results](https://github.com/kjobanputra/Parallel-Corner-Detection/blob/gh-pages/Results%20-%20Midpoint.pdf). We also wrote up the following checkpoint report: [Checkpoint Report]() 

Week 3:
  + We implemented a parallel version of the algorithm up to step 3 of the algorithm specified in the proposal. The computation is pretty similar, except now we allow CUDA threads to execute independent sets of work over looping. After each phase of the algorithm, namely creating a Gaussian blur of the image and then computing the gradients at each pixel, we make sure to put a barrier, as we want to make sure all the updates are done before the next pahse of the algorithm proceeds.

Week 2:
  + We implemented a preliminary serial version of the Harris Corner Detector. On some initial images, it seems to be working properly. We will do more rigorous testing in the weeks to come as we are comparing results with our parallel implementation.

Week 1:
  + We researched our project and then wrote up the following proposal: [Proposal](https://github.com/kjobanputra/Parallel-Corner-Detection/blob/gh-pages/Proposal.pdf)
