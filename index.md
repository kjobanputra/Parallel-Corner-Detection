# Parallel Corner Detection

## Progress
Week 6 (Final Week):
  + After getting some initial numbers, we explored compression and streaming to hide memory latency in the GPU implementation. The rest of the time, we spent on getting final numbers, writing up the [final report](https://github.com/kjobanputra/Parallel-Corner-Detection/blob/gh-pages/418_Final_Project_Paper.pdf), making the [slides](https://github.com/kjobanputra/Parallel-Corner-Detection/blob/gh-pages/Parallel%20Corner%20Detection%20Presentation.pdf), and finally, making a short [video](https://youtu.be/sHniXtd6qsc) going through the slides. 

Week 5:
  + We've researched what optimizations we need to make on the shared memory side and explored different approaches to make optimizations on the GPU implementation. We also implemented the shared memory ideas using OpenMP and optimizing the partial derivative calculation. Namely, we explored unrolling the kernel loop and utilizing shared memory. We started testing and getting numbers.

Week 4:
  + We implemented the last step in the algorithm. There are further serial optimizations that we are considering to implement to compare the best possible serial version as well as the best possible parallel version. You can see results in the following link: [Midpoint Results](https://github.com/kjobanputra/Parallel-Corner-Detection/blob/gh-pages/Results%20-%20Midpoint.pdf). As you can see, the serial version seems to be outperforming the parallel version that we implemented in CUDA. As expected, the main bottleneck is the memory allocation as well as the copying of memory to and from the GPU. We also wrote up the following checkpoint report, in which we detail our future work in the coming weeks: [Checkpoint Report](https://github.com/kjobanputra/Parallel-Corner-Detection/blob/gh-pages/Checkpoint%20report.pdf). 

Week 3:
  + We implemented a parallel version of the algorithm up to step 3 of the algorithm specified in the proposal. The computation is pretty similar, except now we allow CUDA threads to execute independent sets of work over looping. After each phase of the algorithm, namely creating a Gaussian blur of the image and then computing the gradients at each pixel, we make sure to put a barrier, as we want to make sure all the updates are done before the next pahse of the algorithm proceeds.

Week 2:
  + We implemented a preliminary serial version of the Harris Corner Detector. On some initial images, it seems to be working properly. We will do more rigorous testing in the weeks to come as we are comparing results with our parallel implementation. The independent sets of work seem more apparent now. In particular, there are three main phases of work, each phase which can be executed in a pixel-independent fashion. However, each phase must fully complete before proceeding to the next. The phases include: the Gaussian convolution, gradient calculation, and non-maximum supression.

Week 1:
  + We researched our project and then wrote up the following proposal: [Proposal](https://github.com/kjobanputra/Parallel-Corner-Detection/blob/gh-pages/Proposal.pdf). The process of writing the proposal made us make sure that we gained the domain knowledge we needed to make further steps in the project.

## Schedule
Week 1 (11/2 - 11/6): Finish proposal, research hardware, and acquire / set up hardware

Week 2 (11/9 - 11/13): Develop serial version of the Harris Corner Detector to run on the NVIDIA Jetson Nano

Week 3 (11/16 - 11/20): Develop and implement a method to parallelize step 1 and 3 of the Harris Corner Detector. Also, prepare for the checkpoint

Week 4 (11/23 - 11/27): Develop and implement a method to parallelize step 4 of the Harris Corner Detector

Week 5:
  + (11/30 - 12/3): 
    + Kunal: Look into using OpenMP for the parallel implementation
    + Ryan: Eliminate padding
  + (12/4 - 12/6):
    + Ryan, Kunal: Eliminate storing intermediate results in between phases

Week 6: 
  + (12/7 - 12/10):
    + Kunal: Implement OpenMP version
    + Ryan: Test our how implementation translates to an embedded device
  + (12/11 - 12/14):
    + Kunal, Ryan: Prepare for final paper, presentation, and demo
