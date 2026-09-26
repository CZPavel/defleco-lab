# References

The implementation summarizes concepts in original words and does not reproduce paper text or proprietary algorithms.

## Scientific background

1. Laura Arnal, J. Ernesto Solanes, Jaime Molina, Josep Tornero, "Detecting dings and dents on specular car body surfaces based on optical flow," *Journal of Manufacturing Systems* 45 (2017), 306-321. [doi:10.1016/j.jmsy.2017.07.006](https://doi.org/10.1016/j.jmsy.2017.07.006).
2. Jaime Molina, J. Ernesto Solanes, Laura Arnal, Josep Tornero, "On the detection of defects on specular car body surfaces," *Robotics and Computer-Integrated Manufacturing* 48 (2017), 263-278. [doi:10.1016/j.rcim.2017.04.009](https://doi.org/10.1016/j.rcim.2017.04.009).
3. Mariano Rivera, Oscar Dalmau, Adonai Gonzalez, Francisco Hernandez-Lopez, "Two-step fringe pattern analysis with a Gabor filter bank," *Optics and Lasers in Engineering* 85 (2016), 29-37. [doi:10.1016/j.optlaseng.2016.04.014](https://doi.org/10.1016/j.optlaseng.2016.04.014).
4. Qian Kemao, "Windowed Fourier transform for fringe pattern analysis," *Applied Optics* 43(13) (2004), 2695-2702. [doi:10.1364/AO.43.002695](https://doi.org/10.1364/AO.43.002695). WFT is a possible future extension, not a V01 metrology pipeline.
5. Ruiyang Wang, Xinwei Zhang, Dahai Li, "Dynamic speckle deflectometry based on backward digital image correlation," *Measurement* 171 (2021), 108860. [doi:10.1016/j.measurement.2020.108860](https://doi.org/10.1016/j.measurement.2020.108860).

## Software APIs

- OpenCV [`phaseCorrelate`](https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html)
- OpenCV [`calcOpticalFlowFarneback`](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)
- OpenCV [`DISOpticalFlow`](https://docs.opencv.org/4.x/de/d4f/classcv_1_1DISOpticalFlow.html)
- OpenCV [`findTransformECC`](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)
- Official [pypylon repository](https://github.com/basler/pypylon)
