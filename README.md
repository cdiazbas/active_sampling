# Designing wavelength sampling for Fabry-Pérot observations


This repository contains the code to design wavelength sampling schemes for Fabry-Pérot observations ([https://arxiv.org/abs/2303.13875](https://arxiv.org/abs/2303.13875)).

![example](images/sketch.png?raw=true "")

**Figure 1** — Sketch of a uniform sampling and the result of the sampling scheme given by the neural network (left panel) with has been optimized to retrieve the best average temperature across the atmospheric stratification (right panels).

## Abstract
Fabry-Pérot interferometers (FPIs) have become very popular in solar observations because they offer a balance between cadence, spatial resolution, and spectral resolution through a careful design of the spectral sampling scheme according to the observational requirements of a given target. However, an efficient balance requires knowledge of the expected target conditions, the properties of the chosen spectral line, and the instrumental characteristics.

Our aim is to find a method that allows finding the optimal spectral sampling of FPI observations in a given spectral region. The selected line positions must maximize the information content in the observation with a minimal number of points. In this study, we propose a technique based on a sequential selection approach where a neural network is used to predict the spectrum (or physical quantities, if the model is known) from the information at a few points. Only those points that contain relevant information and improve the model prediction are included in the sampling scheme.

We have quantified the performance of the new sampling schemes by showing the lower errors in the model parameter reconstructions. The method adapts the separation of the points according to the spectral resolution of the instrument, the typical broadening of the spectral shape, and the typical Doppler velocities. The experiments using the Ca II 8542A line show that the resulting wavelength scheme naturally places (almost a factor 4) more points in the core than in the wings, consistent with the sensitivity of the spectral line at each wavelength interval. As a result, observations focused on magnetic field analysis should prioritize a denser grid near the core, while those focused on thermodynamic properties benefit from a larger coverage. The method can also be used as an accurate interpolator, to improve the inference of the magnetic field when using the weak-field approximation. Overall, this method offers an objective approach for designing new instrumentation or observing proposals with customized configurations for specific targets. This is particularly relevant when studying highly dynamic events in the solar atmosphere with a cadence that preserves spectral coherence without sacrificing much information.
