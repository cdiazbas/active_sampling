- realdata_example_stokesI_symmetric_half_interpolate: no NNs, just interpolation to get a cheap approximation of the spectra from the points I already have (linear or splines).

- realdata_example_stokesI_symmetric_half_fullmodel: train a NN with input a given parametrized scheme and output the spectra, so I can later evaluate the error and get the best scheme.
