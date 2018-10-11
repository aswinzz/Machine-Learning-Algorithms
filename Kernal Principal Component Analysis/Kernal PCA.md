### KPCA: *Kernel Principal Component Analysis*

Algorithm:
- Calculate the [Squared-Euclidean](https://en.wikipedia.org/wiki/Euclidean_distance) Distance of the data
- Convert the resulting data into a square matrix
- Apply the [Radius Basis Function](https://en.wikipedia.org/wiki/Radial_basis_function_kernel) kernel method on the resulting Square matrix.
- Center the resulting kernel matrix (say K) following the below method :
	* Calculate the mean of kernel matrix (say K_mean)
	* K_center_data = K - K_mean
	(Here an inbuilt data centerer algorithm is used for this purpose)
- Find the Eigen Values and corresponding Eigen vectors of the centered data
- Sort the Eigen vectors in descending order of eigen values
- Calculate the final result as dot product of K_center_data and first *N* Eigen vectors (here *N* is the feature_size)

Implementaion:
- We have a 2D non-linear data containing 2 classes marked with red and blue.
- Kernel Principal Component Analysis Algorithm was applied on the data to separate both classes of data.
- The resulting data can be visualised in 3D and 2D plot.
