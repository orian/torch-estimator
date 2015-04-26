# torch-estimator
A tiny lib which estimates a memory requirements and sizes of Torch ANN model.

The basic usage is as follow:

	model=nn.Sequential();  -- make a multi-layer perceptron
	model:add(nn.SpatialConvolutionMM(1, 32, 4, 4)) -- output size: 29x29
	model:add(nn.Tanh())
	model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- output size 14x14

	model:add(nn.Reshape(32 * 98 * 98, true))

	model:add(nn.Linear(307328, 200))
	model:add(nn.Tanh())

	-- Describe(model)
	estimator = NewEstimatator(model)

	size,dim = estimator({10, 200, 200})
	v,g = readableBytes(size)
	print(string.format('Estimated memory: %d%s (%d), output dimetion %dx%d', v,g,size, dim[1], dim[2]))


The library supports only few basic layer types:
	- Sequential
	- Linear
	- ReLU
	- Tanh
	- Reshape
	- SpatialConvolutionMM
	- SpatialMaxPooling

