require 'nn'
require 'cunn'
require 'optim'

function loadAndTest(testData, testLabels, modelPath)
	
	--load model
	mp = modelPath or 'mnist_model.net'
	local model = torch.load(mp)
	
	-- prepare data if none passed
	if not testData or not testLabels then
		local mnist = require 'mnist'
		testData = mnist.testdataset().data:float()
		testLabels = mnist.testdataset().label:add(1)
		
		-- normalize data
		local trainData = mnist.traindataset().data:float()
		local mean = trainData:mean()
		local std = trainData:std()
		testData:add(-mean):div(std)
	end
	
	local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
    local numBatches = 0
	local batchSize = 1000
	
    for i = 1, testData:size(1), batchSize do
        numBatches = numBatches + 1
        local x = testData:narrow(1, i, batchSize):cuda()
        local yt = testLabels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
        confusion:batchAdd(y,yt)
    end
    
    confusion:updateValids()
    local avgError = 1 - confusion.totalValid
    return avgError
end
