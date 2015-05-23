--[[
  Allows to estimate the momory based on model. It also shows the sizes of layers.
--]]
require('torch')
require('os')

local groupSizes = {1, math.pow(2,10),math.pow(2,20),math.pow(2,30),math.pow(2,40),math.pow(2,50)}
local groupNames = {'B', 'KB', 'MB', 'GB', 'TB', 'PB'}

--[[
  Pretty print a bytes value. If group is not specified than it chooses
  the biggest group where the value is > 0.
--]]
function readableBytes(value, group)
  if group == nil then
    local last = 0
    local idx = 1
    for i = 1,#groupSizes do
      local v = value / groupSizes[i]
      if v < 1 then
        return last, groupNames[idx]
      end
      last = v
      idx = i
    end
    return last
  end

  local div = {['B'] = 0, ['KB']=10, ['MB']=20, ['GB']=30, ['TB']=40, ['PB']=50}
  local v = div[group]
  if v == nil then
    return nil
  end
  return value / math.pow(2, v)
end

function shallowcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in pairs(orig) do
            copy[orig_key] = orig_value
        end
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

function Describe(el)
  print('Describe type '..type(el)..' '..el.__typename)
  for key,value in pairs(el) do
      print(string.format("found member %s, type: %s", key, type(value)));
  end

  for key,value in pairs(getmetatable(el)) do
      print(key, value)
  end
  os.exit()
end

function EstimateSpatialConvolutionMM(el)
  local kW = el.kW
  local kH = el.kH
  local dW = el.dW
  local dH = el.dH
  local nOutputPlane = el.nOutputPlane
  local padding = el.padding
  return function(input)
    assert(#input>=3)
    local dim = shallowcopy(input)
    if #input == 3 then
      dim[2],dim[3],dim[4] = 1,dim[2],dim[3]
    end
    dim[2] = nOutputPlane
    dim[3]  = math.floor((dim[3]  + 2*padding - kW) / dW + 1)
    dim[4] = math.floor((dim[4] + 2*padding - kH) / dH + 1)
    return Product(dim), dim
  end
end

function EstimateSpatialMaxPooling(el)
  local kW = el.kW
  local kH = el.kH
  local dW = el.dW
  local dH = el.dH
  return function(input)
    assert(#input==4)
    local dim = shallowcopy(input)
    dim[3]  = math.floor((dim[3]  - kW) / dW + 1)
    dim[4] = math.floor((dim[4]  - kH) / dH + 1)
    return Product(dim),dim
  end
end

function EstimateLinear(el)
  local tnsSize = TensorSize(el.weight) + TensorSize(el.bias) +
    TensorSize(el.gradWeight) + TensorSize(el.gradBias)
  local outSize = el.bias:size(1)
  local inSize = el.weight:size(2)
  return function(input)
    assert(input[2]==inSize, string.format('nn.Linear, want: %d inputs, got %d', inSize, input[2]))
    local batchSize = input[1]
    return tnsSize+outSize*batchSize, {batchSize, outSize}
  end
end

function EstimateSum(el)
  local sumDim = el.dimension
  return function(input)
    local dim = shallowcopy(input)
    dim[sumDim] = 1
    for i=sumDim+1,#input do
      dim[i-1] = dim[i]
    end
    table.remove(dim)
    return Product(dim),dim
  end
end

function EstimateAdd(input)
  local scalar=input.scalar
  return function(input)
    local s = 1
    if not scalar then
      local dim = shallowcopy(input)
      dim[1] = 1
      s = Product(dim)
    end
    return s, shallowcopy(input)
  end
end

-- Multiplies all values Product({100, 32, 32}) = 102400
function Product(sizes)
  local size = 1
  for _,v in pairs(sizes) do
    size = size * v
  end
  return size
end

function EstimateTransferFunc(el)
  if el.inplace then
    return function(input)
      return 0,input
    end
  end
  return function(input)
    return Product(input),input
  end
end

function TensorSize(tns)
  local elements = 1
  for i=1,tns:dim() do
    elements = elements*tns:size(i)
  end
  return elements
end

function EstimateSequential(el)
  local estimators = {}
  for k,v in pairs(el.modules) do
    table.insert(estimators, NewEstimatator(v))
  end
  return function(input)
    local score = 0
    local dim = input
    local size = 0
    print('Sequential <')
    for _,v in pairs(estimators) do
      size,dim = v(dim)
      print(dim)
      score = score + size
    end
    print('>')
    return score, dim
  end
end

function EstimateReshape(el)
  print('Warning: Reshape depends on a input being continguous (if no than copies the input into continguous memory)')
  -- print(el.size)
  local outS = {}
  local shift = 0
  if el.batchMode then
    shift = 1
  end
  outS[1] = 1
  for i = 1,#el.size do
    outS[i+shift] = el.size[i]
  end
  return function(input)
    if shift ~= 0 then
      outS[1] = input[1]
    end
    assert(Product(input) == Product(outS), string.format('Reshape: want %s, got %s', Product(input), Product(outS)))
    -- print(outS)
    return 0, outS
  end
end

function EstimateView(el)
  local outS = {}
  local shift = 0
  if el.batchMode then
    shift = 1
  end
  outS[1] = 1
  local negDim = -1
  for i = 1,#el.size do
    outS[i+shift] = el.size[i]
    if el.size[i]==-1 then
      negDim = i
    end
  end
  if negDim > 0 then
    return function(input)
      assert(#input >= 2 and #input <= 4)
      local out = {input[1], input[2]}
      if #input > 2 then
        out[2] = out[2] * input[3]
        if #input > 3 then
          out[2] = out[2] * input[4]
        end
      end
      return 0, out
    end
  end
  return function(input)
    out = shallowcopy(outS)
    if shift ~= 0 then
      out[1] = input[1]
    end
    local expected = Product(out)
    local got = Product(input)
    if expected<0 then
      out[negDim] = got/expected*-1
    end
    assert(Product(input) == Product(out), string.format('View: want %s, got %s', Product(out), Product(input)))
    print(out)
    return 0, out
  end
end

-- mapping between class names and estimation functions
local funcMap = {[{'nn', 'cudnn'}]={
    ['Sequential']=EstimateSequential,
    ['Linear']=EstimateLinear,
    [{'ReLU', 'Tanh', 'LogSoftMax'}]=EstimateTransferFunc,
    ['View']=EstimateView,
    ['Reshape']=EstimateReshape,
    [{'SpatialConvolutionMM', 'SpatialConvolution'}]=EstimateSpatialConvolutionMM,
    ['SpatialMaxPooling']=EstimateSpatialMaxPooling,
    ['Sum']=EstimateSum,
    ['Add']=EstimateAdd,
  }
}

function unroll(arr, prefix, ret)
  ret = ret or {}
  prefix = prefix or ''
  for k,v in pairs(arr) do
    if type(k) == 'table' and type(v) == 'function' then
      for _,k1 in pairs(k) do
        ret[prefix..k1] = v
      end
    elseif type(k) == 'string' then
      if type(v) == 'function' then
        ret[prefix..k] = v
      elseif type(v) == 'table' then
        unroll(v, prefix..k..'.', ret)
      else
        print('?')
        os.exit(1)
      end
    elseif type(k) == 'table' and type(v) == 'table' then
      for _,k1 in pairs(k) do
        unroll(v, prefix..k1..'.', ret)
      end
    else
      print('??')
      os.exit(1)
    end
  end
  return ret
end

local unrollFuncMap = unroll(funcMap)

function NewEstimatator(el)
  local t = el.__typename or type(el)
  print(string.format('Estimator for %s', t))
  local f = unrollFuncMap[t]
  if f == nil then
    Describe(el)
    print('could not find a coorect function for '..t)
    os.exit(1)
  end
  return f(el)
end

-- model=nn.Sequential();  -- make a multi-layer perceptron
-- model:add(nn.SpatialConvolutionMM(1, 32, 4, 4)) -- output size: 29x29
-- model:add(nn.Tanh())
-- model:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- output size 14x14

-- model:add(nn.Reshape(32 * 98 * 98, true))

-- model:add(nn.Linear(307328, 200))
-- model:add(nn.Tanh())

-- -- Describe(model)
-- estimator = NewEstimatator(model)

-- size,dim = estimator({10, 200, 200})
-- v,g = readableBytes(size)
-- print(string.format('Estimated memory: %d%s (%d), output dimetion %dx%d', v,g,size, dim[1], dim[2]))
