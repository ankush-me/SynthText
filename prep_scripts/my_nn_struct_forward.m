function res = my_nn_struct_forward(net, x, res, varargin)

opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts.keep_layer_output=true;


opts.ds_info=[];

opts = vl_argparse(opts, varargin);

n = numel(net.layers) ;

gpuMode = isa(x, 'gpuArray') ;


if isempty(res)
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1)), ...
    'output_info', []) ;
end

res(1).x = x ;


ds_info=opts.ds_info;


for i=1:n
  l = net.layers{i} ;
  res(i).time = tic ;
  switch l.type
    case 'conv'
      assert(size(l.filters, 3) == size(res(i).x, 3) );
      res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, 'pad', l.pad, 'stride', l.stride) ;
    case 'pool'
      res(i+1).x = vl_nnpool(res(i).x, l.pool, 'pad', l.pad, 'stride', l.stride, 'method', l.method) ;
    case 'normalize'
      res(i+1).x = vl_nnnormalize(res(i).x, l.param) ;
    case 'softmax'
      res(i+1).x = vl_nnsoftmax(res(i).x) ;
    case 'loss'
      res(i+1).x = vl_nnloss(res(i).x, ds_info.label_data) ;
    case 'softmaxloss'
      res(i+1).x = vl_nnsoftmaxloss(res(i).x, ds_info.label_data) ;
    case 'relu'
      res(i+1).x = vl_nnrelu(res(i).x) ;
    case 'noffset'
      res(i+1).x = vl_nnnoffset(res(i).x, l.param) ;
    case 'dropout'
      if opts.disableDropout
        res(i+1).x = res(i).x ;
      elseif opts.freezeDropout
        [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate, 'mask', res(i+1).aux) ;
      else
        [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate) ;
      end
    case 'custom'
        res(i+1).x = l.forward(l, res(i).x, ds_info) ;
      
    case 'squared_loss' 
        [res(i+1).x, res(i+1).output_info] = ...
            my_nn_squared_loss(res(i).x, ds_info) ;

     case 'logistic' 
        res(i+1).x = my_nn_logistic(res(i).x) ;

    
    case 'structured_loss' 
        res(i+1).output_info= my_nn_structured_loss(res(i).x, l, ds_info) ;
        res(i+1).x=res(i+1).output_info.objective_value;  
      
      
    otherwise
      error('Unknown layer type %s', l.type) ;
  end
  
  if opts.conserveMemory & ~opts.keep_layer_output & i < numel(net.layers) - 1
    res(i).x = [] ;
  end
  
  if gpuMode & opts.sync
    wait(gpuDevice) ;
  end
  res(i).time = toc(res(i).time) ;
  
end



