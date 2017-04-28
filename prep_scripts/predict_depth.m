% MATLAB script to regress a depth mask for an image.
% uses: (1) https://bitbucket.org/fayao/dcnf-fcsp/
%       (2) vlfeat
%       (3) matconvnet

% Author: Ankush Gupta

function predict_depth()
    % setup vlfeat
    run( '../libs/vlfeat-0.9.18/toolbox/vl_setup');
    % setup matconvnet
    dir_matConvNet='../libs/matconvnet/matlab/';
    addpath(genpath(dir_matConvNet));
    run([dir_matConvNet 'vl_setupnn.m']);

    opts=[];
    opts.useGpu=true;
    opts.inpaint = true;
    opts.normalize_depth = false; % limit depth to [0,1]
    opts.imdir = '/path/to/image/dir';

    opts.out_h5 = '/path/to/save/output/depth.h5';

    % these should point to the pre-trained models from:
    %  https://bitbucket.org/fayao/dcnf-fcsp/
    opts.model_file.indoor =  '../model_trained/model_dcnf-fcsp_NYUD2';
    opts.model_file.outdoor =  '../model_trained/model_dcnf-fcsp_Make3D';

    fprintf('\nloading trained model...\n\n');
    mdl = load(opts.model_file.indoor);
    model.indoor = mdl.data_obj;
    mdl = load(opts.model_file.outdoor);
    model.outdoor = mdl.data_obj;

    if gpuDeviceCount==0
        fprintf(' ** No GPU found. Using CPU...\n');
        opts.useGpu=false;
    end

    imnames = dir(fullfile(opts.imdir),'*');
    imnames = {imnames.name};
    N = numel(imnames);
    for i = 1:N
        fprintf('%d of %d\n',i,N);
        imname = imnames{i};
        imtype = 'outdoor';
        img = read_img_rgb(fullfile(opts.imdir,imname));
        if strcmp(imtype, 'outdoor')
            opts.sp_size=16;
            opts.max_edge=600;
        elseif strcmp(imtype, 'indoor')
            opts.sp_size=20;
            opts.max_edge=640;
        end
        depth = get_depth(img,model.(imtype),opts);
        save_depth(imname,depth,opts);
    end
end

function save_depth(imname,depth,opts)
    dset_name = ['/',imname];
    h5create(opts.out_h5, dset_name, size(depth), 'Datatype', 'single');
    h5write(opts.out_h5, dset_name, depth);
end

function depth = get_depth(im_rgb,model,opts)
    % limit the maximum edge size of the image:
    if ~isempty(opts.max_edge)
        sz = size(im_rgb);
        [~,max_dim] = max(sz(1:2));
        osz = NaN*ones(1,2);
        osz(max_dim) = opts.max_edge;
        im_rgb = imresize(im_rgb, osz);
    end

    % do super-pixels:
    fprintf(' > super-pix\n');
    supix = gen_supperpixel_info(im_rgb, opts.sp_size);
    pinfo = gen_feature_info_pairwise(im_rgb, supix);

    % build "data-set":
    ds=[];
    ds.img_idxes = 1;
    ds.img_data = im_rgb;
    ds.sp_info{1} = supix;
    ds.pws_info = pinfo;
    ds.sp_num_imgs = supix.sp_num;
    % run cnn:
    fprintf(' > CNN\n');
    depth = do_model_evaluate(model, ds, opts);

    if opts.inpaint
        fprintf(' > inpaint\n');
        depth = do_inpainting(depth, im_rgb, supix);
    end

    if opts.normalize_depth
        d_min = min(depth(:));
        d_max = max(depth(:));
        depth = (depth-d_min) / (d_max-d_min);
        depth(depth<0) = 0;
        depth(depth>1) = 1;
    end
end