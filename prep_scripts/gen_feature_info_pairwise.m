


%TODO more pairwise feature defined by kernel similarity


function pws_info=gen_feature_info_pairwise(img_data, sp_info)


%NOTE: the order should match the pairwise params order in the trained model
sim=cell(3,1);
sim{1}=gen_feat_texture_diff( img_data, sp_info);
sim{2}=gen_feat_color_diff_color_hist(img_data, sp_info);
sim{3}=gen_feat_color_diff(img_data, sp_info);

pws_info=cell(1,3);
for k=1:3
    
    S = sim{k};   
    diag_S = diag(sum(S));
    
    pws_info{1, k} = diag_S - S;
    
end
    


end







function one_pws_info=gen_feat_per_img(feat_gen_config, img_data, sp_info)


    
one_img_info=[];
one_img_info.sp_info=sp_info;
one_img_info.img_data=img_data;


one_pws_info=feat_gen_config.gen_feat_info_fn(feat_gen_config, one_img_info);


end












function tmp_img_data=gen_img_color_space(img_data, color_space)

if strcmp(color_space, 'luv')
    tmp_img_data=vl_xyz2luv(vl_rgb2xyz(img_data));
    
    tmp_img_data(isnan(tmp_img_data))=0;
end


if strcmp(color_space, 'rgb')
    
    tmp_img_data=img_data;
    
end

end








function feat_data_sps=gen_feat_data_sps_color(feat_gen_config, one_img_info)


img_data=one_img_info.img_data;
sp_info=one_img_info.sp_info;

color_space=feat_gen_config.color_space;
one_box_size=feat_gen_config.one_box_size;

tmp_img_data=gen_img_color_space(img_data, color_space);
feat_data_sps=gen_mean_color_sps(sp_info, tmp_img_data, one_box_size);


end





function feat_data_sps=gen_feat_data_sps_texture(feat_gen_config, one_img_info)


cell_size=feat_gen_config.cell_size;
box_size=[cell_size cell_size];

img_data=one_img_info.img_data;
img_data=im2single(img_data);
img_data=rgb2gray(img_data);

img_data_pad=padarray(img_data, [box_size 0], 0, 'both' );

sp_info=one_img_info.sp_info;
sp_num=sp_info.sp_num;
img_size=sp_info.img_size;

one_feat_img=[];
for sp_idx=1:sp_num
    
    pixel_inds=sp_info.pixel_ind_sps{sp_idx};
    one_sp_mask=zeros(img_size);
    one_sp_mask(pixel_inds)=1;
    
    box_img_data=gen_box_img_pad(img_data_pad, one_sp_mask, box_size);
    one_feat = vl_lbp(box_img_data, cell_size);
    
    
    one_feat=one_feat(:)';
    
    
    if isempty(one_feat_img)
        one_feat_img=zeros(sp_num, size(one_feat,2));
    end
    
    one_feat_img(sp_idx,:)=one_feat;
end


feat_data_sps=one_feat_img;


end









function one_pws_info=gen_feat_color_diff( img_data, sp_infos)


feat_type='color_diff';


color_spaces=[];
color_spaces{end+1}='luv';


gamma_factors=[0.1];
box_size=[0];

    
    
for color_idx=1:length(color_spaces)

    base_gamma=1;

    gammas=gamma_factors.*base_gamma;
    color_space=color_spaces{color_idx};

    feat_gen_config=[];

    feat_id=sprintf('%s_color%s_box%d', feat_type, color_space, box_size);
    feat_gen_config.feat_id=feat_id;
    feat_gen_config.feat_type=feat_type;

    feat_gen_config.color_space=color_space;
    feat_gen_config.gammas=gammas;

    feat_gen_config.one_box_size=box_size;

    feat_gen_config.gen_feat_data_sps_fn=@gen_feat_data_sps_color;

    feat_gen_config.gen_feat_info_fn=@gen_feat_one_img_feat_diff;

    one_pws_info=gen_feat_per_img(feat_gen_config, img_data, sp_infos);

end
    


end










function one_pws_info=gen_feat_texture_diff(img_data, sp_infos)


feat_type='texture_diff_lbp';
cell_size=8;
gamma_factors=50;
base_gamma=1;
gammas=gamma_factors.*base_gamma;
feat_id=sprintf('%s_cell%d%s', feat_type, cell_size);


feat_gen_config=[];
feat_gen_config.feat_id=feat_id;
feat_gen_config.feat_type=feat_type;

feat_gen_config.cell_size=cell_size;
feat_gen_config.gammas=gammas;


feat_gen_config.gen_feat_data_sps_fn=@gen_feat_data_sps_texture;

feat_gen_config.gen_feat_info_fn=@gen_feat_one_img_feat_diff;

one_pws_info=gen_feat_per_img(feat_gen_config, img_data, sp_infos);



end








function feat_data_sps=gen_feat_data_sps_color_hist(feat_gen_config, one_img_info)


one_box_size=feat_gen_config.one_box_size;
box_size=[one_box_size one_box_size];



img_data=one_img_info.img_data;

img_data=im2uint8(img_data);


sp_info=one_img_info.sp_info;
sp_num=sp_info.sp_num;
img_size=sp_info.img_size;

one_feat_img=[];
for sp_idx=1:sp_num
    
    pixel_inds=sp_info.pixel_ind_sps{sp_idx};
    one_sp_mask=zeros(img_size);
    one_sp_mask(pixel_inds)=1;
    
    if one_box_size>0
        
        box_img_data=gen_box_img_nopad(img_data, one_sp_mask, box_size);
        tmp_one_sp_mask=ones(size(box_img_data, 1), size(box_img_data, 2));
        one_feat=color_hist(box_img_data, tmp_one_sp_mask);
        
    else
        
        one_feat=color_hist(img_data, one_sp_mask);
        
    end
    
    
    if isempty(one_feat_img)
        one_feat_img=zeros(sp_num, size(one_feat,2));
    end
    
    one_feat_img(sp_idx,:)=one_feat;
end


feat_data_sps=one_feat_img;


end








function one_pws_info=gen_feat_color_diff_color_hist(img_data, sp_infos)


feat_type='color_diff_color_hist';
box_size=[0];
gamma_factors=25;
base_gamma=1;
gammas=gamma_factors.*base_gamma;
feat_id=sprintf('%s_box%d', feat_type, box_size);


feat_gen_config=[];
feat_gen_config.feat_id=feat_id;
feat_gen_config.feat_type=feat_type;

feat_gen_config.one_box_size=box_size;
feat_gen_config.gammas=gammas;


feat_gen_config.gen_feat_data_sps_fn=@gen_feat_data_sps_color_hist;

feat_gen_config.gen_feat_info_fn=@gen_feat_one_img_feat_diff;

one_pws_info=gen_feat_per_img(feat_gen_config, img_data, sp_infos);
    


end










function mean_color_sps=gen_mean_color_sps(sp_info, tmp_img_data, one_box_size)

sp_num=sp_info.sp_num;
pixel_ind_sps=sp_info.pixel_ind_sps;

img_dim=size(tmp_img_data, 3);
mean_color_sps=zeros(sp_num, img_dim);

img_size=sp_info.img_size;
box_size=[one_box_size, one_box_size];

if one_box_size<=0
    for dim_idx=1:img_dim
        one_dim_img=tmp_img_data(:,:, dim_idx);
        for sp_idx=1:sp_num
            
            pixel_inds=pixel_ind_sps{sp_idx};
            mean_color_sps(sp_idx,dim_idx)=mean(one_dim_img(pixel_inds));
        end
    end
    
else
    
    for sp_idx=1:sp_num
        
        pixel_inds=pixel_ind_sps{sp_idx};
        one_sp_mask=zeros(img_size);
        one_sp_mask(pixel_inds)=1;
        box_img_data=gen_box_img_nopad(tmp_img_data, one_sp_mask, box_size);
        
        for dim_idx=1:img_dim
            one_dim_img=box_img_data(:,:, dim_idx);
            mean_color_sps(sp_idx,dim_idx)=mean(one_dim_img(:));
        end
    end
    
    
end

end









function one_pws_info=gen_feat_one_img_feat_diff(feat_gen_config, one_img_info)


sp_info=one_img_info.sp_info;
sp_num=sp_info.sp_num;

img_data=one_img_info.img_data;
img_data=im2single(img_data);
img_size=sp_info.img_size;
assert(img_size(1)==size(img_data, 1));
assert(img_size(2)==size(img_data, 2));



feat_data_sps=feat_gen_config.gen_feat_data_sps_fn(feat_gen_config, one_img_info);

gammas=feat_gen_config.gammas;
assert(numel(gammas)==1);

sp_relation_infos=sp_info.relation_infos;

ind_cols=[];
ind_rows=[];
val_adj=[];

for sp_idx = 1:sp_num
    
    one_sp_info=sp_relation_infos{sp_idx};
    adjacent_sp_inds=one_sp_info.adjacent_sp_inds;
    adj_num=length(adjacent_sp_inds);
    
    adj_feat_data=zeros(adj_num, 1);
    
    for adj_sp_idx_idx = 1:adj_num
        
        adj_sp_idx = adjacent_sp_inds(adj_sp_idx_idx);
        one_feat_diff=feat_data_sps(sp_idx, :) - feat_data_sps(adj_sp_idx, :);
        
        
        one_dist=norm(one_feat_diff);

        one_dist=one_dist./sqrt(size(feat_data_sps, 2));
        
        adj_feat_data(adj_sp_idx_idx) = exp(-gammas*one_dist);
        
    end
    
    
    ind_cols=[ind_cols; double(adjacent_sp_inds)];
    ind_rows=[ind_rows; sp_idx*ones(adj_num, 1)];
    val_adj=[val_adj; adj_feat_data];
    
    try
        assert(min(adj_feat_data(:))>=0);
    catch
        dbstack;
        keyboard;
    end
    
end

one_pws_info=sparse(ind_rows, ind_cols, val_adj, sp_num, sp_num);




end











