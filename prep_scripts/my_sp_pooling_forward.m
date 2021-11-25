function [Y]= my_sp_pooling_forward(l, X, ds_info) 



sp_num_imgs=ds_info.sp_num_imgs;
dim=size(X, 3);


Y=zeros([1 1 dim sum(sp_num_imgs)], 'like', X);


num_img_batch=size(X, 4);
for idx_img=1:num_img_batch
    
    one_sp_info=ds_info.sp_info{idx_img};
    feat_sps_img=do_sp_pooling(X(:,:,:,idx_img), one_sp_info);
    feat_sps_img=feat_sps_img';
    [dim, N]=size(feat_sps_img);
    assert(N==sp_num_imgs(idx_img));
    
    idx_sp_batch_begin = sum(sp_num_imgs(1: max(0, idx_img-1)))+1; 
    idx_sp_batch_end = idx_sp_batch_begin+sp_num_imgs(idx_img)-1;
    Y(:,:,:,idx_sp_batch_begin:idx_sp_batch_end)=reshape(feat_sps_img, [1 1 dim N]);
    
end




end