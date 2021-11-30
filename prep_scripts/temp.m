imdir = "/media/shubham/One Touch/Indic_OCR/datasets/ocr_dataset/SynthText/image_preprocessed_data/3/images";
imnames = dir(fullfile(imdir,'*.jpg'));
imnames = {imnames.name};
N = numel(imnames);
out_file = "/media/shubham/One Touch/Indic_OCR/datasets/ocr_dataset/SynthText/image_preprocessed_data/3/depth3.h5";
for i = 1:N  
    imname = 'COCO_train2014_00a000009999.jpg'
    dset_name = ['/',imname];
    disp(dset_name)
    info = h5info(out_file);
    disp(info);
end