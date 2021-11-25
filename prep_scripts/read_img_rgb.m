function img_data=read_img_rgb(img_file, use_uint8)

if nargin<2
    use_uint8=true;
end

try

	[img_data, map] = imread(img_file);

catch err_info

	disp(err_info);
	disp('do converting to PNG....');
	tmp_img_file=[img_file '_tmp.png'];
	unix(sprintf('convert %s %s' , img_file, tmp_img_file));
	[img_data, map] = imread(tmp_img_file);
end



if size(img_data, 4)>1
    img_data=img_data(:,:,:, 1);
end


if ~isempty(map)
	img_data = ind2rgb(img_data,map);
end


if size(img_data, 3)==1

    [img_data map]=gray2ind(img_data);
    img_data=ind2rgb(img_data, map);

end


if use_uint8
    img_data=im2uint8(img_data);
else
    img_data=im2double(img_data);
end


try
    assert(size(img_data, 3)==3);
catch error_msg
    disp(error_msg);
    dbstack;
    keyboard;
end



end
