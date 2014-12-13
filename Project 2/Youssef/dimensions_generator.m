for i = 1:length(depths)
    temp = zeros(1, depths(i));
    temp(1) = 2;
    
    for j = 2:depths(i)
        temp(j) = temp(j-1)*4;
    end
    
    dimensions{i} = temp(end:-1:1);
end