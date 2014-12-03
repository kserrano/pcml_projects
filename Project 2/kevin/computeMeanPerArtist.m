function m = computeMeanPerArtist(ua)
%% Take the mean per artist 
s = sum(ua>0);
s(s==0) = 1;
m = sum(ua)./s;
end



