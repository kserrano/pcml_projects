% You have to fill in the code in this function for Task 2A
function region_mask = region_growing(I,x,y)
% % You HAVE TO RE-IMPLEMENT ACCORDING TO THE INSTRUCTIONS
% warning('YOU HAVE TO COMMENT THIS CODE OUT AND PLACE YOURS');
% region_mask = ones(size(I));

negList = [];
SeedVal = 1;
region_mask = FALSE(size(I));

noOfPixels = size(I, 1)*size(I, 2);

noCheckedPixels = 0;

while (noCheckedPixels <= noOfPixels) && (seedVal > 0)
    
    %checking wheter adjacent pixels are neighbours, if not add them,
    %storing the indices in a N by 2 matrix, first column indicates the row
    %and second one indicates the column (in the original image I)
     potentialNeighbours = [x, y+1; x, y-1; x+1, y; x-1, y];
     
    % removing out of border indices
%      for l = 1:size(potentialNeighbours, 1)
%          
%          
%      end
     
     for i=1:size(potentialNieghbours, 1)
     
         isNeighbour = max((negList(:, 1) == potentialNeighbours(i, 1)).*...
                        (negList(:, 2) == potentialNeighbours(i, 2)));
         
        if ~isNeighbour
            
            negList = [negList; potentialNeighbours(i, :)];
            noCheckedPixels = noCheckedPixels + 1;
        end
     end
     
     % getting the pixel values of the neighbours
     negValues = zeros(1, size(negList, 1));
     
     for j=1:length(negValues)
         
         negValues(j) = I(negList(j, 1), negList(j, 2));
     end
     
     % detect which neighbour has the highest pixel value, get its coord.
     [~, maxNeighbour]  = max(negValues);
     % keeping only one neighbour if multiple maximums found
     maxNeighbour = maxNeighbour(1);
     
     % setting the found max-valued neighbour as the new seed
     x = negList(maxNeighbour, 1);
     y = negList(maxNeighbour, 2);
     
     % removing the found max-valued neighbour from the neighbour list, and
     % marking him on the region mask
     region_mask(x, y) = 1;
     seedVal = I(x, y);
     
     % the removing from the neighbour list
     for k = 1:size(negList, 1)
     
         if ~((negList(k, 1) == x) && (negList(k, 2) == y))
             temp = [temp; negList(k, :)]
         end
     end
     negList = temp;
     
end

end



