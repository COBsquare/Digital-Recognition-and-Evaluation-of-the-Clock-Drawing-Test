function clockscore = scoreclock(file)
%scoreclock Detects and scores the hand drawn clock image
%   For the initial version, we pass the file name as a parameter.

clockscore = []; 

% Read the image, make it grayscale
% In contrast to RGB (red, green, blue) images that have three
% layers, grayscale images have one layer. 
I=rgb2gray(imread(file)); 
% imhist creates 256 bins and counts how many pixels each one 
% of them has. We are interested with the max values, because 
% think that the white A4 paper will cover most of the image
% and bright pixels will be abundant. Therefore, after imhist,
% I find the index of the maximum value. 
[c,~] = imhist(I);
[~,mi] = max(c(:));


% I need a threshold value to create a binary image. All values
% below that value will be 0, and higher will be 1. To decide
% that point, I start from the peak index and go left (lower
% values) until the values begin to increase 3 times. Then, I 
% stop, and I create a binary image using that value. 
devam = true;
tol = 0; % tolerance - a few bins can be increasing.
while devam
    cx = c(mi); % previous
    mi = mi - 1; % go left
    if c(mi) > cx && tol > 3
        devam = false;
    elseif c(mi) > cx 
        tol = tol + 1;
    end
end

% mi is my threshold after the loop is over.

Ib = I > mi; % binary, but still large
% open image to close gaps
% the structuring element should be proportional to the image. 
Io = imopen(Ib, strel('rectangle', floor(size(Ib)*0.01))); % it was [6 6]

% I find the connected components. These are like the bucket
% fill in Paint program. The command checks all of the adjacent
% pixels and if they have the value of 1, then the command marks
% them as the same region. 
CC = bwconncomp(Io);
% I use the regionprops command to find the Areas and the
% BoundingBoxes of the connected components. I find the largest
% area and get its bounding box to display it. 
PA = struct2array(regionprops(CC, 'Area'));
PB = regionprops(CC, 'BoundingBox');

[~, aidx] = max(PA(:));
PC = floor(PB(aidx).BoundingBox);
%imshow(Ib); 
%hold on;
%rectangle('Position', PB(aidx).BoundingBox, 'EdgeColor', 'yellow');
%hold off;
s1 = PC(1);
s2 = PC(2);
% rounding down generates a value of 0 which is non-existent in 
% matlab arrays/matrices
if(s1 == 0) 
    s1 = 1;
end
if(s2==0)
    s2=1;
end
piece = Ib(s2:PC(2)+PC(4), s1:PC(1)+PC(3));
% same for imopen here - make it proportional to the image instead
% of magic numbers, such as [100 100]
circles = ~imopen(piece,strel('rectangle',floor(size(Ib)*0.1)));
CC = bwconncomp(circles);
% we should get as many as we can from the regionprops of this conncomp...
PA = regionprops(CC, 'Area');
PB =  regionprops(CC, 'BoundingBox');
PC = regionprops(CC, 'Centroid'); 
PO = regionprops(CC, 'Orientation');

idx = 1;
prarea = 0;
for i=1:size(PA,1)
    area = PA(i).Area;
    bb = floor(PB(i).BoundingBox);
    rratio = bb(3)/bb(4);
    %rarea = (bb(3)/2 * bb(4)/2 * pi)/area
    %if rarea > 0.9 && rarea < 1.1 && rratio > 0.7 && rratio < 1.3 && area > prarea
    if rratio > 2/3 && rratio < 3/2 && area > prarea
        idx = i;
        prarea = area;
    end
end

% idx has the circle
% Use PA, PB, PC, PO to come up with a score about the circle. 
% PC(idx)
orient = PO(idx).Orientation;
% Once we crop the image, the coordinates will be reset. We need the 
% centroid coordinates relative to the bounding box. 

bb = floor(PB(idx).BoundingBox);
bc = floor(PC(idx).Centroid);
merkez = [bc(1)-bb(1) bc(2)-bb(2)];

% further crop the image to the circle:
IC = ~piece(bb(2):bb(2)+bb(4), bb(1):bb(1)+bb(3)); % invert: ~
rad = floor(max(size(IC))/2);

% SCORE: for the circle we will take the average of two radii and the
% orientation. 

clockscore(1) =  min(size(IC))/max(size(IC));
clockscore(2) =  abs(abs(orient)-45)/45; %(1-((abs(orient)-45)/45));

% Find the connected components, then loop through them to find the 
% numbers. 
% Here's the proposed algorithm; for each number, 1 to 12, loop through
% the components, find the one which fits the number the most. Then, 
% check its position, if it's in the correct region. To do so, find the
% line from its centroid, to the IC's centroid. 

CC = bwconncomp(IC);
PB = regionprops(CC, 'BoundingBox');
NC = regionprops(CC, 'Centroid');

load trained.mat; % trained A.N.N.
b1 = 16;
b2 = 16;

rs = size(PB,1);
marked = zeros(size(PB,1));
acilar = [ 75 45 15 345 315 285 255 225 195 165 135 105 75 ]';
% candidates is a 3d variable where we keep the matching values from
% the neural network, and the angles related to the center of the clock,
% the distance to the center of the clock

% once we go over all of the numbers, 1 to 0, we can use this information
% to form 10, 11, 12, and also get an evaluation value for them. 
evas = zeros(12, size(PB,1),3); 

 % we need another loop for 10, 11, 12 - but we need to find candidates
 % for 0, too. So, loop up to 10, then we'll handle 10, 11, 12
for i=1:12 
    %mi = 0;
    if(rs > 50) 
        continue;
    end
    for j=1:rs
        mi=0;
        if(marked(j) == 1) % we already marked this region, continue...
            continue;
        end
        
        bb = floor(PB(j).BoundingBox);
        tm = IC(bb(2):bb(2)+bb(4), bb(1):bb(1)+bb(3));
        if size(tm,1) == size(IC,1) && size(tm,2) == size(IC,2) % largest one
            marked(j) = 1; 
            continue;
        end
        
        if (i<=10) % at least give us a clue about 10, 11 and 12
            % consider rotations
            for k=-20:5:20
                
                m2 = imrotate(tm, k);
                % check to see if it matches any number
                mm = imresize(m2, [b1 b2])';
                mm = reshape(mm, [b1*b2 1]);
                cvp = net(mm);
                v = cvp(i);
                if(v > mi)
                    %mj = j;
                    mi = v;
                    evas(i, j, 1) = v; % save it here
                end
            end
        end
        
        % we are looking for the numeral i, we have 12 numbers on the 
        % clock. Each gets 30 degress, starting from -15 to 15. 
        ic = floor(NC(j).Centroid);
        
        % images have the top-left corner as (0,0)
        vk = [ic(1)-merkez(1) merkez(2)-ic(2)];
        evas(i,j,3) = floor(norm(vk))/rad; % ratio of distance to center
        vk = vk/norm(vk);
        aci = atand(vk(2)/vk(1));
        if aci > 0 % first and third quadrants
            if vk(1) < 0 % either both is negative or none is.
                % third quadrant
                aci = 180 + aci;
            end
        else
            if vk(1) < 0
                % second quadrant
                aci = 180 + aci;
            else
                % fourth quadrant
                aci = 360 + aci;
            end
        end
        
        acifark = NaN;
        if i == 3 % number three is right at 0 degrees, so be careful
            if aci < 16
                acifark = aci;
            elseif aci > 344
                acifark = 360-aci;
            end
        else
            if aci < acilar(i) && aci > acilar(i+1)
                acifark = abs(aci - ( (acilar(i) + acilar(i+1))/2));
            end
        end
        evas(i, j, 2) = acifark; 
    end
end % end of number search

% evas will be very helpful. use both the ANN and angle difference
% to get an almost exact match! insert happy face here!
points = zeros(12,1);
for i=1:12
    % check if we have a candidate for location. 
    idx=find(evas(i,:,2)>0);
    [vals, ids] = sort(evas(i,:,1));
    found = 0;
    if (size(idx,2) == 0) 
        % then we only have one candidate, the largest of the 
        % matching region from ANN. 
        
        % Even though we don't have the quadrant, check if it is close to
        % the center or not. If it is, then move on to the next candidate
        % in vals. 
        rsd = rs;
        while (evas(i,ids(rsd),3) < 0.5)
            rsd = rsd - 1;
        end
        found = vals(rsd);
        fid = ids(rsd);
        %disp(['for ' int2str(i) ' there is no position candidate, using matching region with id: ' int2str(fid)]);
        points(i) = found;
    elseif (size(idx,2) == 1)
        % there is only one candidate - if the distance to the center and
        % the ANN reply agrees, then set it as the sought number. 
        if (evas(i,idx,3) < 0.5)
            rsd = rs;
            while(evas(i,ids(rsd),3) < 0.5)
                rsd = rsd - 1;
            end
            found = vals(rsd);
            fid = rsd;
        else
            found = evas(i,idx,1) + (1 - evas(i, idx, 2)/45); 
            fid = idx;
        end
        points(i) = found;
        %disp(['for ' int2str(i) ' there is only one position candidate, using matching position with id: ' int2str(fid)]);
    else
        % more than one candidate for position. choose the least angle
        % be careful, there is a value of 0 for the largest circle.
        
        mx = 0;
        fid = 0;
        for j=1:size(idx,2)
            if(evas(i,idx(j),3) > 0.5 && evas(i,idx(j),2) > mx)
                mx = evas(i,idx(j),2);
                fid = idx(j);
                found = evas(i,idx(j),1) + (1 - evas(i, idx(j),2)/45);
            end
        end
        
        if (fid == 0) % then none of these are far enough from center
            rsd = rs;
            while(evas(i,ids(rsd),3) < 0.5)
                rsd = rsd - 1;
            end
            found = vals(rsd);
            fid = rsd;
        end
        %found = evas(i,pids(2),1);
        %fid = pids(2);
        %disp(['for ' int2str(i) ' there was more than one position candidate, using position with the least angle with id: ' int2str(fid)]);
        points(i) = found;
    end
    %showRegion(IC,PB,fid);
    %pause;
end

clockscore(3) = sum(points(:))/6; % points has a maximum of 24, divide by 6

% NOW ON TO THE LINES
% it works fine as it is - just make sure they are 
% as close as possible to the center of the circle, and 
% their length is as long as possible...

% here's a new idea
% loop through the connected components, run the hough transform for the
% components close enough to the center... 

akr = []; % coordinates - will be used to find the angle between them
yel = [];
akrs = []; % final versions
yels = [];
al = 0; % akrep length
yl = 0;
ca = false;
cy = false;
for i=1:rs
    % we already have the distance ratio in evas.
    bb = floor(PB(i).BoundingBox);
    tm = IC(bb(2):bb(2)+bb(4), bb(1):bb(1)+bb(3));
    if max(size(tm)./size(IC)) > 0.8 % this covers more than 0.8 of the clock face
        continue;
    end
    
    f = merkez - [
        bb(2) bb(1)
        bb(2) bb(1)+bb(3)
        bb(2)+bb(4) bb(1)
        bb(2)+bb(4) bb(1)+bb(3)
    ];
    mx = min([norm(f(1,:)) norm(f(2,:)) norm(f(3,:)) norm(f(4,:))]);
    % is it close enough
    if mx / rad < 0.5
        [h,t,r] = hough(tm);
        P=houghpeaks(h,5,'threshold', ceil(0.3*max(h(:))));
        lines=houghlines(tm,t,r,P,'FillGap',5,'MinLength',7);
        for k=1:length(lines)
            xy=[lines(k).point1; lines(k).point2];
            ff = lines(k).point2 - lines(k).point1;
            ll = norm(ff);
            %dd = min(norm(lines(k).point1 - merkez), norm(lines(k).point2 - merkez));
            
            dotakr = 0;
            dotyel = 0;
            if(isempty(akr) == 0 && isempty(yel) == 0)
                f2 = yel(2,:) - yel(1,:);
                dotakr = dot(ff/norm(ff), f2/norm(f2));
            end
            if(isempty(yel) == 0 && isempty(akr) == 0)
                f2 = akr(2,:) - akr(1,:);
                dotyel = dot(ff/norm(ff), f2/norm(f2));
            end
            if ll > yl &&  dotyel < 0.5
                yl = ll;
                yel = xy;
                cy = true;
            elseif ll > al &&  dotakr < 0.5
                al = ll;
                akr = xy;
                ca = true;
            end
        end
        if ca
            akrs = [ akr(:,1) + bb(1) akr(:,2) + bb(2) ];
            ca = false;
        end
        if cy
            yels = [ yel(:,1) + bb(1) yel(:,2) + bb(2) ];
            cy = false;
        end
    end
end

if(isempty(akrs) == 0)
    clockscore(4) = 1;
end
if(isempty(yels) == 0)
    clockscore(5) = 1;
end

if(isempty(akrs) == 0 && isempty(yels) == 0)

    expectedangle = angleClockHands(10,10);
    % top left corner is 0,0
    % make bottom left corner 0,0
    a1 = [akrs(:,1) size(IC,1)-akrs(:,2)];
    y1 = [yels(:,1) size(IC,1)-yels(:,2)];
    
    if(a1(1,2) > a1(2,2))
        a2 = a1(1,:) - a1(2,:);
    else
        a2 = a1(2,:) - a1(1,:);
    end
    
    if(y1(1,2) > y1(2,2))
        y2 = y1(1,:) - y1(2,:);
    else
        y2 = y1(2,:) - y1(1,:);
    end
    
    a3 = a2 / norm(a2);
    y3 = y2 / norm(y2);
    
    ayangle = acosd(dot(a3,y3));
    cangle = 1 - abs(expectedangle - ayangle)/180;
    if cangle < 0
        cangle = 0;
    end
    
    clockscore(6) = cangle;
    
    ma = min( norm(akrs(1,:)-merkez), norm(akrs(2,:)-merkez) );
    my = min( norm(yels(1,:)-merkez), norm(yels(2,:)-merkez) );
    mm = max(size(IC))*0.25;
    clockscore(7) = ((1- (ma/mm)) + (1-(my/mm)))/2;
end

end

