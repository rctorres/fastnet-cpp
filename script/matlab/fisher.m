function w = fisher(x1, x2)
%function w = fisher(x1, x2)
%Implements a linear fisher discriminant
%This function will receive 2 classes, x1 and x2, where each event is a
%collumn, and will return the row vector w containing the direction that
%maximizes the discrimination between x1 and x2.
%

%Calculating the means 
m1 = mean(x1,2);
m2 = mean(x2,2);

%Calculting the centered versions of x1 and x2.
c1 = x1 - repmat(m1,1,size(x1,2));
c2 = x2 - repmat(m2,1,size(x2,2));

%Calculating the scatter matrices.
s1 = c1*c1';
s2 = c2*c2';
sw = s1 + s2;

%Calculating w with unit norm.
w = inv(sw) * (m1 - m2);
w = (w ./ norm(w))';
