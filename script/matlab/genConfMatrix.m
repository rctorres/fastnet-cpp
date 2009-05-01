function ret = genConfMatrix(out, target)
%function ret = genConfMatrix(out, target)
%Returns the confusion matrix of the results obtained. The 'out' and
%'target' vectors are, respectively, the network's output and the desired
%(target) values, which one event per column. The resulting matrix places
%each class in a row, so if ret(2,3) = 5 means that the five events from class 2 
%where classified as being from class 3.
%If only 'out' is provided, then, it must be organized as a cell vector, 
%where the output of each class will be in a cell.
%

if (nargin == 2),
	[Nc,Nev] = size(target);

	%in case of being a two class discriminator, with minimum sparsed output,
	%we add the second output node, with the inverse result.
	if (Nc < 2),
    	out = [out; -out];
    	target = [target; -target];
    	Nc = 2;
	end

	ret = zeros(Nc);
	tot = zeros(Nc, 1);

	for i=1:Nev,
    	[val, It] = max(target(:,i));
    	[val, Io] = max(out(:,i));
    	ret(It, Io) = ret(It, Io) + 1;
    	tot(It) = tot(It) + 1;
	end

	ret = (ret ./ repmat(tot, 1, Nc));
else
	%Getting the number of patterns.
	Nc = length(out);

	%Creating the output matrix.
	ret = zeros(Nc);
	
	%If true, we have two classes with minimum sparsed outputs.
	%So, we create a maximum sparsed output.
	if ( (Nc == 2) && (size(out{1},1) == 1) ),
		for i=1:Nc,
			out{i} = [out{i}; -out{i}];
		end
	end

	binsCenters = [1:Nc];

	for i=1:Nc,
		%Getting the total number of events for each class.
		totEv = size(out{i}, 2);
		%Calculating (in ind) which output node has the highest value.
		[vals, ind] = max(out{i});
		%Saving the ith line of the confusion matrix, ny taking the number of occurrences of each node and dividing by the total number of events in each class.
		ret(i,:) = (hist(ind, binsCenters) ./ totEv);
	end
end
