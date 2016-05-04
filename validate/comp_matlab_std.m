%clear all;
%close all;

%Creating the data for validation.
nClasses = 2;
nEvents = 3000;
c1 = [randn(1,nEvents); randn(1,nEvents)];
c2 = [2.5 + randn(1,nEvents); 2.5 + randn(1,nEvents)];
plot(c1(1,:), c1(2,:), 'bo', c2(1,:), c2(2,:), 'ro');
title('Classes Distribution');
legend('Class 1', 'Class 2');
xlabel('X');
ylabel('Y');

%Matlab data sets.
in_data = [c1 c2];
out_data = [ones(1,nEvents) -ones(1,nEvents)];

%Creating the data indices. So the FastNet and Matlab will be trained,
%validated and tested with the same data sets.
trnInd = 1:3:size(in_data,2);
valInd = 2:3:size(in_data,2);
tstInd = 3:3:size(in_data,2);

%Creating datasets.
in_trn = in_data(:,trnInd);
in_val = in_data(:,valInd);
in_tst = in_data(:,tstInd);

out_trn = out_data(trnInd);
out_val = out_data(valInd);
out_tst = out_data(tstInd);

%Creating the neural network.
net = newff2(in_trn, [-1 1], 2, {'tansig', 'tansig'});
net.trainParam.epochs = 5000;
net.trainParam.batchSize = size(in_trn,2);
net.trainParam.max_fail = 50;
net.trainParam.show = 1;
net.trainParam.showWindow = false;
net.trainParam.showCommandLine = true;
net.divideParam.trainInd = trnInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = [];

cases = {'mat' 'fn'};
color = 'br';
rocFig = figure;
outFig = figure;
evoFig = figure;

for i=1:length(cases),
  c = cases{i};
  col = color(i);
  isMat = strcmp(c, 'mat');
  
  %Training the networks to be compared.
  if isMat,
    tic
    [onet evo] = train(net, in_data, out_data);
    etime = toc;
    %Generating the network output after training.
    out = {sim(onet, in_tst(:, out_tst == 1)) sim(onet, in_tst(:, out_tst == -1))};
  else
    tic
    [onet, evo] = ntrain(net, in_trn, out_trn, in_val, out_val);
    etime = toc;
    %Generating the network output after training.
    out = {nsim(onet, in_tst(:, out_tst == 1)) nsim(onet, in_tst(:, out_tst == -1))};
  end
 
  %First analysis: RoC.
  figure(rocFig);
  [sp, cut, det.(c), fa.(c)] = genROC(out{1}, out{2});
  [maxSP.(c) spIdx.(c)] = max(sp);
  plot(100*fa.(c), 100*det.(c), col);
  hold on;
  
  %Second analysis: output distribution.
  figure(outFig);
  subplot(2,1,i);
  nBims = 200;
  hist(out{1}, nBims);
  hold on;
  hist(out{2}, nBims);
  hold off;
  h = findobj(gca, 'Type', 'patch');
  set(h(1), 'FaceColor', 'r');
  title(sprintf('Output Distribution (%s)', c));
  xlabel('Network Output');
  ylabel('Counts');
  legend(sprintf('C1 (%f+-%f)', mean(out{1}), std(out{1})), sprintf('C2 (%f+-%f)', mean(out{2}), std(out{2})), 'Location', 'North');
  
  %Third analysis: training evolution.
  figure(evoFig);
  if isMat,
    plot(evo.epoch, evo.perf, 'b', evo.epoch, evo.vperf, 'r' ,evo.epoch, evo.tperf, 'k')
    hold on;
  else
    plot(evo.epoch, evo.mse_trn, 'c', evo.epoch, evo.mse_val, 'm')
    hold on;  
  end
end

figure(rocFig);
grid on;
legend(sprintf('Matlab (SP = %f)', 100*maxSP.mat), sprintf('FastNet (SP = %f)', 100*maxSP.fn), 'Location', 'SouthEast');
plot(100*fa.mat(spIdx.mat), 100*det.mat(spIdx.mat), 'b*');
plot(100*fa.fn(spIdx.fn), 100*det.fn(spIdx.fn), 'r*');
hold off;
title('RoC');
xlabel('False Alarm (%)');
ylabel('Detection (%)');


figure(evoFig);
hold off;
grid on;
set(gca, 'XScale', 'Log');
title('Training Evolution');
xlabel('Epochs');
ylabel('MSE');
legend('trn (mat)', 'val (mat)', 'tst (mat)', 'trn (fn)', 'val (fn)', 'tst (fn)');
