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

%Creating the data indices.
trnInd = (1:3:nEvents);
valInd = (2:3:nEvents);
tstInd = (3:3:nEvents);

%Creating the training, validating and testing data sets.
trn = {c1(:,trnInd) c2(:,trnInd)};
val = {c1(:,valInd) c2(:,valInd)};
tst = {c1(:,tstInd) c2(:,tstInd)};
in_data = [c1 c2];
out_data = [ones(1,nEvents) -ones(1,nEvents)];

%Creating the neural network.
net = newff2(trn, [-1 1], 2, {'tansig', 'tansig'});
net.trainParam.epochs = 5000;
net.trainParam.batchSize = length(trnInd);
net.trainParam.max_fail = 50;
net.trainParam.show = 1;
net.trainParam.showWindow = true;
net.trainParam.showCommandLine = true;
net.divideParam.trainInd = trnInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = tstInd;

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
    [net evo] = train(net, in_data, out_data);
    etime = toc;
    %Generating the network output after training.
    out = {sim(net, tst{1}) sim(net, tst{2})};
  else
    tic
    [net, evo] = ntrain(net, trn, val, tst);
    etime = toc;
    %Generating the network output after training.
    out = nsim(net, tst);
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
    plot(evo.epoch, evo.mse_trn, 'c', evo.epoch, evo.mse_val, 'm' ,evo.epoch, evo.mse_tst, 'g')
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
