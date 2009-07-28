clear all;
close all;

%Creating the data for validation.
nClasses = 2;
c1 = [randn(1,3000); randn(1,3000)];
c2 = [2.5 + randn(1,3000); 2.5 + randn(1,3000)];
plot(c1(1,:), c1(2,:), 'bo', c2(1,:), c2(2,:), 'ro');
title('Classes Distribution');
xlabel('X');
ylabel('Y');

%Creating the training, validating and testing data sets.
inTrn = {c1(:,1:3:end) c2(:,1:3:end)};
inVal = {c1(:,2:3:end) c2(:,2:3:end)};
inTst = {c1(:,3:3:end) c2(:,3:3:end)};
contInTrn = [inTrn{1} inTrn{2}];
contInVal = [inVal{1} inVal{2}];
contInTst = [inTst{1} inTst{2}];
outTrn = [ones(1, size(inTrn{1},2)) -ones(1, size(inTrn{2},2))];
outVal = [ones(1, size(inVal{1},2)) -ones(1, size(inVal{2},2))];
outTst = [ones(1, size(inTst{1},2)) -ones(1, size(inTst{2},2))];
val.P = contInVal;
val.T = outVal;


%Creating the neural network.
net = newff2(inTrn, outTrn, 2, {'tansig', 'tansig'});
net.trainParam.epochs = 3000;
net.trainParam.max_fail = 20;
net.trainParam.show = 1;
net.trainParam.showWindow = 0;
net.trainParam.showCommandLine = 1;
net.trainParam.deltamax = 50.0;
net.trainParam.min_grad = 1E-6;
net.trainParam.delt_inc = 1.10;
net.trainParam.delt_dec = 0.5;
net.trainParam.delta0 = 0.1;

%Training the networks to be compared.
tic
matNet = train(net, contInTrn, outTrn, [], [], val);
toc
tic
fastNetCont = ntrain(net, contInTrn, outTrn, val.P, val.T);
toc
tic
fastNet = ntrain(net, inTrn, inVal);
toc

%Generating the testing outputs.
matOut = {sim(matNet, inTst{1}) sim(matNet, inTst{2})};
fastNetContOut = {nsim(fastNetCont, inTst{1}) nsim(fastNetCont, inTst{2})};
fastNetOut = {nsim(fastNet, inTst{1}) nsim(fastNet, inTst{2})};

%First analysis: RoC.
nPoints =50000;
[matSP, matCut, matDet, matFa] = genROC(matOut{1}, matOut{2}, nPoints);
[matMaxSP matIdx] = max(matSP);
[fastNetContSP, fastNetContCut, fastNetContDet, fastNetContFa] = genROC(fastNetContOut{1}, fastNetContOut{2}, nPoints);
[fastNetContMaxSP fastNetContIdx] = max(fastNetContSP);
[fastNetSP, fastNetCut, fastNetDet, fastNetFa] = genROC(fastNetOut{1}, fastNetOut{2}, nPoints);
[fastNetMaxSP fastNetIdx] = max(fastNetSP);
figure;
plot(matFa, matDet, 'b-');
hold on;
plot(fastNetContFa, fastNetContDet, 'r-');
plot(fastNetFa, fastNetDet, 'k-');
legend(sprintf('Matlab (SP = %f)', matMaxSP), sprintf('FastNet (cont) (SP = %f)', fastNetContMaxSP), sprintf('FastNet (SP = %f)', fastNetMaxSP));
plot(matFa(matIdx), matDet(matIdx), 'bo');
plot(fastNetContFa(fastNetContIdx), fastNetContDet(fastNetContIdx), 'rx');
plot(fastNetFa(fastNetIdx), fastNetDet(fastNetIdx), 'k*');
title('RoC Between the Algorithms');
xlabel('False Alarm (%)');
ylabel('Detection (%)');

%Second analysis: output distribution.
nBims = 200;
[matH1, matX1] = hist(matOut{1}, nBims);
[matH2, matX2] = hist(matOut{2}, nBims);
[fastNetContH1, fastNetContX1] = hist(fastNetContOut{1}, nBims);
[fastNetContH2, fastNetContX2] = hist(fastNetContOut{2}, nBims);
[fastNetH1, fastNetX1] = hist(fastNetOut{1}, nBims);
[fastNetH2, fastNetX2] = hist(fastNetOut{2}, nBims);
figure;
subplot(1,3,1);
bar(matX1, matH1, 'r');
hold on;
bar(matX2, matH2, 'b');
hold off;
title('Output Distribution for the Matlab Version')
xlabel('Network Output');
subplot(1,3,2);
bar(fastNetContX1, fastNetContH1, 'r');
hold on;
bar(fastNetContX2, fastNetContH2, 'b');
hold off;
title('Output Distribution for the FastNet (Cont) Version')
xlabel('Network Output');
subplot(1,3,3);
bar(fastNetX1, fastNetH1, 'r');
hold on;
bar(fastNetX2, fastNetH2, 'b');
hold off;
title('Output Distribution for the FastNet Version')
xlabel('Network Output');
