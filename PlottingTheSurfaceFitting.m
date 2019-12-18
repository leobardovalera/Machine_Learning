close all

lay1 = load('layer_1_vals.txt');
lay1_bias = load('layer_1_bias_vals.txt');
lay1 = lay1';
lay2 = load('layer_2_vals.txt');
lay2 = lay2';
lay2_bias = load('layer_2_bias_vals.txt');

lay3 = load('layer_3_vals.txt');
lay3 = lay3';
lay3_bias = load('layer_3_bias_vals.txt');


output = load('output_layer_vals.txt');
output = output';
output_bias = load('output_layer_bias_vals.txt');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load Input.tex
load Target.tex

plot3(Input(:,1), Input(:,2), Target(:,1),'.','Markersize',20)

hold on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = linspace(-2,2,100);
y = x;
[X,Y] = meshgrid(x,y);

XX = [X(:)'; Y(:)'];

Pred = output*(sigmoid((lay3*(sigmoid(lay2*(sigmoid(lay1*XX+lay1_bias))+lay2_bias))+lay3_bias)))+output_bias;

Pred = reshape(Pred,length(x),length(x));

sf = surf(X, Y, Pred);
grid on

set(sf,'MeshStyle','none','FaceAlpha',0.9)









