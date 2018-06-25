clear all;
close all;
clc;

alpha=10;
ite_time=100;

% 读取数据
data = load('ex2data1.txt');
x = data(:, [1, 2]); y = data(:, 3);   % 成绩和录取结果

% 在图中表示出原始数据
figure 
for i=1:length(x)
    if y(i)==1
      plot(x(i,1),x(i,2),'r*')
    else
      plot(x(i,1),x(i,2),'bo')  
    end
    hold on
end

% 归一化
x_processed(:,1)=(x(:,1)-min(x(:,1)))/(max(x(:,1)-min(x(:,1))));
x_processed(:,2)=(x(:,2)-min(x(:,2)))/(max(x(:,2)-min(x(:,2))));

% 在图中表示出归一化处理后的数据
figure 
for i=1:length(x)
    if y(i)==1
      plot(x_processed(i,1),x_processed(i,2),'r*')
    else
      plot(x_processed(i,1),x_processed(i,2),'bo')  
    end
    hold on
end

% 梯度下降训练参数
theta=zeros(3,1);
for i=1:ite_time
    temp0=0;temp1=0;temp2=0;
    for j=1:length(x)
        temp0=temp0+sigmoid(theta(1)+theta(2)*x_processed(j,1)+theta(3)*x_processed(j,2))-y(j);
        temp1=temp1+(sigmoid(theta(1)+theta(2)*x_processed(j,1)+theta(3)*x_processed(j,2))-y(j))*x_processed(j,1);
        temp2=temp2+(sigmoid(theta(1)+theta(2)*x_processed(j,1)+theta(3)*x_processed(j,2))-y(j))*x_processed(j,2);
    end
    theta=theta-alpha*[temp0;temp1;temp2]/length(x);
end

% 作出预测线
plot(0:0.01:1,-1/theta(3)*(theta(1)+theta(2).*(0:0.01:1)),'m-')

%计算正确率
count=0;
for i=1:length(x)
    if (theta(1)+theta(2)*x_processed(i,1)+theta(3)*x_processed(i,2)>0&&y(i)==1)||...
        (theta(1)+theta(2)*x_processed(i,1)+theta(3)*x_processed(i,2)<0&&y(i)==0)
        count=count+1;
    end
end
%输出正确率
fprintf('Accuracy: %f \n',count/length(x));

