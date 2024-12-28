%% extracted data from the fault inception time (2.5 Cycles)
clear all

path = "..\..\Complete_Data.xlsx";

data=readtable(path);

%% ploting the 2.5 Cycles
sample_ = 149
% sample_ = 6+1
c_star=[data{sample_+1,2:201}.',data{sample_+1,201:400}.',data{sample_+1,401:600}.']*400/15.75;
t=0.2+[1:200]*250e-6;
% t=[1:200];
c_delta=-[data{sample_+1,602:801}.',data{sample_+1,801:1000}.',data{sample_+1,1001:1200}.'];
% 
idiff=abs(c_star-c_delta);
% ibias=abs(c_star+c_delta)/2; 
% plot(t,idiff(:,3),'-','LineWidth',1.5);
% axis off;
figure();plot([1:200],c_star(:,3),'-','LineWidth',1.5)
% hold;
figure();plot(t,c_delta(:,3));

% Ip=readtable("Star_2.txt");
% Is=readtable("delta_2.txt");
% c_star=[Ip.a,Ip.b,Ip.c]*400/21;

% 
% k= 6;
% B=(1/3)*[2,-1,-1;
% -1,2,-1;
% -1,-1,2];
% A=(2/3)*[cos(k*30),cos((k+4)*30),cos((k-4)*30)
% cos((k-4)*30),cos((k)*30),cos((k+4)*30)
% cos((k+4)*30),cos((k-4)*30),cos((k)*30)];
% 
% d_star=(B*c_star')';
% f_delta=(A*c_delta')';
% figure('Name','After Compensation');
% plot(t,f_delta); hold; plot(t,d_star);
%% Full data from pscad(0.5s)
path = "..\aa.xlsx";

data_2=readtable(path,'Sheet',"S_3_2");
data_3=readtable(path,'Sheet',"S_3_H");

c_star=[data_2{:,2},data_2{:,3}, data_2{:,4}];
c_star_H=[data_3{:,2},data_3{:,3}, data_3{:,4}];
% c_delta=[data_2{:,5},data_2{:,6}, data_2{:,7}]*400/15.75;
t=data_2{:,1}.'+0.2;

% plot(t(:),c_star(:,2),'-','color',[.7 .7 .7],'LineWidth',2.5);
% hold on;
% plot(t(:),c_star_H(:,2),'-','color',[0 0 0],'LineWidth',2.5);

plot(t(:),c_star(:,:),'-','LineWidth',2);
hold on;
plot(t(:),c_star_H(:,2:3),'-.','LineWidth',2.5);

% title('\itAccuracy and Loss during training stage','FontSize',30, 'FontName', 'Times New Roman')
xlabel('\itTime(S)','FontSize',10, 'FontName', 'Times New Roman')
ylabel('\itCurrent(A)','FontSize',10, 'FontName', 'Times New Roman')
% ylim([0 0.5])
set(gca, 'FontSize',15, 'FontName', 'Times New Roman');
% yyaxis right
% plot(T,LOSS,'-','LineWidth',3.5);
% ylabel('\itLoss','FontSize',30, 'FontName', 'Times New Roman')
% set(gca, 'XTick',0.2:0.02:0.21);



% hold on;
% plot(t,c_delta(:,:),'-','LineWidth',1.5);
% axis off;
% ylim([-60 60]);
% ylim([-2 2]);

% idiff=abs(c_star+c_delta);
% figure();plot(t,idiff(:,2),'-','LineWidth',1.5)




