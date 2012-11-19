clc; clear all; addpath('tools'); close all; warning off;

Base     = '../';
Sequence = {'01_david','02_jumping','03_pedestrian1','04_pedestrian2','05_pedestrian3','06_car','07_motocross','09_carchase','10_panda'};
Name     = {'David','Jumping', 'Pedestrian 1', 'Pedestrian 2', 'Pedestrian 3','Car', 'Motocross', 'Carchase','Panda','mean'};
Tracker  = {'EUGE', 'EUGE2', 'original'};
THR      = 0.25; % threshold on bounding box overlap to declare true positive
show     = [];   % empty/single number, show figures and trajectories of specific sequence and save it
color    = {'b','r'};

% =========================================================================

prec = nan(length(Sequence),length(Tracker));
rec  = nan(length(Sequence),length(Tracker));

if isempty(show), 
    idx = 1:length(Sequence);
else
    idx = show;
end

for ii = idx % for every sequence
    clear ovrlp bb gt;
    sequence = Sequence{ii};
    sequence_directory = [Base sequence filesep];
    files = img_dir(sequence_directory);
    
    % Load ground truth
    
    gt    = dlmread([sequence_directory 'gt.txt'])';
    N     = size(gt,2);
    gt    = bb_cut(gt(1:4,:),N);
    nframe(ii) = size(gt,2); % number of frames to be evaluated
    
    % Load Trackers outputs
    
    in = []; % lists indexes to Trackers that are found in the tested directory
    for i = 1 : length(Tracker)
        filename = [sequence_directory Tracker{i} '.txt'];
        if exist(filename,'file')
            in(end+1) = i;
            bb{i} = dlmread(filename)';
            bb{i} = bb_cut(bb{i}(1:4,:),N);
        else
            bb{i} = [];
        end
    end
    
    % Normalization
    
    for i = in % for every Tracker
        bb{i} = bb_normalize(bb{i},gt);
        [prec(ii,i),rec(ii,i),ovrlp(i,:)] = bb_performance(gt,bb{i},THR);
    end
    
    if ~isempty(show)
        
        out_dir  = Sequence{show}; 
        mkdir(out_dir);
        
        % Figure 1 - overlap curves ---------------------------------------
        
        figure(1), clf; hold on,
        set(gca, 'FontSize', 10, 'FontWeight', 'bold', 'LineWidth', 2)
        title(sequence,'Interpreter','none'), xlabel('frame'), ylabel('bounding box overlap')
        
        for i = in
            plot(ovrlp(i,:), ['.:' color{i}], 'LineWidth', 1,'markersize',5);
            legend_labels{i} = sprintf('%s: prc %3.3f / rec %3.3f', Tracker{i}, prec(ii,i), rec(ii,i));
        end
        legend(legend_labels(in), 'Interpreter', 'none');
        G = gt(1,:); G(~isnan(G)) = 0; plot(G,'linewidth',10,'color','k');
        plot(xlim,[THR THR],'k');
        grid on;
        print(gcf,'-dpng',[out_dir filesep sequence '_overlap.png']);
        
        % Figure 2 - histograms of overlap --------------------------------
        
        figure(2);
        hist(ovrlp(in,:)',20);
        legend(legend_labels(in),'interpreter','none',2);
        title([sequence ', # frames: ' num2str(size(gt,2))],'Interpreter','none'),
        xlabel('overlap'); ylabel('#'); colormap hot;
        xlim([0 1]);
        print(gcf,'-dpng',[out_dir filesep sequence '_histogram.png']);
        
        % Figure 3 - show bounding boxes ----------------------------------
        
        for showidx = 1:size(gt,2)
            figure(3); clf;
            img = imread(files(showidx).name);
            imshow(img); hold on;
            for i = in % for every Tracker
                bb_draw(bb{i}(:,showidx),'edgecolor',color{i},'linewidth',2,'curvature',[.1 .1]);
                cp = bb_center(bb{i}(:,showidx));
                text(cp(1),cp(2),Tracker{i},'color',color{i},'horizontalalignment','center');
                
            end
            img = getframe;
            imwrite(img.cdata,[out_dir filesep num2str(showidx,'%05d') '.png']);
            %bb_draw(gt(:,showidx),'edgecolor','w','linestyle',':','linewidth',2);
        end
        return;
    end
end


%% Comparison table: Precision/Recall/F-measure (LaTeX format)

p = prec;
r = rec;
f = (2*p.*r) ./ (p+r);

p(end+1,:) =  nframe * p ./ sum(nframe); % weighted precision
r(end+1,:) =  nframe * r ./ sum(nframe); % weighted recall
f(end+1,:) =  nframe * f ./ sum(nframe); % weighted area
nframe(end+1) = sum(nframe);

fout = fopen('comparison_table.txt','w');

% Header
fprintf(fout,['Sequence \t& Frames ' ]);
for i = 1:length(Tracker), fprintf(fout,['\t& ' Tracker{i}]); end
fprintf(fout,'\\\\ \n');

for j = 1:size(f,1)
    
    fprintf(fout,[Name{j} ' \t& ' num2str(nframe(j)) ' \t& ']);
    
    for i = 1:size(p,2)
        if f(j,i) == max(f(j,:))
            s = ['\\textbf{' num2str(p(j,i),'%0.2f') ' / ' num2str(r(j,i),'%0.2f') ' / ' num2str(f(j,i),'%0.2f') '}']; % Recall
        else
            s = [num2str(p(j,i),'%0.2f') ' / ' num2str(r(j,i),'%0.2f') ' / ' num2str(f(j,i),'%0.2f')]; % Recall
        end
        
        if isnan(p(j,i))
            fprintf(fout,'-');
        else
            fprintf(fout,s);
        end
        if i ~= size(p,2)
            fprintf(fout, ' \t& ');
        end
    end
    fprintf(fout,' \\\\ \n');
end
fclose(fout);

disp('Results saved to comparison_table.txt');