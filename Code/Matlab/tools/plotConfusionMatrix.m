function toto = plotConfusionMatrix(data, Name)
    sizeData = size(data, 1);
    names = arrayfun(@(i) char('A' + i - 1), 1:sizeData, 'UniformOutput', false);

    
    ax = heatmap(names,names,data);
    ax.CellLabelFormat = '%.1f';
    caxis(ax, [0 1]);
    colormap(jet(512));
    title(Name);
end