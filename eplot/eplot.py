from typing import Sequence, Mapping, Any, MutableMapping, Optional, Dict, Union

import numpy as np
import pandas as pd
import pyecharts
import pyecharts.options as opts
from pandas.core.accessor import CachedAccessor
from pandas.core.base import PandasObject
from pyecharts.charts.chart import RectChart
from pyecharts.commons import utils
from pyecharts.render.display import HTML
from pyecharts.render.engine import RenderEngine


class EchartsBasePlotMethods(PandasObject):
    def __init__(self, data):
        self._parent = data  # can be Series or DataFrame

    def __call__(self, kind='bar', title=None, bins=10, **kwargs):
        if kind == 'bar':
            return self.bar(title=title, **kwargs)
        if kind == 'line':
            return self.line(title=title, **kwargs)
        if kind == 'pie':
            return self.pie(title=title, **kwargs)
        if kind == 'hist':
            return self.hist(title=title, bins=bins, **kwargs)
        if kind == 'box':
            return self.box(title=title, **kwargs)
        if kind == 'count_plot':
            return self.count_plot(title=title, **kwargs)

    @staticmethod
    def _render_html(chart: RectChart, path: str = None):
        if path is None:
            path = "render.html"
        template_name = "simple_chart.html"
        chart._prepare_render()
        RenderEngine().render_chart_to_file(chart=chart, path=path, template_name=template_name)

    @staticmethod
    def _render_to_notebook(chart: RectChart) -> HTML:
        chart._prepare_render()
        require_config = utils.produce_require_dict(
            chart.js_dependencies, chart.js_host
        )
        return HTML(
            RenderEngine().render_chart_to_notebook(
                template_name="jupyter_notebook.html",
                charts=(chart,),
                config_items=require_config["config_items"],
                libraries=require_config["libraries"]))

    @staticmethod
    def _set_options(figure, config: Mapping[str, Mapping[str, Any]] = None):
        figure.set_global_opts(title_opts=opts.TitleOpts(**config['title_opts']),
                               legend_opts=opts.LegendOpts(**config['legend_opts']),
                               tooltip_opts=opts.TooltipOpts(**config['tooltips_opts'])
                               )
        figure.set_series_opts(label_opts=opts.LabelOpts(**config['label_opts']))
        return figure

    @staticmethod
    def _combine_options(config: Mapping[str, MutableMapping[str, Any]] = None,
                         title: str = None, subtitle: str = None,
                         legend_pos: str = 'center', legend_orient: str = 'vertical',
                         label_format: str = None) -> Mapping[str, Mapping[str, Any]]:
        if config is None:
            config = {'title_opts': {}, 'legend_opts': {}, 'tooltips_opts': {}, 'label_opts': {}}
        config['title_opts']['title'] = title
        config['title_opts']['subtitle'] = subtitle
        config['legend_opts']['legend_pos'] = legend_pos
        config['legend_opts']['legend_orient'] = legend_orient
        config['label_opts']['formatter'] = label_format

        return config

    @staticmethod
    def _split_data(data: Union[pd.DataFrame, pd.Series], columns: Sequence[str] = None) -> Dict:
        if columns:
            columns = list(set(columns) & set(data.columns))
        storage = {}
        if isinstance(data, pd.Series):
            storage[data.name] = (data.index.tolist(), data.values.tolist())
        elif isinstance(data, pd.DataFrame):
            useful_df = data.loc[:, columns]
            for name, series in useful_df.iteritems():
                x = series.index.tolist()
                y = series.values.tolist()
                storage[name] = (x, y)
        return storage

    def _render(self, chart, path: str = None) -> Optional[HTML]:
        if path is None:
            return self._render_to_notebook(chart)
        if path.split('.')[-1] == 'html':
            # todo: debug
            self._render_html(chart, path)
        elif path.split('.')[-1] in ['png', 'jpg', 'jpeg']:
            from pyecharts.render import make_snapshot
            from snapshot_selenium import snapshot
            # noinspection PyUnresolvedReferences
            import chromedriver_binary

            make_snapshot(snapshot, chart.render(), path)
        else:
            raise ValueError('Unknown Format, only [html, png, jpg, jpeg] supported')

    def bar(self, columns: Sequence[str] = None, path: str = None,
            config: Mapping[str, MutableMapping[str, Any]] = None,
            **kwargs) -> Optional[HTML]:

        bar_fig = pyecharts.charts.Bar()
        for name, (x, y) in self._split_data(self._parent, columns).items():
            bar_fig.add_xaxis(x)
            bar_fig.add_yaxis(name, y)

        config = self._combine_options(config, **kwargs)
        bar_fig = self._set_options(bar_fig, config)

        return self._render(bar_fig, path)

    def line(self, columns: Sequence[str] = None, path: str = None,
             config: Mapping[str, MutableMapping[str, Any]] = None,
             line_config: Mapping[str, str] = None, **kwargs) -> Optional[HTML]:

        line_fig = pyecharts.charts.Line()
        for name, x, y in self._split_data(self._parent, columns):
            line_fig.add_xaxis(x)
            line_fig.add_yaxis(name, y, **line_config)

        config = self._combine_options(config, **kwargs)
        line_fig = self._set_options(line_fig, config)
        return self._render(line_fig, path)

    def pie(self, columns: Sequence[str] = None, path: str = None,
            config: Mapping[str, MutableMapping[str, Any]] = None, **kwargs) -> Optional[HTML]:

        pie_fig = pyecharts.charts.Pie()
        for name, (x, y) in self._split_data(self._parent, columns).items():
            pie_fig.add(name, list(zip(x, y)), **kwargs)

        config = self._combine_options(config, **kwargs)
        pie_fig = self._set_options(pie_fig, config)
        return self._render(pie_fig, path)

    def hist(self, title: str = None, bins: int = 10, **kwargs):
        data = self._parent
        hist_fig = pyecharts.charts.Bar()
        hist_fig.set_global_opts(title_opts=opts.TitleOpts(title=title))
        y, x = np.histogram(data, bins=bins)
        x = x.astype(int).astype(str)
        x_labels = [x[i - 1] + '-' + x[i] for i in range(1, len(x))]
        hist_fig.add_xaxis(x_labels)
        hist_fig.add_yaxis(data.name, y.tolist(), **kwargs)
        return self._render_to_notebook(hist_fig)

    def box(self, title=None, **kwargs):
        data = self._parent
        box_fig = pyecharts.charts.Boxplot()
        box_fig.set_global_opts(title_opts=opts.TitleOpts(title=title))
        if isinstance(data, pd.Series):
            box_fig.add_xaxis([data.name])
            box_fig.add_yaxis('', box_fig.prepare_data(data.values.reshape((1, -1)).tolist()))
        elif isinstance(data, pd.DataFrame):
            box_fig.add_xaxis(data.columns.tolist())
            box_fig.add_yaxis('', box_fig.prepare_data(data.values.T.tolist()))
        return self._render_to_notebook(box_fig)

    def count_plot(self, title=None, **kwargs):
        return self._parent.value_counts().eplot.bar(title=title, **kwargs)

    def scatter(self, x, y, category_col=None, title=None, category_name=None, **kwargs):
        data = self._parent
        scatterFig = pyecharts.charts.Scatter()
        scatterFig.set_global_opts(title_opts=opts.TitleOpts(title=title))
        if category_col is None:
            (scatterFig.add_xaxis(data[x].values.tolist())
             .add_yaxis('', data[y].values.tolist())
             .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
             )
        else:
            for cat, d in data.groupby(category_col):
                # scatterFig.add(cat, d[x], d[y], **kwargs)
                (scatterFig.add_xaxis(d[x].values.tolist())
                 .add_yaxis(cat, d[y].values.tolist())
                 .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
                 )
        return self._render_to_notebook(scatterFig)

    def scatter3d(self, x, y, z, category_col=None, title=None, category_name=None, **kwargs):
        data = self._parent
        scatter3dFig = pyecharts.charts.Scatter3D(title)
        if category_col is None:
            scatter3dFig.add(category_name, data[[x, y, z]].values, **kwargs)
        else:
            for cat, d in data.groupby(category_col):
                scatter3dFig.add(cat, d[[x, y, z]].values, **kwargs)
        return scatter3dFig


pd.Series.eplot = CachedAccessor("eplot", EchartsBasePlotMethods)
pd.DataFrame.eplot = CachedAccessor("eplot", EchartsBasePlotMethods)
