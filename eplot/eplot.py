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

    def __call__(self, kind='bar', **kwargs):
        if kind == 'bar':
            return self.bar(**kwargs)
        if kind == 'line':
            return self.line(**kwargs)
        if kind == 'pie':
            return self.pie(**kwargs)
        if kind == 'hist':
            return self.hist(**kwargs)
        if kind == 'box':
            return self.box(**kwargs)
        if kind == 'count_plot':
            return self.count_plot(**kwargs)

    @staticmethod
    def _render_html(chart: RectChart, path: str = None) -> None:
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
                         legend_orient: str = 'horizontal',
                         show_label: bool = True, label_format: str = None) -> Dict[str, Dict[str, Any]]:
        if config is None:
            config = {'title_opts': {}, 'legend_opts': {}, 'tooltips_opts': {}, 'label_opts': {}}
        config['title_opts']['title'] = title
        config['title_opts']['subtitle'] = subtitle

        config['legend_opts']['orient'] = legend_orient
        # config['legend_opts']['pos_left'] = 'left'
        # config['legend_opts']['pos_top'] = 'center'

        config['label_opts']['is_show'] = show_label
        config['label_opts']['formatter'] = label_format

        return config

    @staticmethod
    def _split_data(data: Union[pd.DataFrame, pd.Series], columns: Sequence[str] = None) -> Dict:
        storage = {}
        if isinstance(data, pd.Series):
            storage[data.name] = (data.index.tolist(), data.values.tolist())
        elif isinstance(data, pd.DataFrame):
            columns = list(set(columns) & set(data.columns)) if columns else data.columns
            useful_df = data.loc[:, columns]
            for name, series in useful_df.iteritems():
                x = series.index.tolist()
                y = series.values.tolist()
                storage[name] = (x, y)
        return storage

    # draw funcs
    def bar(self, columns: Sequence[str] = None, path: str = None,
            config: Mapping[str, MutableMapping[str, Any]] = None, flip_xy: bool = False,
            **kwargs) -> Optional[HTML]:

        bar_fig = pyecharts.charts.Bar()
        for name, (x, y) in self._split_data(self._parent, columns).items():
            bar_fig.add_xaxis(x)
            bar_fig.add_yaxis(name, y)

        if flip_xy:
            bar_fig.reversal_axis()

        config = self._combine_options(config, **kwargs)
        bar_fig = self._set_options(bar_fig, config)

        return self._render(bar_fig, path)

    def line(self, columns: Sequence[str] = None, path: str = None,
             config: Mapping[str, MutableMapping[str, Any]] = None,
             line_config: Mapping[str, str] = None, **kwargs) -> Optional[HTML]:
        if line_config is None:
            line_config = {}

        line_fig = pyecharts.charts.Line()
        for name, (x, y) in self._split_data(self._parent, columns).items():
            line_fig.add_xaxis(x)
            line_fig.add_yaxis(name, y, **line_config)

        config = self._combine_options(config, **kwargs)
        line_fig = self._set_options(line_fig, config)
        return self._render(line_fig, path)

    def pie(self, columns: Sequence[str] = None, path: str = None,
            pie_config: Mapping[str, Any] = None,
            config: Mapping[str, MutableMapping[str, Any]] = None, **kwargs) -> Optional[HTML]:
        if pie_config is None:
            pie_config = {}

        pie_fig = pyecharts.charts.Pie()
        for name, (x, y) in self._split_data(self._parent, columns).items():
            pie_fig.add(name, list(zip(x, y)), **pie_config)

        config = self._combine_options(config, **kwargs)
        if config['label_opts']['formatter'] is None:
            config['label_opts']['formatter'] = '{b}:{c}'
        pie_fig = self._set_options(pie_fig, config)
        return self._render(pie_fig, path)

    def hist(self, bins: int = 10, path: str = None,
             config: Mapping[str, MutableMapping[str, Any]] = None, **kwargs) -> Optional[HTML]:
        assert isinstance(self._parent, pd.Series), 'Input must be a pd.Series'

        y, x = np.histogram(self._parent, bins=bins)
        x = [f'{it:0.2f}' for it in x]
        x_labels = [x[i - 1] + '-' + x[i] for i in range(1, len(x))]

        hist_fig = pyecharts.charts.Bar()
        hist_fig.add_xaxis(x_labels)
        hist_fig.add_yaxis(self._parent.name, y.tolist(), category_gap=0)

        config = self._combine_options(config, **kwargs)
        hist_fig = self._set_options(hist_fig, config)

        return self._render(hist_fig, path)

    def box(self, name: str = '', columns: Sequence[str] = None, path: str = None,
            config: Mapping[str, MutableMapping[str, Any]] = None, **kwargs) -> Optional[HTML]:
        box_fig = pyecharts.charts.Boxplot()
        box_fig.add_xaxis([name])
        for col_name, (_, y) in self._split_data(self._parent, columns).items():
            box_fig.add_yaxis(col_name, box_fig.prepare_data([y]))

        config = self._combine_options(config, **kwargs)
        box_fig = self._set_options(box_fig, config)
        return self._render(box_fig, path)

    def count_plot(self, columns: Sequence[str] = None, path: str = None,
                   config: Mapping[str, MutableMapping[str, Any]] = None,
                   flip_xy: bool = False, **kwargs) -> Optional[HTML]:
        return self._parent.value_counts().eplot.bar(columns=columns, path=path, config=config, flip_xy=flip_xy,
                                                     **kwargs)

    def scatter(self, x: str, y: str, category_col=None, path: str = None,
                config: Mapping[str, MutableMapping[str, Any]] = None, **kwargs) -> Optional[HTML]:
        scatter_fig = pyecharts.charts.Scatter()
        if category_col is None:
            x_val = [int(val) for val in self._parent[x].values.tolist()]
            y_val = [float(val) for val in self._parent[y].values.tolist()]
            scatter_fig.add_xaxis(x_val)
            scatter_fig.add_yaxis(y, y_val)
        else:
            for cat, d in self._parent.groupby(category_col):
                scatter_fig.add_xaxis(d[x].values.tolist())
                scatter_fig.add_yaxis(cat, d[y].values.tolist())
        config = self._combine_options(config, show_label=False, **kwargs)
        scatter_fig = self._set_options(scatter_fig, config)
        return self._render(scatter_fig, path)

    def scatter3d(self, x, y, z, category_col=None, title=None, category_name=None, **kwargs) -> Optional[HTML]:
        data = self._parent
        scatter3d_fig = pyecharts.charts.Scatter3D(title)
        if category_col is None:
            scatter3d_fig.add(category_name, data[[x, y, z]].values, **kwargs)
        else:
            for cat, d in data.groupby(category_col):
                scatter3d_fig.add(cat, d[[x, y, z]].values, **kwargs)
        return scatter3d_fig


pd.Series.eplot = CachedAccessor("eplot", EchartsBasePlotMethods)
pd.DataFrame.eplot = CachedAccessor("eplot", EchartsBasePlotMethods)
