import unittest

import numpy as np
import pandas as pd

# noinspection PyUnresolvedReferences
import eplot.eplot


class EPlotTestCase(unittest.TestCase):
    def setUp(self) -> None:
        n = 20
        self.df = pd.DataFrame([np.random.uniform(1, 10, size=n), np.random.uniform(5, 10, size=n),
                                np.random.randint(1, high=10, size=n), np.random.choice(list('ABCD'), size=n)],
                               index=['col1', 'col2', 'col3', 'col4']).T

    def test_df(self):
        print(self.df.head())

    def test_call(self):
        self.df.eplot(kind='line', path='./html/line_call.html', show_label=False)

    def test_line(self):
        self.df.eplot.line(path='./html/line.html', show_label=False)

    def test_bar(self):
        v = self.df.col4.value_counts()
        v.eplot.bar(title='bar_plot_test', path='./html/bar.html')

    def test_hist(self):
        self.df.col1.eplot.hist(bins=8, path='./html/hist.html')

    # def test_scatter(self):
    #     self.df.eplot.scatter(x='col1', y='col2', path='./html/scatter.html')
    #     self.df.eplot.scatter(x='col1', y='col2', category_col='col4', path='./html/scatter_category.html')
    #
    # def test_scatter_3d(self):
    #     self.df.eplot.scatter3d(x='col1', y='col2', z='col3')
    #     self.df.eplot.scatter3d(x='col1', y='col2', z='col3', category_col='col4')

    def test_pie(self):
        v = self.df.col4.value_counts()
        v.eplot.pie(path='./html/pie.html')
        v.eplot.pie(path='./html/pie_radius.html', pie_config={'radius': ["40%", "75%"]})
        v = self.df.col4.value_counts()
        v.eplot.pie(path='./html/pie_rose.html', pie_config={'radius': ["40%", "75%"], 'rosetype': 'rodius'})
        # self.df.head(10).eplot.pie(y='col2', legend_pos='center', legend_orient='')

    def test_count_plot(self):
        self.df.col4.eplot.count_plot(path='./html/count_plot_test.html')

    def test_box(self):
        self.df.eplot.box(name='val', columns=['col1', 'col2'], path='./html/box_series_test.html')
        self.df.col1.eplot.box(path='./html/box_df_test.html')


if __name__ == '__main__':
    unittest.main()
