import unittest

import numpy as np
import pandas as pd

# noinspection PyUnresolvedReferences
import eplot.eplot


class EPlotTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame([np.random.uniform(10, 1, size=1000), np.random.uniform(10, 5, size=1000),
                                np.random.randint(1, high=10, size=1000), np.random.choice(list('ABCD'), size=1000)],
                               index=['col1', 'col2', 'col3', 'col4']).T

    def test_df(self):
        print(self.df.head())

    def test_line(self):
        self.df.eplot.line(path='./img/line_plot_test.png')

    def test_bar(self):
        v = self.df.col4.value_counts()
        v.eplot.bar(title='bar_plot_test', path='./img/bar_plot.png')

    def test_hist(self):
        self.df.col1.eplot.hist(bins=8)

    def test_scatter(self):
        self.df.eplot.scatter(x='col1', y='col2')
        self.df.eplot.scatter(x='col1', y='col2', category_col='col4')

    def test_scatter_3d(self):
        self.df.eplot.scatter3d(x='col1', y='col2', z='col3')
        self.df.eplot.scatter3d(x='col1', y='col2', z='col3', category_col='col4')

    def test_pie(self):
        v = self.df.col4.value_counts()
        v.eplot.pie('pie_plot_test.html')
        v.eplot.pie('pie_plot_test_radius.html', radius=["40%", "75%"])
        # v = self.df.col4.value_counts()
        # v.eplot.pie(inner_radius_from=30, rosetype='rodius')
        # self.df.head(10).eplot.pie(y='col2', legend_pos='center', legend_orient='')

    def test_count_plot(self):
        self.df.col4.eplot.countplot()

    def test_box(self):
        self.df.eplot.box()
        self.df.col1.eplot.box()


if __name__ == '__main__':
    unittest.main()
