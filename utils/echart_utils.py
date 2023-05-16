import numpy as np
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line, Bar
from pyecharts.globals import ThemeType

from utils.DateUtils import datetime2str


def render_echarts(data, params, save_path):
    line = (
        Line(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width='792px', height='464px'))
            .add_xaxis([datetime2str(x.to_pydatetime()) for x in list(data.index)])
            .add_yaxis('Original', data.values.flatten(), linestyle_opts=opts.LineStyleOpts(color='blue'),
                       is_symbol_show=False)
            .set_global_opts(
            title_opts=opts.TitleOpts(
                title='城市:{}  作物:{}  采样频率:{}  数据量:{}'.format(params['city'], params['crop'], params['freq'], len(data)),
                subtitle="折线图"),
            tooltip_opts=opts.TooltipOpts(trigger='axis', axis_pointer_type='cross'),
            toolbox_opts=opts.ToolboxOpts(is_show=True, orient='vertical', pos_left='right'),
            legend_opts=opts.LegendOpts(pos_top=30)
        )
            .set_colors(['blue'])
    )
    # 保存至本地
    line.render(save_path)


def render_predict(model, params, original, train_predict, test_predict, future, train_RMSE, test_RMSE, save_path):
    pre = pd.Series([np.nan for x in range(len(train_predict))])
    test_predict = pd.concat([pre, test_predict], axis=0)
    pre = pd.Series([np.nan for x in range(len(original))])
    future_ = pd.concat([pre, future], axis=0)
    new_index = pd.concat([original, future], axis=0)
    line = (
        Line(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width='792px', height='464px'))
            .add_xaxis([datetime2str(x) for x in new_index.index])
            .add_yaxis('Original', [round(x, 2) for x in original.values.flatten()],
                       linestyle_opts=opts.LineStyleOpts(color='blue'), is_symbol_show=False)
            .add_yaxis('Train Prediction', [round(x, 2) for x in train_predict.to_list()],
                       linestyle_opts=opts.LineStyleOpts(color='red'), is_symbol_show=False)
            .add_yaxis('Test Prediction', [round(x, 2) for x in test_predict.tolist()],
                       linestyle_opts=opts.LineStyleOpts(color='orange'), is_symbol_show=False)
            .add_yaxis('Future', [round(x, 2) for x in future_.to_list()],
                       linestyle_opts=opts.LineStyleOpts(color='green'), is_symbol_show=False)
            .set_global_opts(
            title_opts=opts.TitleOpts(
                title='模型:{}  城市:{}  作物:{}  Train RMSE:{:.2f}  Test RMSE:{:.2f}'.format(
                    model,
                    params['city'],
                    params['crop'],
                    train_RMSE,
                    test_RMSE
                ),
                subtitle="折线图"),
            tooltip_opts=opts.TooltipOpts(trigger='axis', axis_pointer_type='cross'),
            toolbox_opts=opts.ToolboxOpts(is_show=True, orient='vertical', pos_left='right'),
            legend_opts=opts.LegendOpts(pos_top=30)
        )
            .set_colors(['blue', 'red', 'orange', 'green'])
    )
    line.render(save_path)


def render_static(statistic, save_path):
    bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width='740px', height='410px'))
            .add_xaxis(['ARIMA', 'LR', 'RFR', 'LSTM'])
            .add_yaxis("记录条数", statistic['count'])
            .add_yaxis("复杂度", statistic['complex'])
            .set_global_opts(
            title_opts=opts.TitleOpts(title="统计", subtitle="柱状图"),
            tooltip_opts=opts.TooltipOpts(trigger='axis', axis_pointer_type='cross'),
            toolbox_opts=opts.ToolboxOpts(is_show=True, orient='vertical', pos_left='right'),
            legend_opts=opts.LegendOpts(pos_top=30)
        )
    )
    bar.render(save_path)
