from pyecharts import options as opts
from pyecharts.charts import Line
from pyecharts.globals import ThemeType

def render_echarts(data, params, save_path):
    # 时间
    date_list = [x['date'] for x in data]
    # 价格
    price_list = [x['price'] for x in data]
    line = (
        Line(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width='792px', height='464px'))
            .add_xaxis(date_list)
            .add_yaxis('Original', price_list, linestyle_opts=opts.LineStyleOpts(color='blue'),
                       is_symbol_show=False)
            .set_global_opts(
            title_opts=opts.TitleOpts(
                title='城市:{}  作物:{}  采样频率:{}  数据量:{}'.format(params['city'],params['crop'], params['freq'], len(data)),
                subtitle="折线图"),
            tooltip_opts=opts.TooltipOpts(trigger='axis', axis_pointer_type='cross'),
            toolbox_opts=opts.ToolboxOpts(is_show=True, orient='vertical', pos_left='right'),
            legend_opts=opts.LegendOpts(pos_top=30)
        )
            .set_colors(['blue'])
    )
    # 保存至本地
    line.render(save_path)