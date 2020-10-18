# ----------------引用套件--------------
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os
from pandas.tseries.offsets import MonthEnd
import json
from value_transform import value_transfrom
import matplotlib.pyplot as plt
# import seaborn as sns
import dash_bootstrap_components as dbc
from datetime import datetime as dt
import plotly.figure_factory as ff
import base64
import datetime
import io
# import dash_table
# import plotly.express as px
import statsmodels.api as sm
from glob import glob
files = glob('temp3/*.csv')
# ----------------載入資料--------------
# origin_data = pd.read_csv("temp3/yearly_trading_data_transformed_Momentum_by_sectors.csv") #財報指標和月頻交易日報酬，由"profit"和股價資料merge產生
origin_data = pd.DataFrame()
for i in files:
    origin_data = origin_data.append(pd.read_csv(i))
    
def add_bull_bear_signal_score(data):
    sig = pd.read_csv('market_signal.csv')
    select_bull = ['marketing_bull_lin','marketing_bull_svr']
    select_bear = ['marketing_bear_lin1','marketing_bear_lin2','marketing_bear_svr1','marketing_bear_svr2']
    sig = sig.query('name in @select_bull or name in @select_bear')
    sig['score'] = sig.apply(lambda x:x.perd*(-0.5) if x['name'] in select_bear else x.perd,1)
    sig = sig.groupby('date')['score'].sum().reset_index()
    sig['date'] = pd.to_datetime(sig.date, format='%Y-%m-%d').apply(lambda x:x.strftime('%Y/%m'))
    data = data.merge(sig, left_on='yyyymm', right_on='date', how='left')
    data['score'] = data['score'].fillna(0)
    return data
origin_data = add_bull_bear_signal_score(origin_data)

benchmark = pd.read_csv('temp2/index_monthly_cumulated_return.csv', index_col="Date") #各項比較指數
benchmark.index = pd.to_datetime(benchmark.index)
Analysis_Factors_Table = pd.read_csv('Raw_data/FF_Factors.csv')
Analysis_Factors_Table.Date = pd.to_datetime(Analysis_Factors_Table.Date)
Analysis_Factors_Table = Analysis_Factors_Table.query('證券名稱 == "{}" '.format("Y8888")).set_index('Date')
index_name=[
    '證券名稱', '簡稱', '年月', '市場風險溢酬', '規模溢酬 (3因子)', '規模溢酬 (5因子)', '淨值市價比溢酬',
    '益本比溢酬', '動能因子', '短期反轉因子', '長期反轉因子', '盈利能力因子', '投資因子',
    '無風險利率', '市場投組'
    ]
Analysis_Factors_Table = Analysis_Factors_Table[index_name]


#把yyyymm改成int
origin_data.yyyymm = origin_data.yyyymm.apply(lambda x:int(x.replace("/","")))
data = origin_data.copy()

with open('PortfolioApp_data/app_lists.json','r')as f: 
    app_lists = json.load(f)

with open('temp2/benchmark_index_name.json','r')as f: 
    benchmark_name_dict = json.load(f)

# ----------------APP初始化--------------
# app = dash.Dash("__name__")
# app.config['suppress_callback_exceptions']=True
# app.css.append_css({"external_url":"https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css"})
##-----新版CSS-------
app = dash.Dash(external_stylesheets=[
    "https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap-grid.min.css"],
    suppress_callback_exceptions=True
)
app.title = 'EMAQ Strategy'
server = app.server

#---------Layout 排版 ---------------
app.layout = html.Div([
    
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Row(dbc.Col(html.Div(html.H3(["基本條件設定"])), style={'textAlign': 'center'})),
                    dbc.Row(
                        [dbc.Col(html.Div(html.P(["回測期間", html.Br(),
                            dcc.DatePickerRange(
                                id='date-picker-range',
                                month_format='MMMM Y',
                                display_format='YYYY/M/D',
                                start_date=dt(2004, 1, 1),
                                end_date=dt(2020, 6, 30),
                                initial_visible_month=dt(2002, 1, 1),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px'
                                },
                            )
                        ]
                            ))),
                        dbc.Col(html.Div(html.P(["特殊篩選", html.Br(),
                            dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    '上傳自訂資產池列表',
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px'
                                },
                                multiple=False
                            ),
                            html.Div(id='output-data-upload'),
                        ]
                            )))
                            
                            ]
                    ),
                    dbc.Row(
                        dbc.Col(html.Div(html.P(["產業類別", html.Br(),
                        dcc.Dropdown(
                            id='industry',
                            options=[{'label': 'All Sectors', 'value': 'All Sectors'}]+[
                                {'label': i, 'value': i} for i in data.tse_industry_name.drop_duplicates()
                            ],
                            value='All Sectors',
                            multi=True
                        )
                        ]
                            )))
                    ),
                    dbc.Row(
                        dbc.Col(html.Div(html.P(["上市櫃類別", html.Br(),
                        dcc.Dropdown(
                            id='exchange',
                            options=[
                                {'label': '上市(TSE)', 'value': 'TSE'},
                                {'label': '上櫃(OTC)', 'value': 'OTC'},
                            ],
                            value=['TSE', 'OTC'],
                            multi=True
                        ),
                        ]
                            )))
                    ),
                    dbc.Row(
                        dbc.Col(html.Div(html.P(["比較基準", html.Br(),
                        dcc.Dropdown(
                            id='BENCHMARKS',
                            options=[
                                {'label': "{} ({})".format(benchmark_name_dict[i], i), 'value': i} for i in benchmark.columns
                            ],
                            value=['0050', "Y9999"],
                            multi=True
                        ),
                        ]
                            )))
                    ),
                    dbc.Row([
                        dbc.Col(html.Div(html.P(["流動性(市值門檻)", html.Br(), html.Div(id='slider-output-container')]))),
                        dbc.Col(
                        dcc.Slider(
                            id='liquidity-slider',
                            min=0,
                            max=1,
                            step=0.01,
                            value=0.50,
                            marks={
                                0: '0 %',
                                1: '100 %'
                            },
                        ))
                        ]
                    ),
                    dbc.Row(dbc.Col(html.Div(html.H3(["財報品質決策邏輯"])), style={'textAlign': 'center'})),
                    dbc.Row(dbc.Col([
                        html.Div(html.P([
                        dbc.FormGroup(dcc.RadioItems(
                            id='AQ_EM_Select_logit',
                            options=[
                                {'label': '先 AQ 再 EM  \t ', 'value': 1},
                                {'label': '先 EM 再 AQ  \t ', 'value': 2},
                                {'label': '取兩者交集 \t ', 'value': 3},
                                {'label': '取兩者排序相加 \t ', 'value': 4},
                            ],
                            value=1,
                            labelStyle={'display': 'inline-block'}
                        )),
                        ]
                            )),

                    ], style={'textAlign': 'center'}
                    )),
                    dbc.Row([
                        dbc.Col(html.Div(html.P(["AQ量值"]))),
                        dbc.Col(html.Div(
                        dcc.Dropdown(
                            id='AQ_measure',
                            options=[
                                {'label': i, 'value': i} for i in app_lists['options_persistence']
                            ],
                            value='opacity_operating_slope'
                        )
                        
                            )),
                    ]),

                    dbc.Row([
                        dbc.Col(html.Div(html.P([
                        dcc.Checklist(
                            id='positive_AQ',
                            options=[
                                {'label': '挑前面的 / 挑後面的', 'value': 1},
                            ],
                            value=[1]
                        ),
                        ]
                            ))),
                        dbc.Col(
                            dcc.Slider(
                                id='threshhold-slider-AQ',
                                min=0,
                                max=1,
                                step=0.01,
                                value=0.5,
                                marks={
                                0: '0 %',
                                0.2: '20 %',
                                0.4: '40 %',
                                0.6: '60 %',
                                0.8: '80 %',
                                1: '100 %'
                            },
                            )
                        )
                    ]),
                    dbc.Row([
                        dbc.Col(html.Div(html.P(["EM量值"]))),
                        dbc.Col(html.Div(
                        dcc.Dropdown(
                            id='EM_measure',
                            options=[
                                {'label': i, 'value': i} for i in app_lists['options_EM']
                            ],
                            value='Jones_model_measure'
                        )
                        
                            )),
                    ]),
                    dbc.Row([
                        dbc.Col(html.Div(html.P([
                        dcc.Checklist(
                            id='positive_EM',
                            options=[
                                {'label': '挑前面的 / 挑後面的', 'value': 1},
                            ],
                            value=[1]
                        ),
                        ]
                            ))),
                        dbc.Col(
                            dcc.Slider(
                                id='threshhold-slider-EM',
                                min=0,
                                max=1,
                                step=0.01,
                                value=0.5,
                                marks={
                                0: '0 %',
                                0.2: '20 %',
                                0.4: '40 %',
                                0.6: '60 %',
                                0.8: '80 %',
                                1: '100 %'
                            },
                            )
                        )
                    ]),
                    dbc.Row(dbc.Col(html.Div(html.H3(["動能策略決策邏輯"])), style={'textAlign': 'center'})),
                    dbc.Row([
                        dbc.Col(html.Div(html.P(["營收動能多空策略", html.Br(), html.Div(id='threshhold-output-container')]))),
                    ]),
                    dbc.Row([
                        dbc.Col(html.Div(html.P([
                        dcc.Checklist(
                            id='positive_Momentum',
                            options=[
                                {'label': '單月YoY / 累積YoY', 'value': 1},
                            ],
                            value=[1]
                        ),
                        ]
                            ))),
                        dbc.Col(
                            dcc.Slider(
                                id='threshhold_slider_Momentum',
                                min=0,
                                max=0.5,
                                step=0.01,
                                value=0.25,
                                marks={
                                0: '0 %',
                                0.1: '10 %',
                                0.2: '20 %',
                                0.3: '30 %',
                                0.4: '40 %',
                                0.5: '50 %'
                            },
                            )
                        )
                    ]),
                    dbc.Row(
                        [dbc.Col(html.Div(html.P([
                            dcc.Checklist(
                            id='market_signal',
                            options=[
                                {'label': '是否使用多空訊號', 'value': 1},
                            ],
                            value=[1]
                        ),
                        ]
                            )))
                        ]
                    ),
                ],
                width="4",
                style={
                    'border': '1px solid',
                    "height": "100%",
                    'overflow': 'scroll',
                    'padding': '10px 10px 10px 20px'
                    }
            ),
            dbc.Col(
                [
                    dbc.Row(
                        dbc.Col(html.Div([
                                dcc.Tabs(id="tabs", value='tab-1', children=[
                                dcc.Tab(label='投資績效圖', value='tab-1'),
                                dcc.Tab(label='投資績效表', value='tab-2'),
                                dcc.Tab(label='投資標的表', value='tab-3'),
                                dcc.Tab(label='投資組合分析', value='tab-4')
                            ]),
                            # html.Div(id='tabs-content')
                        ]))
                    ),
                    dbc.Row(
                    #-----圖-------------
                        dbc.Col(html.Div([
                            # dcc.Graph(
                            #     id='portfolio-graph', style={'width':'100%', 'height': '65vh','display': 'inline-block' }
                            # ),
                            dcc.Loading(
                                id = "loading-icon", 
                                children=dcc.Graph(
                                    id='portfolio-graph', style={'width':'100%', 'height': '85vh','display': 'inline-block' }
                                    ),
                                type="default"
                            ),
                            
                        ])
                    )
                ),
                html.Div(id='intermediate-value_1', style={'display': 'none'}),
                html.Div(id='intermediate-value_2', style={'display': 'none'}),
                html.Div(id='intermediate-value_3', style={'display': 'none'}),
                html.Div(id='intermediate-value_4', style={'display': 'none'})
                ],
                width="8",
                style={
                    'border': '1px solid',
                    "height": "100%"
                    }
            ),
        ], 
        style={"height": "100vh"}
        )
    ],
    # className="container",
    # style={"height":"900","width":"1000"},
    id="demo_1"
)

#-----相關函式-------------------------------
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df

#----- 動態計算 ---------------------------
#--流動性門檻即時顯示--
@app.callback(
    dash.dependencies.Output('slider-output-container', 'children'),
    [dash.dependencies.Input('liquidity-slider', 'value')])
def update_output(value):
    return '使用前 {:.2%} 資料'.format(value)
#--投組門檻即時顯示--
@app.callback(
    dash.dependencies.Output('threshhold-output-container', 'children'),
    [dash.dependencies.Input('threshhold_slider_Momentum', 'value')])
def update_output(value):
    return '使用前後各 {:.2%} 資料形成投組'.format(value)

#---投組運算流程-------------------

#--資料子集選取-------
@app.callback(
    dash.dependencies.Output('output-data-upload', 'children'),
    [dash.dependencies.Input('upload-data', 'contents')],
    [dash.dependencies.State('upload-data', 'filename'),
    dash.dependencies.State('upload-data', 'last_modified')]
    )
def get_subset_by_outer_filter(list_of_contents, list_of_names, list_of_dates):
    #fetch global inputs
    global data, origin_data
    data = origin_data.copy()

    if list_of_contents is not None:
        df_outer_select = parse_contents(list_of_contents, list_of_names, list_of_dates)
        df_outer_select.yyyymm = pd.to_datetime(df_outer_select.yyyymm)
        origin_date = np.unique(df_outer_select.yyyymm)
        origin_date= pd.to_datetime(origin_date)
    
        expand_data_table = pd.DataFrame()

        for i in range(len(origin_date)-1):
            period = pd.date_range(start = origin_date[i], end = origin_date[i+1], freq='M')
            for p in period:
                    temp = df_outer_select[df_outer_select.yyyymm == origin_date[i]]
                    temp.yyyymm = p
                    expand_data_table = pd.concat([expand_data_table, temp])
        expand_data_table.yyyymm=expand_data_table.yyyymm.apply(lambda x: x.year*100+x.month)

        data = expand_data_table.merge(data, left_on=['company', 'yyyymm'], right_on=['company', 'yyyymm'], how ='left')
        data.yyyymm = data.yyyymm.astype(int)
    
    return list_of_names

@app.callback(
    dash.dependencies.Output('intermediate-value_1', 'children'),
    [dash.dependencies.Input('date-picker-range', 'start_date'),
    dash.dependencies.Input('date-picker-range', 'end_date'),
    dash.dependencies.Input('industry', 'value'),
    dash.dependencies.Input('exchange', 'value'),
    dash.dependencies.Input('AQ_measure', 'value'),
    dash.dependencies.Input('EM_measure', 'value'),
    ],)
def get_subset_dataframe(start_date, end_date, Industry, Exchange, AQ_measure, EM_measure):
    #fetch global inputs
    selected_month = pd.to_datetime(start_date[:10]).year*100 + pd.to_datetime(start_date[:10]).month
    selected_month_end = pd.to_datetime(end_date[:10]).year*100 + pd.to_datetime(end_date[:10]).month
    industry = Industry
    exchange = Exchange
    measure_AQ = AQ_measure
    measure_EM = EM_measure
    global data

    # 時間篩選
    sub_data = data.query('yyyymm >= {}'.format(selected_month)).query('yyyymm <= {}'.format(selected_month_end))

    #產業篩選
    if "All Sectors" in industry:
        pass
    else:
        sub_data = sub_data[sub_data.tse_industry_name.apply(lambda x:x in industry)]
    sub_data = sub_data[sub_data.exchange.apply(lambda x:x in exchange)]

    #上市櫃種類篩選
    #修改exchange功能要牽涉到資料表
    # sub_data = sub_data[sub_data.exchange.apply(lambda x:x in exchange)]
    
    #整理資料子集
    selected_columns = [
    "company",
    "company_abbreviation",
    "exchange",
    "tse_industry_name",
    "yyyymm",
    "close_price_month",
    "log_return_month",
    "return_month",
    "market_capitalization",
    '單月營收成長率％',
    "累計營收成長率％",
    "score"
    ]
    
    selected_columns.append(measure_AQ)
    selected_columns.append(measure_EM)
    subset_data = sub_data[selected_columns]
    return subset_data.to_json()


#--計算策略報酬

@app.callback([
    dash.dependencies.Output('intermediate-value_2', 'children'),
    dash.dependencies.Output('intermediate-value_4', 'children')],
    [dash.dependencies.Input('intermediate-value_1', 'children'),
    dash.dependencies.Input('liquidity-slider', 'value'),
    dash.dependencies.Input('AQ_measure', 'value'),
    dash.dependencies.Input('positive_AQ', 'value'),
    dash.dependencies.Input('threshhold-slider-AQ', 'value'),
    dash.dependencies.Input('EM_measure', 'value'),
    dash.dependencies.Input('positive_EM', 'value'),
    dash.dependencies.Input('threshhold-slider-EM', 'value'),
    dash.dependencies.Input('positive_Momentum', 'value'),
    dash.dependencies.Input('threshhold_slider_Momentum', 'value'),
    dash.dependencies.Input('AQ_EM_Select_logit', 'value'),
    dash.dependencies.Input('market_signal', 'value')
    ])
def get_strategy_return(jsonified_cleaned_data, liquidity_threshold, AQ_Measure, AQ_positive, AQ_threshold, EM_Measure, EM_positive, EM_threshold, positive_Momentum, threshhold_slider_Momentum, AQ_EM_Select_Method, signal_method):
    subset_data = pd.read_json(jsonified_cleaned_data)
    #liquidity_threshold, AQ_Measure, AQ_positive, AQ_threshold, EM_Measure, EM_positive, EM_threshold
    #-----流動性篩選-----
    subset_data['market_cap'] = subset_data.groupby(['yyyymm'])['market_capitalization'].rank(method = \
                                                                        "dense", ascending=False, pct=True)
    subset_data['last_market_cap'] = subset_data.groupby(['company']).shift(1)['market_cap']
    subset_data = subset_data.query('last_market_cap <={}'.format(liquidity_threshold))

    # AQ和EM有不同的選取標準
    #先AQ再EM
    if AQ_EM_Select_Method == 1:
        #-----AQ篩選------
        subset_data['AQ_rank_in_pct'] = subset_data.groupby(['yyyymm'])[AQ_Measure].rank(method = \
                                                                            "dense", ascending= (False if 1 in AQ_positive else True), pct=True)
        subset_data['AQ_hold_pct'] = subset_data.groupby(['company']).shift(1)['AQ_rank_in_pct']
        subset_data = subset_data.query('AQ_hold_pct <={}'.format(AQ_threshold))
        
        #-----EM篩選------
        subset_data['EM_rank_in_pct'] = subset_data.groupby(['yyyymm'])[EM_Measure].rank(method = \
                                                                            "dense", ascending= (False if 1 in EM_positive else True) , pct=True)
        subset_data['EM_hold_pct'] = subset_data.groupby(['company']).shift(1)['EM_rank_in_pct']
        subset_data = subset_data.query('EM_hold_pct <={}'.format(EM_threshold))
    
    #先EM再AQ
    elif AQ_EM_Select_Method == 2:      
        #-----EM篩選------
        subset_data['EM_rank_in_pct'] = subset_data.groupby(['yyyymm'])[EM_Measure].rank(method = \
                                                                            "dense", ascending= (False if 1 in EM_positive else True) , pct=True)
        subset_data['EM_hold_pct'] = subset_data.groupby(['company']).shift(1)['EM_rank_in_pct']
        subset_data = subset_data.query('EM_hold_pct <={}'.format(EM_threshold))
        
        #-----AQ篩選------
        subset_data['AQ_rank_in_pct'] = subset_data.groupby(['yyyymm'])[AQ_Measure].rank(method = \
                                                                            "dense", ascending= (False if 1 in AQ_positive else True), pct=True)
        subset_data['AQ_hold_pct'] = subset_data.groupby(['company']).shift(1)['AQ_rank_in_pct']
        subset_data = subset_data.query('AQ_hold_pct <={}'.format(AQ_threshold))
    
    #同時考慮-->採取交集
    elif  AQ_EM_Select_Method == 3:   
        subset_data['EM_rank_in_pct'] = subset_data.groupby(['yyyymm'])[EM_Measure].rank(method = \
                                                                            "dense", ascending= (False if 1 in EM_positive else True) , pct=True)
        subset_data['EM_hold_pct'] = subset_data.groupby(['company']).shift(1)['EM_rank_in_pct']
        subset_data['AQ_rank_in_pct'] = subset_data.groupby(['yyyymm'])[AQ_Measure].rank(method = \
                                                                            "dense", ascending= (False if 1 in AQ_positive else True), pct=True)
        subset_data['AQ_hold_pct'] = subset_data.groupby(['company']).shift(1)['AQ_rank_in_pct']
        subset_data = subset_data.query('AQ_hold_pct <={} & EM_hold_pct <={}'.format(AQ_threshold, EM_threshold))

    #同時考慮-->採取相加
    elif  AQ_EM_Select_Method == 4:   
        subset_data['EM_rank_in_pct'] = subset_data.groupby(['yyyymm'])[EM_Measure].rank(method = \
                                                                            "dense", ascending= (False if 1 in EM_positive else True) , pct=True)
        subset_data['EM_hold_pct'] = subset_data.groupby(['company']).shift(1)['EM_rank_in_pct']
        subset_data['AQ_rank_in_pct'] = subset_data.groupby(['yyyymm'])[AQ_Measure].rank(method = \
                                                                            "dense", ascending= (False if 1 in AQ_positive else True), pct=True)
        subset_data['AQ_hold_pct'] = subset_data.groupby(['company']).shift(1)['AQ_rank_in_pct']
        subset_data = subset_data.eval('EMAQ_Mean = EM_hold_pct + AQ_hold_pct ')

        subset_data['Total_Pct'] = subset_data.groupby(['yyyymm'])['EMAQ_Mean'].rank(method = \
                                                                            "dense", ascending= True, pct=True)

        subset_data = subset_data.query('Total_Pct <={} '.format( min(AQ_threshold, EM_threshold)))


    #-----動能多空篩選------
    #單月營收成長率％,累計營收成長率％
    if bool(len(positive_Momentum)) == True:
        Momentum_Measure = "單月營收成長率％"
    else:
        Momentum_Measure = "累計營收成長率％"


    subset_data['rank_in_pct'] = subset_data.groupby(['yyyymm'])[Momentum_Measure].rank(method = \
                                                                        "dense", ascending=True, pct=True)
    subset_data['hold_pct'] = subset_data.groupby(['company']).shift(1)['rank_in_pct']

    #-----依據多空指標加減碼------

    if 1 in signal_method:
        multi = (1+0.2*subset_data['score'])
        subset_data['log_return_month'] = 100 * np.log(1 + multi * subset_data['return_month']/100)


    LONG=subset_data[subset_data["hold_pct"]>(1-threshhold_slider_Momentum)].groupby(['yyyymm']).mean()['log_return_month']/100
    LONG.index = pd.to_datetime( LONG.index, format='%Y%m')+ MonthEnd(1)
    
    SHORT=subset_data[subset_data["hold_pct"]<threshhold_slider_Momentum].groupby(['yyyymm']).mean()['log_return_month']/100
    SHORT.index = pd.to_datetime( SHORT.index, format='%Y%m')+ MonthEnd(1)
    
    HEDGE = LONG - SHORT

    LONG[LONG.index[0] - MonthEnd(1)]=0
    LONG.sort_index(inplace=True)
    LONG.rename('Long', inplace=True)
    
    SHORT[SHORT.index[0] - MonthEnd(1)]=0
    SHORT.sort_index(inplace=True)
    SHORT.rename('Short', inplace=True)
    
    HEDGE[HEDGE.index[0] - MonthEnd(1)]=0
    HEDGE.sort_index(inplace=True)
    HEDGE.rename('Hedge', inplace=True)
    
    strategy_return = pd.concat([np.exp(LONG.cumsum()), np.exp(SHORT.cumsum()), np.exp(HEDGE.cumsum())], axis=1)

    Long_list = subset_data[subset_data["hold_pct"] > (1 - threshhold_slider_Momentum)]
    Short_list = subset_data[subset_data["hold_pct"] < threshhold_slider_Momentum]
    Long_list["position"] = "Buy"
    Short_list["position"] = "Short"
    Stock_list = pd.concat([Long_list, Short_list])
    Stock_list.sort_values(by=["yyyymm", "position", "company"], ascending = [True, True, True], inplace=True)

    return strategy_return.to_json(), Stock_list.to_json()


#---取得畫圖資料
@app.callback(
    dash.dependencies.Output('intermediate-value_3', 'children'),
    [dash.dependencies.Input('intermediate-value_2', 'children'),
    dash.dependencies.Input('BENCHMARKS', 'value'),
    ])
def get_plot_source(jsonified_cleaned_data, Benchmarks):
    strategy_return = pd.read_json(jsonified_cleaned_data)
    benchmarks_ = Benchmarks

    long, short, hedge = (strategy_return["Long"], strategy_return["Short"], strategy_return["Hedge"])
    bm_index = benchmark[benchmarks_]
    plot_data = pd.concat([long, hedge, bm_index], axis=1).dropna()
    plot_data.sort_index(inplace=True)
    plot_data = plot_data / plot_data.iloc[0]
    plot_data = plot_data .reset_index().rename({'index':'Date'}, axis=1)

    return plot_data.to_json()

#---------投組分析公式---------------------------------
def Average_Return(Stat_data, portfolio):
    return Stat_data[portfolio].mean()

def STD(Stat_data, portfolio):
    return Stat_data[portfolio].std()

def Sharpe_Ratio(Stat_data, portfolio):
    return Stat_data.eval('{} - 無風險利率/12'.format(portfolio)).mean() *12 / (STD(Stat_data, portfolio)*np.sqrt(12))
    
def Sortino_Ratio(Stat_data, portfolio):
    downside_returns = Stat_data.query("{} < 0".format(portfolio))[portfolio]
    down_stdev = downside_returns.std()
    
    return Stat_data.eval('{} - 無風險利率/12'.format(portfolio)).mean() *12  /(down_stdev *np.sqrt(12))
    
def MDD(Stat_data, portfolio):
    ts = np.exp(Stat_data[portfolio].cumsum())
    peak = ts.cummax()
    
    return (ts / peak -1.0).min()

def FF_Factors(Stat_data, portfolio, Factors=1):
    y = Stat_data[portfolio]
    # 不同的FF因子模型
    if Factors == 1:
        x = Stat_data ['市場風險溢酬']
    elif Factors  == 3:
        x = Stat_data[[
            '市場風險溢酬',
            '規模溢酬 (3因子)',
            '淨值市價比溢酬'
        ]]
    elif Factors  == 4:
        x = Stat_data[[
            '市場風險溢酬',
            '規模溢酬 (3因子)',
            '淨值市價比溢酬',
            '動能因子'
        ]]
    elif Factors  == 5:
        x = Stat_data[[
            '市場風險溢酬',
            '規模溢酬 (5因子)',
            '淨值市價比溢酬',
            '盈利能力因子',
            '投資因子'
        ]]
    
    X = sm.add_constant(x)
    model = sm.OLS(y,X)
    results = model.fit()
    return (results.params)

#---------Tabs切換--------畫圖 & 畫表格---------------------------------
@app.callback(Output('portfolio-graph', 'figure'), #my-graph
                    [Input('tabs', 'value'),
                    Input('intermediate-value_3', 'children'),
                    Input('intermediate-value_4', 'children')
                    ])
def render_content(tab, jsonified_cleaned_data_3, jsonified_cleaned_data_4):
    plot_data = pd.read_json(jsonified_cleaned_data_3)
    plot_data.Date = pd.to_datetime(plot_data.Date)

    global Analysis_Factors_Table

    if tab == 'tab-1':
        # plot_data.to_csv('D:/ipython/hayate/家偉/Dashapp/test.csv')
        return {
        'data':[go.Scatter(x=plot_data['Date'], y=plot_data[col], mode='lines+markers', name= col) for col in plot_data.columns[1:] ],

        'layout': go.Layout (title = '<b>Portfolio vs Benchmark')
        }

    elif tab == 'tab-2':
        # return html.Div([
        #     # html.H3('Tab content 2')
        # ])

        return go.Figure(
            data = [
                go.Table(
                    header = dict(values = plot_data.columns,
                                    ),
                    cells = dict(values= [plot_data['Date'].apply(lambda x:x.date())] + 
                    [
                        [round(i, 3) for i in plot_data[col]] for col in plot_data.columns[1:]
                    ],
                    )
                )
            ]
            )
        # figure = ff.create_table(...) # see docs https://plot.ly/python/table/
        # app.layout = html.Div([
        #     dcc.Graph(id='my-table', figure=figure)
        # ])
    elif tab == 'tab-3':
            # return html.Div([
        #     # html.H3('Tab content 2')
        # ])
        rowEvenColor = 'lightgrey'
        rowOddColor = 'white'
        table_data = pd.read_json(jsonified_cleaned_data_4)
        data = table_data[[
            'yyyymm',
            'company',
            'company_abbreviation',
            'exchange',
            'tse_industry_name',
            'position'
            ]]
        col_name_dict={
            'yyyymm':'時間',
            'company':'公司代碼',
            'company_abbreviation':'公司名稱',
            'exchange':"上市櫃",
            'tse_industry_name':"產業分類",
            'position':"持有部位"
        }
        # colorscale = [[0, '#4d004c'],[.5, '#f2e5ff'],[1, '#ffffff']]
        # data = data.rename(col_name_dict,axis=1)
        return go.Figure(
            data = [
                go.Table(
                    header = dict(
                        values = [ "<b>{}</b>".format(col_name_dict[i]) for i in data.columns],
                        font = dict(size=16),
                        height=20
                    ),
                    cells = dict(
                        values = [ 
                            [i for i in data[col]] for col in data.columns
                            ],
                        font_size=14,
                    
                    )
                )
            ]
            )
        # return ff.create_table(data, colorscale=colorscale)
        


    elif tab == 'tab-4':
        Return_Table = plot_data.copy()
        Return_Table = Return_Table.set_index('Date')
        Return_Table.index = pd.to_datetime(Return_Table.index)

        long_ret = np.log(Return_Table).diff().iloc[1:,]['Long']
        hedge_ret = np.log(Return_Table).diff().iloc[1:,:]['Hedge']


        stat_data = pd.concat([long_ret, hedge_ret, Analysis_Factors_Table], axis=1).dropna()

        factor_table = pd.DataFrame()
        for port in ['Long', 'Hedge']:
            for factors in [1,3,5]:
                df = FF_Factors(stat_data, port, factors).rename("{}_{}".format(port, factors))
                factor_table = pd.concat([factor_table, df], axis=1)

        factor_table.loc['const'] = factor_table.loc['const'] * 12 
        factor_table = factor_table.rename({'const':"Alpha"})

        table_data = [
            ['投資指標', 'Long', '投資指標', 'Hedge'],
            ['平均報酬率', "{:.3%}".format(Average_Return(stat_data, 'Long')*12), '平均報酬率', "{:.3%}".format(Average_Return(stat_data, 'Hedge')*12)],
            ['平均波動率', "{:.3%}".format(STD(stat_data, 'Long')*np.sqrt(12)), '平均波動率', "{:.3%}".format(STD(stat_data, 'Hedge')*np.sqrt(12))],
            ['夏普比例', "{:.3}".format(Sharpe_Ratio(stat_data, 'Long')), '夏普比例', "{:.3}".format(Sharpe_Ratio(stat_data, 'Hedge'))],
            ['索汀諾比例', "{:.3}".format(Sortino_Ratio(stat_data, 'Long')), '索汀諾比例', "{:.3}".format(Sortino_Ratio(stat_data, 'Hedge'))],
            ['最大回撤', "{:.3%}".format(MDD(stat_data, 'Long')), '最大回撤', "{:.3%}".format(MDD(stat_data, 'Hedge'))],
        ]

        fig = ff.create_table(table_data, height_constant=120)
        for i in range(len(fig.layout.annotations)):
            fig.layout.annotations[i].font.size = 20

        trace1 = go.Bar(x=factor_table.index, y=factor_table.Long_1,
                            marker=dict(color='#0099ff'),
                            name='Long1',
                            xaxis='x2', yaxis='y2')
        trace2 = go.Bar(x=factor_table.index, y=factor_table.Hedge_1,
                            marker=dict(color='#404040'),
                            name='Hedge1',
                            xaxis='x2', yaxis='y2')

        trace3 = go.Bar(x=factor_table.index, y=factor_table.Long_3,
                            marker=dict(color='#0099ff'),
                            name='Long3',
                            xaxis='x3', yaxis='y3')
        trace4 = go.Bar(x=factor_table.index, y=factor_table.Hedge_3,
                            marker=dict(color='#404040'),
                            name='Hedge3',
                            xaxis='x3', yaxis='y3')

        trace5 = go.Bar(x=factor_table.index, y=factor_table.Long_5,
                            marker=dict(color='#0099ff'),
                            name='Long5',
                            xaxis='x4', yaxis='y4')
        trace6 = go.Bar(x=factor_table.index, y=factor_table.Hedge_5,
                            marker=dict(color='#404040'),
                            name='Hedge5',
                            xaxis='x4', yaxis='y4')

        fig.add_traces([trace1, trace2])
        fig.add_traces([trace3, trace4])
        fig.add_traces([trace5, trace6])

        # initialize xaxis2 and yaxis2
        fig['layout']['xaxis2'] = {}
        fig['layout']['yaxis2'] = {}
        fig['layout']['xaxis3'] = {}
        fig['layout']['yaxis3'] = {}
        fig['layout']['xaxis4'] = {}
        fig['layout']['yaxis4'] = {}


        # Edit layout for subplots
        fig.layout.xaxis.update({'domain': [0, .4]})
        fig.layout.xaxis2.update({'domain': [0.5, 1]})
        fig.layout.xaxis3.update({'domain': [0.5, 1.]})
        fig.layout.xaxis4.update({'domain': [0.5, 1.]})

        # The graph's yaxis MUST BE anchored to the graph's xaxis
        fig.layout.yaxis2.update({'anchor': 'x2'})
        fig.layout.yaxis2.update({'title': 'CAPM'})

        fig.layout.yaxis3.update({'anchor': 'x3'})
        fig.layout.yaxis3.update({'title': 'FF三因子'})

        fig.layout.yaxis4.update({'anchor': 'x4'})
        fig.layout.yaxis4.update({'title': 'FF五因子'})

        fig.layout.yaxis.update({'domain': [0, 1]})
        fig.layout.yaxis2.update({'domain': [0.7, 1]})
        fig.layout.yaxis3.update({'domain': [.35, 0.65]})
        fig.layout.yaxis4.update({'domain': [0, 0.3]})


        # Update the margins to add a title and see graph x-labels.
        fig.layout.margin.update({'t':40, 'b':20})
        # fig.layout.update({'title': '投資組合分析'})

        return go.Figure(data = fig)



#------運行APP -------
if __name__ == "__main__":
    server.run(debug=False)



