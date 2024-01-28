import pandas as pd
from nilmtk.dataset import DataSet
import nilmtk.utils as utils
import numpy as np
import plotly
import plotly.graph_objs as go

pd.set_option('display.max_columns', None)

all_type_power = ['active', 'apparent']
SAMPLE_RATE = 180

NAME_MAP = {
    'fridge': 'Refrigerator',
    'dish washer': 'Dish washer',
    'washing machine': 'Washing machine',
    'kettle': 'Kettle',
    'microwave': 'Microwave oven'
}


class Converter(object):
    def __init__(self, params: dict):
        self.appliances = []
        for elems in params['appliances']:
            self.appliances.append(elems)
        self.datasets_dict = params['train']['datasets']
        self.app_meta = params['app_meta']

        self.train_mains = pd.DataFrame()
        self.train_submeters = []
        self.power = params['power']
        self.sample_period = params['sample_rate']

    def convert(self):
        d = self.datasets_dict
        self.train_mains = pd.DataFrame()
        self.train_submeters = [pd.DataFrame() for i in range(len(self.appliances))]
        for dataset in d:
            print("Loading data for ", dataset, " dataset")
            train = DataSet(d[dataset]['path'])
            for building in d[dataset]['buildings']:
                print("Loading building ... ", building)
                train.set_window(start=d[dataset]['buildings'][building]['start_time'],
                                 end=d[dataset]['buildings'][building]['end_time'])
                train_df = next(
                    train.buildings[building].elec.mains().load(physical_quantity='power',
                                                                ac_type=all_type_power,
                                                                sample_period=self.sample_period,
                                                                resample=True,
                                                                resample_kwargs={
                                                                    'fill_method': None,
                                                                    'how': 'max'
                                                                }))
                train_df.ffill(axis=0, inplace=True, limit=180 // self.sample_period)
                if 'active' in train_df['power']:
                    train_df['active'] = train_df['power']['active']
                if 'apparent' in train_df['power']:
                    train_df['apparent'] = train_df['power']['apparent']
                # train_df['reactive'] = np.sqrt(
                #     train_df['power']['apparent'] ** 2 - train_df['power']['active'] ** 2)
                train_df.drop(['power'], axis=1, inplace=True)
                train_df = train_df[self.power['mains']]

                appliance_readings = []
                for appliance_name in self.appliances:
                    appliance_df = next(train.buildings[building].elec[appliance_name].load(
                        physical_quantity='power', ac_type=self.power['appliance'],
                        sample_period=self.sample_period,
                        resample=True,
                        resample_kwargs={
                            'fill_method': None,
                            'how': 'max'
                        }
                    ))
                    appliance_df.ffill(axis=0, inplace=True, limit=180 // self.sample_period)
                    # appliance_df = appliance_df[[list(appliance_df.columns)[0]]]
                    # appliance_df.clip(upper=self.app_meta[appliance_name]["max"])
                    appliance_readings.append(appliance_df)
                # if self.DROP_ALL_NANS:
                #     train_df, appliance_readings = self.dropna(train_df, appliance_readings)

                print("Train Jointly")
                self.train_mains = self.train_mains.append(train_df)
                for i, appliance_name in enumerate(self.appliances):
                    self.train_submeters[i] = self.train_submeters[i].append(appliance_readings[i])

        merged_frame = self.train_mains.copy()
        stats_frame = pd.DataFrame(columns=["appliance", "TOTAL", "ON", "OFF"])
        for i, appliance_name in enumerate(self.appliances):
            on_thresh = self.app_meta[appliance_name]['on']
            app = self.train_submeters[i]
            merged_frame[appliance_name] = app
            merged_frame[appliance_name + "_state"] = \
                np.where(merged_frame[appliance_name] >= on_thresh, 1, 0)
            total = len(app)
            on_count = np.count_nonzero(merged_frame[appliance_name + "_state"])
            stats_frame.loc[len(stats_frame.index)] = [appliance_name, total, on_count,
                                                       total - on_count]

        # print(merged_frame.head())
        merged_frame.to_csv("./sample_output.csv")
        # stats_frame.to_csv("./sample_stats.csv", index=False)

        plot(merged_frame, self.appliances)


def plot(on_off_data: pd.DataFrame, appliances: list):
    plotly.offline.init_notebook_mode(connected=True)

    on_off_data['Date'] = on_off_data.index

    longest = max(map(len, appliances))

    data = []
    for ai, app_name in enumerate(appliances):
        time_list = []
        pos_list = []

        y_pos = (len(appliances) - ai) * 0.5 + 1

        grp = app_name + '_grp'
        on_off_data[grp] = (on_off_data[app_name + '_state'].diff(1) != 0).astype('int').cumsum()

        app_consec = pd.DataFrame({'BeginDate': on_off_data.groupby(grp).Date.first(),
                                   'EndDate': on_off_data.groupby(grp).Date.last() + pd.Timedelta(seconds=SAMPLE_RATE),
                                   'Value': on_off_data.groupby(grp)[app_name + '_state'].first(),
                                   'Consecutive': on_off_data.groupby(grp).size()})

        # print(app_name)
        print(app_consec.head())
        # print(on_off_data.head())

        for index, row in app_consec.iterrows():
            on = row['Value'] == 1
            if on:
                timespan = row['EndDate'] - row['BeginDate']
                min_span = pd.Timedelta(minutes=12)
                if timespan < min_span:
                    end_date = row['BeginDate'] + min_span
                else:
                    end_date = row['EndDate']
                time_list.extend([row['BeginDate'], end_date, None])
                pos_list.extend([y_pos, y_pos, None])

        on_count = on_off_data[app_name + '_state'].value_counts()[1]
        percent = on_count / len(on_off_data) * 100

        trace = go.Scatter(
            x=time_list,
            y=pos_list,
            # name=f'{app_name}: {"  " * (longest - len(app_name))}{percent:.2f}%',
            name=f'{percent:.2f}%{"  " if percent < 10 else ""} {NAME_MAP[app_name]}',
            mode='lines',
            line=dict(
                width=15,
            ),
            connectgaps=False
        )
        data.append(trace)

    layout = go.Layout(
        title="Appliance Status",
        yaxis=dict(
            visible=False
        ),
        width=1200,
        height=400,
    )
    # layout.coloraxis.colorbar.title.side = "top"
    # fig = dict(data=data, layout=layout)
    # # Plot and embed in ipython notebook!
    # # plotly.offline.iplot(fig, filename='basic-scatter')
    # plotly.offline.plot(fig)
    fig = go.Figure(data=data, layout=layout)
    fig.write_image("sample_fig.png", scale=5)
    fig.show()


if __name__ == '__main__':
    par = {
        'power': {
            'mains': ['active'],
            # 'appliance': ['active']
            # 'mains': ['apparent'],  # problem: ukdale active, redd apparent
            'appliance': ['active']
        },
        'sample_rate': SAMPLE_RATE,
        # 'appliances': ['fridge'],
        'app_meta': utils.APP_META["ukdale"],
        'appliances': ['fridge', 'microwave', 'washing machine', 'dish washer', 'kettle'],
        'train': {
            'datasets': {
                # 'redd': REDD_TRAIN_STD
                'ukdale': {
                    'path': 'mnt/ukdale.h5',
                    'buildings': {
                        1: {
                            'start_time': '2013-09-01',
                            'end_time': '2013-09-08'
                        },
                    },
                }
            }
        },
    }
    cvt = Converter(par)
    cvt.convert()
