# 常用的函数
import os.path
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
from sklearn.cluster import KMeans
from concurrent.futures import ProcessPoolExecutor
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import f1_score, classification_report
from config import *
import scipy.stats as stats


def IsInsideTriangle(a, b, c, p):
    # Is point [p] in Triangle [abc]?
    def TriangleArea(x, y, z):
        return abs((x[0] * (y[1] - z[1]) + y[0] * (z[1] - x[1]) + z[0] * (x[1] - y[1])) / 2.0)

    ABC = TriangleArea(a, b, c)

    PBC = TriangleArea(p, b, c)
    PAC = TriangleArea(a, p, c)
    PAB = TriangleArea(a, b, p)

    return abs(ABC - PBC - PAC - PAB) < 0.00000001


def point_distance_line(p, a, b):
    # Calculate the distance of point [p] to line [ab]
    vec1 = a - p
    vec2 = b - p
    distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(a - b)
    return distance


def point_triangle(p, a, b, c):
    # find the shortest distance of point in the triangle [abc] to point [p]
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if (d1 <= 0) & (d2 <= 0):
        return a

    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if (d3 >= 0) & (d4 <= d3):
        return b

    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if (d6 >= 0) & (d5 < d6):
        return c

    vc = d1 * d4 - d3 * d2
    if (vc <= 0) & (d1 >= 0) & (d3 <= 0):
        v1 = d1 / (d1 - d3)
        return a + v1 * ab

    vb = d5 * d2 - d1 * d6
    if (vb <= 0) & (d2 >= 0) & (d6 <= 0):
        w1 = d2 / (d2 - d6)
        return a + w1 * ac

    va = d3 * d6 - d5 * d4
    if (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0):
        w1 = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w1 * (c - b)

    denom = 1 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w


def load_regions():
    """
    :return: Return: the regions with 1 for valid and nan for invalid
    valid: GRADES and GLDAS data > valid years (20 years for default)
    """

    if os.path.exists(args.results_path + args.regions_path):
        regions = np.load(args.results_path + args.regions_path)
    else:
        runoff = np.load(args.data_path + args.runoff_idmax_path)
        pre = np.load(args.data_path + args.pre_idmax_path)
        regions = np.zeros((args.global_lat, args.global_lon))
        regions[regions == 0] = np.nan
        for i in range(args.global_lat):
            for j in range(args.global_lon):
                pre1, runoff1 = pre[i, j, :], runoff[i, j, :]
                if ((runoff1 > 0).sum() > args.valid_years) & ((pre1 > 0).sum() > args.valid_years):
                    regions[i, j] = 1
        np.save(args.results_path + args.regions_path, regions)

    return regions


def Relative_Importance(condition, pre_sin, soil_sin, snow_sin, runoff_sin,
                        pre_cos, soil_cos, snow_cos, runoff_cos):
    """
    :param condition: 1 for snow is considered while 2 for snow is ignored
    :param pre_sin: sin of pre
    :param soil_sin: sin of soil
    :param snow_sin: sin of snow
    :param runoff_sin: sin of runoff
    :param pre_cos: cosin of pre
    :param soil_cos: cosin of soil
    :param snow_cos: cosin of snow
    :param runoff_cos: cosin of runoff
    :return: The relative importance of pre, soil and snow
    """

    # all the alpha should be limited in [0,1]
    def cons1(x):
        return x[0]

    def cons2(x):
        return 1 - x[0]

    def cons3(x):
        return x[1]

    def cons4(x):
        return 1 - x[1]

    def cons5(x):
        return x[2]

    def cons6(x):
        return 1 - x[2]

    c1 = {'type': 'ineq', 'fun': cons1}
    c2 = {'type': 'ineq', 'fun': cons2}
    c3 = {'type': 'ineq', 'fun': cons3}
    c4 = {'type': 'ineq', 'fun': cons4}
    c5 = {'type': 'ineq', 'fun': cons5}
    c6 = {'type': 'ineq', 'fun': cons6}

    # alpha is calculated by three other constraints
    if condition == 1:
        def objective(x):
            return 1 - x[0] - x[1] - x[2]

        def constraint1(x):
            return x[0] * pre_sin + x[1] * soil_sin + x[2] * snow_sin - runoff_sin

        def constraint2(x):
            return x[0] * pre_cos + x[1] * soil_cos + x[2] * snow_cos - runoff_cos

        def constraint3(x):
            return 1 - x[0] - x[1] - x[2]

        con1 = {'type': 'eq', 'fun': constraint1}
        con2 = {'type': 'eq', 'fun': constraint2}
        con3 = {'type': 'ineq', 'fun': constraint3}

        cons = ([con1, con2, con3, c1, c2, c3, c4, c5, c6])
        x0 = np.array([0.33, 0.33, 0.33])  # initialization
        solution = minimize(objective, x0, method='SLSQP', constraints=cons)

        x = solution.x
        alpha_sn = x[2]

    else:
        def objective(x):
            return 1 - x[0] - x[1]

        def constraint1(x):
            return x[0] * pre_sin + x[1] * soil_sin - runoff_sin

        def constraint2(x):
            return x[0] * pre_cos + x[1] * soil_cos - runoff_cos

        def constraint3(x):
            return 1 - x[0] - x[1]

        con1 = {'type': 'eq', 'fun': constraint1}
        con2 = {'type': 'eq', 'fun': constraint2}
        con3 = {'type': 'ineq', 'fun': constraint3}

        # 3个约束条件
        cons = ([con1, con2, con3, c1, c2, c3, c4])

        # 5. 求解
        x0 = np.array([0.5, 0.5])  # 定义初始值
        solution = minimize(objective, x0, method='SLSQP', constraints=cons)

        x = solution.x
        alpha_sn = 0

    alpha_sm = x[1]
    alpha_pre = x[0]

    pre_scale = abs(alpha_pre) / (abs(alpha_pre) + abs(alpha_sm) + abs(alpha_sn))
    sm_scale = abs(alpha_sm) / (abs(alpha_pre) + abs(alpha_sm) + abs(alpha_sn))
    sn_scale = abs(alpha_sn) / (abs(alpha_pre) + abs(alpha_sm) + abs(alpha_sn))

    return pre_scale, sm_scale, sn_scale


def global_plt(data, seed='alpha'):
    lon = args.lon_array
    lat = args.lat_array
    if seed == 'alpha':
        cmap = ListedColormap(['#e0e8e5', '#bacacc', '#a0cfd6', '#53bab8', '#2e867d',
                               '#FEDB9B', '#FECA64', '#FCAC23', '#E97D01', '#B53302'])
        levels = MaxNLocator(nbins=11).tick_values(0, 1)
        bar_ticks = np.arange(0, 1.1, 0.1)
        bar_ticks_labels = ['', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '']
    if seed == 'class3':
        cmap = ListedColormap(['LimeGreen', '#f9de59', 'DarkBlue'])
        levels = MaxNLocator(nbins=3).tick_values(0, 3)
        bar_ticks = np.arange(0.5, 2.6, 1)
        bar_ticks_labels = ['P', 'M', 'N']
    if seed == 'class8':
        cmap = ListedColormap(['#546c2a', '#729037', '#c8d86d',
                               '#B53302', '#E97D01', '#FCAC23', '#FECA64',
                               'DarkBlue'])
        levels = MaxNLocator(nbins=8).tick_values(0, 8)
        bar_ticks = np.arange(0.5, 7.6, 1)
        bar_ticks_labels = ['P-MN', 'P-M', 'PN', 'M-PN', 'M-P', 'M-N', 'MP', 'N-PM']
    if seed == 'timing':
        cmap = ListedColormap(['DarkBlue', '#2e867d', '#53bab8', '#a0cfd6', '#bacacc', '#e0e8e5',
                               '#FEDB9B', '#FECA64', '#FCAC23', '#E97D01', '#B53302', 'DarkRed'])
        levels = MaxNLocator(nbins=12).tick_values(0, 12)
        bar_ticks = np.arange(0.5, 11.6, 1)
        bar_ticks_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    if seed == 'trend':
        cmap = ListedColormap(['DarkBlue', '#2e867d', '#53bab8', '#a0cfd6', '#bacacc', '#e0e8e5',
                               '#FEDB9B', '#FECA64', '#FCAC23', '#E97D01', '#B53302', 'DarkRed'])
        levels = MaxNLocator(nbins=13).tick_values(-12, 12)
        bar_ticks = np.arange(-12, 13, 2)
        bar_ticks_labels = ['', '-10', '-8', '-6', '-4', '-2', '0', '2', '4', '6', '8', '10', '']

    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    fontsize = args.fonts

    fig = plt.figure(figsize=(20, 15))  # 设置一个画板，将其返还给fig
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    im = ax.contourf(lon, lat, data, levels=levels, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)  # 添加陆地
    ax.add_feature(cfeature.COASTLINE, lw=1)  # 添加海岸线
    ax.add_feature(cfeature.RIVERS, lw=0.5)  # 添加河流
    ax.add_feature(cfeature.LAKES)  # 添加湖泊
    # ax.add_feature(cfeature.BORDERS, linestyle='-', lw=0.5)  # 不推荐，我国丢失了藏南、台湾等领土
    ax.add_feature(cfeature.OCEAN)  # 添加海洋

    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='k', alpha=0.5, linestyle='--')
    gl.top_labels = False  # 关闭顶端的经纬度标签
    gl.right_labels = False  # 关闭右侧的经纬度标签
    gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
    gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式

    # colorbar
    cax = plt.axes([0.12, 0.20, 0.78, 0.03])
    cb = plt.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_ticks(bar_ticks)
    cb.set_ticklabels(bar_ticks_labels)
    cb.ax.tick_params(labelsize=fontsize)
    font = {'family': 'monospace',
            'weight': 'normal',
            'size': fontsize}
    plt.rc('font', **font)  # pass in the font dict as kwargs

    plt.show()


def Initialized_csv_set():
    """
    :return: csv with columns = ['pre', 'sm', 'sn', 'lon', 'lat'] in valid regions
    """
    results = np.load(args.results_path + args.Alpha_path)
    regions = np.load(args.results_path + args.regions_path)

    pre = (results[:, :, 0] * regions).reshape(-1)
    sm = (results[:, :, 1] * regions).reshape(-1)
    sn = (results[:, :, 2] * regions).reshape(-1)
    lon = np.reshape([np.arange(0, args.global_lon)] * args.global_lat, -1)
    lat = np.array([0] * args.global_lon)
    for i in range(args.global_lat - 1):
        lat = np.append(lat, np.array([i + 1] * args.global_lon))

    csv_data = pd.DataFrame(columns=['pre', 'sm', 'sn', 'lon', 'lat'])
    csv_data['pre'], csv_data['sm'], csv_data['sn'] = pre, sm, sn
    csv_data['lon'], csv_data['lat'] = lon, lat

    csv_data = csv_data.dropna(axis=0)
    csv_data = csv_data.round(decimals=3)
    return csv_data


def Adding_TrainingFeatures(csv_data):
    """
    :param csv_data: initialized and labeled csv_data
    :return: Adding columns = [training features] for RF model
    """

    def add_data(csv_data, csv_set, index_i):
        id_lon, id_lat = csv_data.iloc[index_i, 3], csv_data.iloc[index_i, 4]
        a = [id_lon, id_lat]
        a.extend(pre[id_lat, id_lon, :])
        a.extend(soil[id_lat, id_lon, :])
        a.extend(snow[id_lat, id_lon, :])

        pd_a = pd.DataFrame(a)
        csv_set = pd.concat((csv_set, pd_a), axis=0)
        print('\r当前进度:  --- ' + str(index_i), end="")

    def multi_process():
        with ProcessPoolExecutor() as pool:
            pool.map(add_data, (csv_data, csv_set, range(len(csv_data))))

    def tack_data_692(i, j, b):
        id_lon, id_lat = csv_data.iloc[i + j * 5000, 3], csv_data.iloc[i + j * 5000, 4]
        a = [id_lon, id_lat]
        for factor in [pre, soil, snow]:
            temp = factor[id_lat, id_lon, :]
            temp = list(map(int, temp))
            temp_sort = np.sort(temp)
            a.extend(temp_sort)
            temp[temp == 0] = np.nan
            a.extend([np.nanmin(temp), np.nanquantile(temp, 0.25), np.nanmean(temp), np.nanmedian(temp),
                      np.nanquantile(temp, 0.75), np.nanmax(temp), np.nanstd(temp)])
        p = np.array(a[37:44])
        m = np.array(a[79:86])
        n = np.array(a[121:128])
        a.extend(p + m)
        a.extend(m + n)
        a.extend(p + n)
        a.extend(p / (m + 366))
        a.extend(m / (n + 366))
        a.extend(p / (n + 366))
        a.extend(p + m + n)
        b = np.vstack((b, np.reshape(a, [1, -1])))
        print('\r当前进度: ' + str(j) + ' --- ' + str(i), end="")
        return b

    def tack_data(i, j, b):
        id_lon, id_lat = csv_data.iloc[i + j * 5000, 3], csv_data.iloc[i + j * 5000, 4]
        a = [id_lon, id_lat]
        for factor in [pre, soil, snow]:
            temp = factor[id_lat, id_lon, :]
            temp = list(map(int, temp))
            temp[temp == 0] = np.nan
            temp = [x for x in temp if np.isnan(x) == False]
            a.extend([np.min(temp), np.quantile(temp, 0.2), np.quantile(temp, 0.3), np.quantile(temp, 0.4),
                      np.mean(temp), np.median(temp), np.quantile(temp, 0.6), np.quantile(temp, 0.7),
                      np.quantile(temp, 0.8), np.max(temp), np.std(temp), np.ptp(temp),
                      np.std(temp) / (np.mean(temp) + 1),
                      (np.quantile(temp, 0.2) - np.mean(temp)) / (np.std(temp) + 1),
                      (np.quantile(temp, 0.3) - np.mean(temp)) / (np.std(temp) + 1),
                      (np.quantile(temp, 0.4) - np.mean(temp)) / (np.std(temp) + 1),
                      (np.quantile(temp, 0.5) - np.mean(temp)) / (np.std(temp) + 1),
                      (np.quantile(temp, 0.6) - np.mean(temp)) / (np.std(temp) + 1),
                      (np.quantile(temp, 0.7) - np.mean(temp)) / (np.std(temp) + 1),
                      (np.quantile(temp, 0.8) - np.mean(temp)) / (np.std(temp) + 1)])
        p = np.array(a[2:22])
        m = np.array(a[22:42])
        n = np.array(a[42:62])
        a.extend(p + m)
        a.extend(m + n)
        a.extend(p + n)
        a.extend(p - m)
        a.extend(m - n)
        a.extend(p - n)
        a.extend(p / (m + 366))
        a.extend(m / (n + 366))
        a.extend(p / (n + 366))
        a.extend(p + m + n)
        b = np.vstack((b, np.reshape(a, [1, -1])))
        print('\r当前进度: ' + str(j) + ' --- ' + str(i), end="")
        return b

    snow = np.load(args.data_path + args.snow_idmax_path)
    pre = np.load(args.data_path + args.pre_idmax_path)
    soil = np.load(args.data_path + args.soil_idmax_path)
    csv_data = csv_data.reset_index(drop=True)

    csv_set = pd.DataFrame()
    num_data = len(csv_data)
    num_1, num_2 = num_data // 5000, num_data % 5000  # This set to quickly concat arrays

    # feature engineer
    for j in range(num_1 + 1):
        # Here, we set 198 features for each grid
        b = np.zeros((1, 177))

        if j == num_1:
            for i in range(num_2):
                b = tack_data_692(i, j, b)
        else:
            for i in range(5000):
                b = tack_data_692(i, j, b)

        pd_b = pd.DataFrame(b[1:, :])
        csv_set = pd.concat((csv_set, pd_b))

    csv_set = csv_set.reset_index(drop=True)
    csv_set = pd.concat((csv_data, csv_set), axis=1, ignore_index=True)
    return csv_set


def Adding_ClassLabels(csv):
    """
    :param csv: The initialized csv_set
    :return: Adding columns = ['class3', 'class8'] which is the labels for RF model
    """
    X_train = csv.copy()
    X_train = X_train[['pre', 'sm', 'sn']]
    model = KMeans(n_clusters=8, random_state=args.seed).fit(X_train)
    pred8 = model.predict(X_train)
    csv['class8'] = pred8

    csv['class3'] = pred8
    # the priority is sm > pre > sn
    for i in range(len(csv)):
        # 1 for sm
        if (csv.iloc[i, 1] >= csv.iloc[i, 0]) & (csv.iloc[i, 1] >= csv.iloc[i, 2]):
            csv.iloc[i, -1] = 1
        # 0 for pre
        elif (csv.iloc[i, 0] > csv.iloc[i, 1]) & (csv.iloc[i, 0] >= csv.iloc[i, 2]):
            csv.iloc[i, -1] = 0
        # 2 for sn
        else:
            csv.iloc[i, -1] = 2
    return csv


def load_training_set(path=args.results_path + args.set_path):
    """
    :param path: default path of csv_set
    :return: the csv for training rf model
    """
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        csv_data = Initialized_csv_set()
        csv_data = Adding_ClassLabels(csv_data)
        csv_set = Adding_TrainingFeatures(csv_data)
        csv_set.to_csv(path, index=False)
        return csv_set


def train_RF_class3(save_path=args.results_path + args.RFresults):
    """
    :return: RF class3 results in csv
    """
    if os.path.exists(save_path):
        return pd.read_csv(save_path)
    else:
        csv_set = load_training_set()
        X = csv_set.copy()

        csv_RFresult = pd.DataFrame()
        for i in range(args.RF_split):
            lon_left = i * args.global_lon / args.RF_split
            lon_right = (i + 1) * args.global_lon / args.RF_split

            train_set = X[~((X.iloc[:, 3] < lon_right) & (X.iloc[:, 3] >= lon_left))]
            test_set = X[(X.iloc[:, 3] < lon_right) & (X.iloc[:, 3] >= lon_left)]

            x_train, y_train = train_set.iloc[:, 7:], train_set.iloc[:, 6]
            x_test, y_test = test_set.iloc[:, 7:], test_set.iloc[:, 6]

            model = RF(n_estimators=200, oob_score=True, n_jobs=-1, random_state=args.seed)
            model.fit(x_train, y_train)

            rf_score = model.predict_proba(x_test)
            rf_pred = np.argmax(rf_score, axis=1)
            test_set['rf_class3'] = rf_pred
            test_set['rf3_score'] = rf_score[:, 0]
            test_set['rf3_score1'] = rf_score[:, 1]
            test_set['rf3_score2'] = rf_score[:, 2]
            csv_RFresult = pd.concat((csv_RFresult, test_set), axis=0, ignore_index=True)
            print('\rProcessing: ' + str(i), end="")

        # save
        csv_save = pd.DataFrame()
        csv_save['lat'] = csv_RFresult.iloc[:, 4]
        csv_save['lon'] = csv_RFresult.iloc[:, 3]
        csv_save['class3'] = csv_RFresult.iloc[:, 6]
        csv_save['rf_class3'] = csv_RFresult['rf_class3']
        csv_save['rf3_score0'] = csv_RFresult['rf3_score0']
        csv_save['rf3_score1'] = csv_RFresult['rf3_score1']
        csv_save['rf3_score2'] = csv_RFresult['rf3_score2']
        csv_save.to_csv(save_path, index=False)
        return csv_save


def train_RF_class8(save_path=args.results_path + args.RFresults8):
    """
    :return: RF class8 results in csv
    """
    if os.path.exists(save_path):
        return pd.read_csv(save_path)
    else:
        csv_set = load_training_set()
        csv_class8 = pd.DataFrame()
        csv_class8['pre'] = csv_set.iloc[:, 0]
        csv_class8['sm'] = csv_set.iloc[:, 1]
        csv_class8['sn'] = csv_set.iloc[:, 2]
        csv_class8['original_class8'] = csv_set.iloc[:, 5]  # label

        class8_group = csv_class8.groupby('original_class8')
        class8_group.size()

        # Table 2
        for i in range(8):
            df_group = class8_group.get_group(i)
            print(df_group.min(), '\n',
                  df_group.mean(), '\n',
                  df_group.max(), '\n',
                  len(df_group), '\n\n')

        # Here, we reset the index for better plot (0-2 for P-type, 3-6 for M-type, 7 for N-type)
        csv_class8['class8'] = csv_class8['original_class8'].values
        for i in range(len(csv_class8)):
            if csv_class8.iloc[i, -2] == 0:
                csv_class8.iloc[i, -1] = 6

            if csv_class8.iloc[i, -2] == 1:
                csv_class8.iloc[i, -1] = 3

            if csv_class8.iloc[i, -2] == 3:
                csv_class8.iloc[i, -1] = 1

            if csv_class8.iloc[i, -2] == 4:
                csv_class8.iloc[i, -1] = 0

            if csv_class8.iloc[i, -2] == 5:
                csv_class8.iloc[i, -1] = 7

            if csv_class8.iloc[i, -2] == 6:
                csv_class8.iloc[i, -1] = 4

            if csv_class8.iloc[i, -2] == 7:
                csv_class8.iloc[i, -1] = 5

        csv_set['class8'] = csv_class8['class8'].values
        X = csv_set.copy()
        X = X.drop(columns=['5', '6'])
        csv_classRF8 = pd.DataFrame()

        for i in range(args.RF_split):
            lon_left = i * args.global_lon / args.RF_split
            lon_right = (i + 1) * args.global_lon / args.RF_split

            train_set = X[~((X.iloc[:, 3] < lon_right) & (X.iloc[:, 3] >= lon_left))]
            test_set = X[(X.iloc[:, 3] < lon_right) & (X.iloc[:, 3] >= lon_left)]

            x_train, y_train = train_set.iloc[:, 3:-1], train_set.iloc[:, -1]
            x_test, y_test = test_set.iloc[:, 3:-1], test_set.iloc[:, -1]

            model = RF(n_estimators=200, oob_score=True, n_jobs=-1, random_state=args.seed)
            model.fit(x_train, y_train)

            rf_pred = model.predict(x_test)
            test_set['rf_class8'] = rf_pred

            csv_classRF8 = pd.concat((csv_classRF8, test_set), axis=0, ignore_index=True)
            print('\rProcessing: ' + str(i), end="")

        # Table 3
        print(classification_report(csv_classRF8['class8'], csv_classRF8['rf_class8'], digits=3))

        # save
        csv_save = pd.DataFrame()
        csv_save['lat'] = csv_classRF8.iloc[:, 4]
        csv_save['lon'] = csv_classRF8.iloc[:, 3]
        csv_save['class8'] = csv_classRF8['class8']
        csv_save['rf_class8'] = csv_classRF8['rf_class8']
        csv_save.to_csv(save_path, index=False)
        return csv_save


def find_thresholds(csv_RFresult):
    # 查找最佳阈值的数组
    thresholds = np.arange(0.0, 1.0, 0.01)
    fscore = np.zeros(shape=(len(thresholds)))
    print('Length of sequence: {}'.format(len(thresholds)))

    # 拟合模型
    for index, elem in enumerate(thresholds):
        # 修正概率
        y_pred_prob = (csv_RFresult['rf3_score1'] > elem).astype('int')
        # 计算f值
        fscore[index] = f1_score(csv_RFresult['class3'] == 1, y_pred_prob)
        print('\r当前进度: ' + str(index), end="")

    # 查找最佳阈值
    index = np.argmax(fscore)
    thresholdOpt = round(thresholds[index], ndigits=4)
    fscoreOpt = round(fscore[index], ndigits=4)
    print('Best Threshold: {} with F-Score: {}'.format(thresholdOpt, fscoreOpt))

    threshold = [0.35, 0.43, 0.23]

    csv_RFresult.iloc[csv_RFresult['rf3_score2'] > 0.21, -5] = 2
    csv_RFresult.iloc[csv_RFresult['rf3_score0'] > 0.35, -5] = 0
    csv_RFresult.iloc[csv_RFresult['rf3_score1'] > 0.43, -5] = 1


def load_matrix(class_name='class3', csv_path=args.results_path + args.RFresults,
                save_path=args.results_path + args.matrix3):
    """
    :param class_name: 'class3' or 'class8'
    :param csv_path: csv_RF_results
    :param save_path: save path
    :return: Transform csv to the matrix of class3 or class8 using for plot
    """
    if os.path.exists(save_path):
        matrix = np.load(save_path)
        return matrix[:, :, 0], matrix[:, :1]
    else:
        if os.path.exists(csv_path):
            csv_RFresult = pd.read_csv(csv_path)
            # initialization
            matrix_class = np.zeros((args.global_lat, args.global_lon))
            matrix_rfclass = np.zeros((args.global_lat, args.global_lon))
            matrix_class[matrix_class == 0] = np.nan
            matrix_rfclass[matrix_rfclass == 0] = np.nan
            # series
            series_class, series_rfclass = csv_RFresult[class_name].values, csv_RFresult['rf_' + class_name].values
            for i in range(len(csv_RFresult)):
                matrix_class[csv_RFresult.iloc[i, 0], csv_RFresult.iloc[i, 1]] = series_class[i]
                matrix_rfclass[csv_RFresult.iloc[i, 0], csv_RFresult.iloc[i, 1]] = series_rfclass[i]
                print('\rProcessing: ' + str(i), end="")

            matrix = np.concatenate((matrix_class[:, :, np.newaxis], matrix_rfclass[:, :, np.newaxis]), axis=2)
            np.save(save_path, matrix)
            return matrix_class, matrix_rfclass
        else:
            print('Please ensure that RFresult path file exists!')


def load_beta(path=args.results_path + args.Beta_path):
    if os.path.exists(path):
        beta = np.load(path)
    else:
        runoff = np.load(args.data_path + args.runoff_idmax_path)
        regions = load_regions()
        beta = np.zeros((args.global_lat, args.global_lon)) * regions
        for i in range(args.global_lat):
            for j in range(args.global_lon):
                if ~np.isnan(regions[i, j]):
                    beta[i, j] = stats.theilslopes(runoff[i, j, :])[0]
                    print('\rProcessing: ' + str(i) + ' - ' + str(j), end="")
        np.save(path, beta)
    return beta


def Pettitt_change_point_detection(inputdata):
    inputdata = np.array(inputdata)
    n = inputdata.shape[0]
    k = range(n)
    inputdataT = pd.Series(inputdata)
    r = inputdataT.rank()
    Uk = [2 * np.sum(r[0:x]) - x * (n + 1) for x in k]
    Uka = list(np.abs(Uk))
    U = np.max(Uka)
    K = Uka.index(U)
    pvalue = 2 * np.exp((-6 * (U ** 2)) / (n ** 3 + n ** 2))
    if pvalue <= 0.05:
        change_point_desc = 1  # 显著
    else:
        change_point_desc = 0  # '不显著'
    return K, change_point_desc, pvalue



