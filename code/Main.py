from Code_re03.utils import *
from sklearn import metrics
import h5py


def zip_SourceData(path=args.data_path + args.source_path):
    with h5py.File(path, 'w') as f:
        f.create_dataset('runoff', data=np.load(args.data_path+args.runoff_idmax_path),
                         compression="gzip", compression_opts=5)
        f.create_dataset('pre', data=np.load(args.data_path + args.pre_idmax_path),
                         compression="gzip", compression_opts=5)
        f.create_dataset('soil', data=np.load(args.data_path + args.soil_idmax_path),
                         compression="gzip", compression_opts=5)
        f.create_dataset('snow', data=np.load(args.data_path + args.snow_idmax_path),
                         compression="gzip", compression_opts=5)


def unzip_SourceData(path=args.data_path + args.source_path):
    with h5py.File(path, 'r') as f:
        runoff = f['runoff'][:, :, :]
        np.save(args.data_path + args.runoff_idmax_path, runoff)
        pre = f['pre'][:, :, :]
        np.save(args.data_path + args.pre_idmax_path, pre)
        soil = f['soil'][:, :, :]
        np.save(args.data_path + args.soil_idmax_path, soil)
        snow = f['snow'][:, :, :]
        np.save(args.data_path + args.snow_idmax_path, snow)


def Table_1_RF3():
    csv_RFresult = train_RF_class3(args.results_path + args.RFresults_path)
    print(classification_report(csv_RFresult.iloc[:, 6], csv_RFresult['rf_class3'], digits=3))


def Figure_2_RelativeImportance(Alpha_path=args.results_path + args.Alpha_path,
                                regions_path=args.results_path + args.regions_path):
    if os.path.exists(Alpha_path) & os.path.exists(regions_path):
        Alpha = np.load(Alpha_path)
        regions = np.load(regions_path)
        pre_alpha, soil_alpha, snow_alpha = Alpha[:, :, 0] * regions, Alpha[:, :, 1] * regions, Alpha[:, :, 2] * regions

        print('The mean of alpha precipitation: ', np.nanmean(pre_alpha), '\n',
              'There are {} % areas >= 0.8'.format((pre_alpha >= 0.8).sum() / args.grids), '\n',
              'There are {} % areas <= 0.2'.format((pre_alpha <= 0.2).sum() / args.grids), '\n\n',

              'The mean of alpha soil moisture: ', np.nanmean(soil_alpha), '\n',
              'There are {} % areas >= 0.8'.format((soil_alpha >= 0.8).sum() / args.grids), '\n',
              'There are {} % areas <= 0.2'.format((soil_alpha <= 0.2).sum() / args.grids), '\n\n',

              'The mean of alpha snow melt: ', np.nanmean(snow_alpha), '\n',
              'There are {} % areas >= 0.8'.format((snow_alpha >= 0.8).sum() / args.grids), '\n',
              'There are {} % areas <= 0.2'.format((snow_alpha <= 0.2).sum() / args.grids), '\n\n',
              )

        global_plt(pre_alpha)
        global_plt(soil_alpha)
        global_plt(snow_alpha)
    else:
        print('Please ensure that Alpha/region path file exists or use "python Calculate_RelativeImportance.py"')


def Figure_3_FloodTiming(path=args.data_path + args.runoff_idmax_path):
    def days_to_months(data):
        data[data <= 31] = 1
        data[data > 334] = 12
        data[data > 304] = 11
        data[data > 273] = 10
        data[data > 243] = 9
        data[data > 212] = 8
        data[data > 181] = 7
        data[data > 151] = 6
        data[data > 120] = 5
        data[data > 90] = 4
        data[data > 59] = 3
        data[data > 31] = 2
        return data

    runoff = np.load(path)
    runoff[runoff == 0] = np.nan
    runoff = np.nanmean(runoff, axis=2)
    runoff = days_to_months(runoff)
    global_plt(runoff, 'timing')


def Figure_4_FloodTrends():
    trends = load_beta() * 10
    global_plt(trends, 'trend')

    print('late: ' + str((trends > 0).sum()))
    print(trends[trends > 0].mean())
    print('early: ' + str((trends < 0).sum()))
    print(trends[trends < 0].mean())


def Figure_5_class3():
    matrix_class, matrix_rfclass = load_matrix()
    global_plt(matrix_class + 0.5, 'class3')
    global_plt(matrix_rfclass + 0.5, 'class3')


def Figure_6_ClusterScore(path=args.results_path + args.cluster_score):
    if os.path.exists(path):
        score = np.load(path)
    else:
        c_score = []
        s_score = []
        csv_set = Initialized_csv_set()
        X_train = csv_set[['pre', 'sm', 'sn']]
        for k in np.arange(3, 11):
            k_model = KMeans(n_clusters=k, random_state=1).fit(X_train)
            labels = k_model.labels_
            c_score.append(metrics.calinski_harabasz_score(X_train, labels))
            s_score.append(metrics.silhouette_score(X_train, labels, metric='euclidean'))
            print('\rProcessing: ' + str(k), end="")

        score = np.concatenate((c_score, s_score))
        np.save(path, score)

    # Fig.6 plot c_score and s_score
    SC = score[8:]
    CI = score[:8]

    def score_plot(y1=SC, y2=CI, x=np.arange(3, 11)):
        fig, ax1 = plt.subplots(figsize=(20, 15))

        ax1.plot(x, y1, color="blue", linewidth=3, label="SC")
        ax1.set_xlabel("classses")
        ax1.set_ylabel("SC")
        ax1.set_ylim([0.2, 0.6])

        ax2 = ax1.twinx()
        ax2.plot(x, y2, color="red", linewidth=3, label="CI")
        ax2.set_ylabel("CI")
        ax2.set_ylim([200000, 400000])

        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
        font = {'family': 'monospace',
                'weight': 'normal',
                'size': 30}
        plt.rc('font', **font)  # pass in the font dict as kwargs

    score_plot()


def Figure_7_class8():
    matrix_class, matrix_rfclass = load_matrix(class_name='class8', csv_path=args.results_path + args.RFresults8,
                                               save_path=args.results_path + args.matrix8)
    global_plt(matrix_class + 0.5, 'class8')
    global_plt(matrix_rfclass + 0.5, 'class8')
