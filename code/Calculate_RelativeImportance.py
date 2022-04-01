from utils import *
from config import *

############ load basic data  #####################
snow = np.load(args.data_path + args.snow_idmax_path)
runoff = np.load(args.data_path + args.runoff_idmax_path)
pre = np.load(args.data_path + args.pre_idmax_path)
soil = np.load(args.data_path + args.soil_idmax_path)
regions = load_regions()

############## initialized ###########################
alpha_pre, alpha_sm, alpha_sn = np.zeros((args.global_lat, args.global_lon)), \
                                np.zeros((args.global_lat, args.global_lon)), \
                                np.zeros((args.global_lat, args.global_lon))

for i in range(args.global_lat):
    for j in range(args.global_lon):
        pre1, soil1, snow1, runoff1 = pre[i, j, :], soil[i, j, :], snow[i, j, :], runoff[i, j, :]

        # 洪水资料需要有至少20年
        if ~np.isnan(regions[i, j]):

            # 返回数值不为0的年份
            n_index = np.nonzero(runoff1)
            pre1, soil1, snow1, runoff1 = pre1[n_index], soil1[n_index], snow1[n_index], runoff1[n_index]
            pre1, soil1, snow1 = pre1[np.nonzero(pre1)], soil1[np.nonzero(soil1)], snow1[np.nonzero(snow1)]

            pre_sin1 = np.mean(np.sin(pre1 / 365 * 2 * np.pi))
            soil_sin1 = np.mean(np.sin(soil1 / 365 * 2 * np.pi))
            snow_sin1 = np.mean(np.sin(snow1 / 365 * 2 * np.pi))
            runoff_sin1 = np.mean(np.sin(runoff1 / 365 * 2 * np.pi))
            pre_cos1 = np.mean(np.cos(pre1 / 365 * 2 * np.pi))
            soil_cos1 = np.mean(np.cos(soil1 / 365 * 2 * np.pi))
            snow_cos1 = np.mean(np.cos(snow1 / 365 * 2 * np.pi))
            runoff_cos1 = np.mean(np.cos(runoff1 / 365 * 2 * np.pi))

            # 是否有一半以上的融雪数据
            if len(snow1) < len(runoff1) / 2:
                condition = 2
            else:
                condition = 1
                # 判断洪水的点是否在三角形内部
                if ~IsInsideTriangle(np.array([pre_cos1, pre_sin1]), np.array([soil_cos1, soil_sin1]),
                                     np.array([snow_cos1, snow_sin1]), np.array([runoff_cos1, runoff_sin1])):
                    # 如果不在内部，将用三角形上与洪水最近的点代替原洪水点
                    new_point = point_triangle(np.array([runoff_cos1, runoff_sin1]),
                                               np.array([pre_cos1, pre_sin1]),
                                               np.array([soil_cos1, soil_sin1]),
                                               np.array([snow_cos1, snow_sin1]))
                    runoff_cos1, runoff_sin1 = new_point[0], new_point[1]

            alpha_pre[i, j], alpha_sm[i, j], alpha_sn[i, j] = Relative_Importance(condition, pre_sin1, soil_sin1,
                                                                                  snow_sin1, runoff_sin1,
                                                                                  pre_cos1, soil_cos1,
                                                                                  snow_cos1, runoff_cos1)

            print('\rProcessing on: ' + str(i) + ' -- ' + str(j), end="")

###############  save results ############################
Alpha_data = np.concatenate((alpha_pre[:, :, np.newaxis], alpha_sm[:, :, np.newaxis],
                             alpha_sn[:, :, np.newaxis]), axis=2)
np.save(args.results_path + args.Alpha_path, Alpha_data)
