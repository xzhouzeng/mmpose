import numpy as np

def get_skeleton(dataset="TopDownCocoDataset"):
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255],
                        [255, 0, 0], [255, 255, 255]])

    if dataset in ('TopDownCocoDataset', 'BottomUpCocoDataset',
                    'TopDownOCHumanDataset', 'AnimalMacaqueDataset'):
            # show the results
            skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                        [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                        [3, 5], [4, 6]]

            pose_link_color = palette[[
                0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
            ]]
            pose_kpt_color = palette[[
                16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
            ]]

    elif dataset == 'TopDownCocoWholeBodyDataset':
        # show the results
        skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                    [8, 10], [1, 2], [0, 1], [0, 2],
                    [1, 3], [2, 4], [3, 5], [4, 6], [15, 17], [15, 18],
                    [15, 19], [16, 20], [16, 21], [16, 22], [91, 92],
                    [92, 93], [93, 94], [94, 95], [91, 96], [96, 97],
                    [97, 98], [98, 99], [91, 100], [100, 101], [101, 102],
                    [102, 103], [91, 104], [104, 105], [105, 106],
                    [106, 107], [91, 108], [108, 109], [109, 110],
                    [110, 111], [112, 113], [113, 114], [114, 115],
                    [115, 116], [112, 117], [117, 118], [118, 119],
                    [119, 120], [112, 121], [121, 122], [122, 123],
                    [123, 124], [112, 125], [125, 126], [126, 127],
                    [127, 128], [112, 129], [129, 130], [130, 131],
                    [131, 132]]

        pose_link_color = palette[[
            0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
        ] + [16, 16, 16, 16, 16, 16] + [
            0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
            16
        ] + [
            0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
            16
        ]]
        pose_kpt_color = palette[
            [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0] +
            [0, 0, 0, 0, 0, 0] + [19] * (68 + 42)]

    elif dataset == 'TopDownAicDataset':
        skeleton = [[2, 1], [1, 0], [0, 13], [13, 3], [3, 4], [4, 5],
                    [8, 7], [7, 6], [6, 9], [9, 10], [10, 11], [12, 13],
                    [0, 6], [3, 9]]

        pose_link_color = palette[[
            9, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 0, 7, 7
        ]]
        pose_kpt_color = palette[[
            9, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 0, 0
        ]]

    elif dataset == 'TopDownMpiiDataset':
        skeleton = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [6, 7],
                    [7, 8], [8, 9], [8, 12], [12, 11], [11, 10], [8, 13],
                    [13, 14], [14, 15]]

        pose_link_color = palette[[
            16, 16, 16, 16, 16, 16, 7, 7, 0, 9, 9, 9, 9, 9, 9
        ]]
        pose_kpt_color = palette[[
            16, 16, 16, 16, 16, 16, 7, 7, 0, 0, 9, 9, 9, 9, 9, 9
        ]]

    elif dataset == 'TopDownMpiiTrbDataset':
        skeleton = [[12, 13], [13, 0], [13, 1], [0, 2], [1, 3], [2, 4],
                    [3, 5], [0, 6], [1, 7], [6, 7], [6, 8], [7,
                                                            9], [8, 10],
                    [9, 11], [14, 15], [16, 17], [18, 19], [20, 21],
                    [22, 23], [24, 25], [26, 27], [28, 29], [30, 31],
                    [32, 33], [34, 35], [36, 37], [38, 39]]

        pose_link_color = palette[[16] * 14 + [19] * 13]
        pose_kpt_color = palette[[16] * 14 + [0] * 26]

    elif dataset in ('OneHand10KDataset', 'FreiHandDataset',
                    'PanopticDataset'):
        skeleton = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7],
                    [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13],
                    [13, 14], [14, 15], [15, 16], [0, 17], [17, 18],
                    [18, 19], [19, 20]]

        pose_link_color = palette[[
            0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
            16
        ]]
        pose_kpt_color = palette[[
            0, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16,
            16, 16
        ]]

    elif dataset == 'InterHand2DDataset':
        skeleton = [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7], [8, 9],
                    [9, 10], [10, 11], [12, 13], [13, 14], [14, 15],
                    [16, 17], [17, 18], [18, 19], [3, 20], [7, 20],
                    [11, 20], [15, 20], [19, 20]]

        pose_link_color = palette[[
            0, 0, 0, 4, 4, 4, 8, 8, 8, 12, 12, 12, 16, 16, 16, 0, 4, 8, 12,
            16
        ]]
        pose_kpt_color = palette[[
            0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
            16, 0
        ]]

    elif dataset == 'Face300WDataset':
        # show the results
        skeleton = []

        pose_link_color = palette[[]]
        pose_kpt_color = palette[[19] * 68]
        kpt_score_thr = 0

    elif dataset == 'FaceAFLWDataset':
        # show the results
        skeleton = []

        pose_link_color = palette[[]]
        pose_kpt_color = palette[[19] * 19]
        kpt_score_thr = 0

    elif dataset == 'FaceCOFWDataset':
        # show the results
        skeleton = []

        pose_link_color = palette[[]]
        pose_kpt_color = palette[[19] * 29]
        kpt_score_thr = 0

    elif dataset == 'FaceWFLWDataset':
        # show the results
        skeleton = []

        pose_link_color = palette[[]]
        pose_kpt_color = palette[[19] * 98]
        kpt_score_thr = 0

    elif dataset == 'AnimalHorse10Dataset':
        skeleton = [[0, 1], [1, 12], [12, 16], [16, 21], [21, 17],
                    [17, 11], [11, 10], [10, 8], [8, 9], [9, 12], [2, 3],
                    [3, 4], [5, 6], [6, 7], [13, 14], [14, 15], [18, 19],
                    [19, 20]]

        pose_link_color = palette[[4] * 10 + [6] * 2 + [6] * 2 + [7] * 2 +
                                [7] * 2]
        pose_kpt_color = palette[[
            4, 4, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 7, 7, 7, 4, 4, 7, 7, 7,
            4
        ]]

    elif dataset == 'AnimalFlyDataset':
        skeleton = [[1, 0], [2, 0], [3, 0], [4, 3], [5, 4], [7, 6], [8, 7],
                    [9, 8], [11, 10], [12, 11], [13, 12], [15, 14],
                    [16, 15], [17, 16], [19, 18], [20, 19], [21, 20],
                    [23, 22], [24, 23], [25, 24], [27, 26], [28, 27],
                    [29, 28], [30, 3], [31, 3]]

        pose_link_color = palette[[0] * 25]
        pose_kpt_color = palette[[0] * 32]

    elif dataset == 'AnimalLocustDataset':
        skeleton = [[1, 0], [2, 1], [3, 2], [4, 3], [6, 5], [7, 6], [9, 8],
                    [10, 9], [11, 10], [13, 12], [14, 13], [15, 14],
                    [17, 16], [18, 17], [19, 18], [21, 20], [22, 21],
                    [24, 23], [25, 24], [26, 25], [28, 27], [29, 28],
                    [30, 29], [32, 31], [33, 32], [34, 33]]

        pose_link_color = palette[[0] * 26]
        pose_kpt_color = palette[[0] * 35]

    elif dataset == 'AnimalZebraDataset':
        skeleton = [[1, 0], [2, 1], [3, 2], [4, 2], [5, 7], [6, 7], [7, 2],
                    [8, 7]]

        pose_link_color = palette[[0] * 8]
        pose_kpt_color = palette[[0] * 9]

    elif dataset in 'AnimalPoseDataset':
        skeleton = [[0, 1], [0, 2], [1, 3], [0, 4], [1, 4], [4, 5], [5, 7],
                    [6, 7], [5, 8], [8, 12], [12, 16], [5, 9], [9, 13],
                    [13, 17], [6, 10], [10, 14], [14, 18], [6, 11],
                    [11, 15], [15, 19]]

        pose_link_color = palette[[0] * 20]
        pose_kpt_color = palette[[0] * 20]
    else:
        NotImplementedError()

    return skeleton,pose_kpt_color,pose_link_color
