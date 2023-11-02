import os
import pandas as pd
import numpy as np
r_folder = r'C:\Users\mhd01\source\repos\ajegorovs\bubble_process_py\post_tests'
dict_glob = {'HFS 200 mT Series 4':{}} # HFS 125 mT Series 1 ; Field OFF Series 7; VFS 125 mT Series 5; HFS 200 mT Series 4
pref = ["segments", "graphs", "ms-events"]
for p_folder in dict_glob:
    folder0 = os.path.join(r_folder,p_folder)
    dirs0_all = os.listdir(folder0)
    dirs0 = [item for item in dirs0_all if os.path.isdir(os.path.join(folder0, item))]
    
    for s_folder in dirs0:
        N = s_folder[4:7]#'sccm100-meanFix'
        root1 = os.path.join(folder0,s_folder)
        dirs1_all = os.listdir(root1)
        dirs1 = [item for item in dirs1_all if os.path.isdir(os.path.join(root1, item))]
        #for root1, dirs1, files in os.walk(os.path.join(root0,s_folder)):
        folder1 = dirs1[0]#'00001-03000'
        #K = folder1[:5]
        #L = folder1[-5:]
        root2 = os.path.join(root1,folder1,'archives')
        for root3, dirs3, files3 in os.walk(os.path.join(root2)):
            dict_glob[p_folder][int(N)] = {prefix: os.path.join(root3, file) for file in files3 for prefix in pref if file.startswith(prefix)}


for proj,proj_d in dict_glob.items():
    with pd.ExcelWriter(f'{proj}.xlsx', engine= 'xlsxwriter') as writer: #'xlsxwriter' #'openpyxl'
        for sccm, params_d in proj_d.items():
            max_length = 0
            path_segments = params_d["segments"]
            with open(path_segments, 'rb') as handle:
                segments_d = pickle.load(handle)

            path_graphs = params_d["graphs"]
            with open(path_graphs, 'rb') as handle:
                graphs_d = pickle.load(handle)
        
            families = list(segments_d.keys())
            centroids_fam = []
            for family in families:
                #centroid_fam = []
                G1, G2  = graphs_d[family]
                segments = segments_d[family]
                centroid = lambda node: (G1.nodes[node]['cx'],G1.nodes[node]['cy'])
                for segment in segments:
                    if len(segment) >= 2:
                        centroids = [centroid(node) for node in segment]
                        avg_speed = np.linalg.norm((np.array(centroids[-1]) - np.array(centroids[0]))/len(centroids))
                        if avg_speed > 2:
                            max_length = max(max_length,len(centroids))

                            centroids_fam.append(centroids)
                        else:
                            a = 1
                #centroids.append(centroid_fam)

            a =1
            
            df = pd.DataFrame()
            xy_dataframes = []
            col_0 = list(range(max_length))
            col_0[1] = "pix/cm-ratio"
            col_0[2] = 108.5
            col_0 = pd.Series(col_0)
            xy_dataframes.extend([col_0.rename('y-offset')])
            for i, coord_set in enumerate(centroids_fam):
                x_col = pd.Series([x[0] for x in coord_set])
                y_col = pd.Series([x[1] for x in coord_set])
                xy_dataframes.extend([x_col.rename(f'f_{i}_cx'), y_col.rename(f'f_{i}_cy')])

            df = pd.concat(xy_dataframes, axis=1)
            df.to_excel(writer, sheet_name=str(sccm), index=False)
    #writer.close()
